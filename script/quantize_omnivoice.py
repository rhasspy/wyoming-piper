#!/usr/bin/env python3
"""Produce a compact OmniVoice ONNX model for the omnivoice backend.

The public quantized OmniVoice ONNX exports use per-tensor int8, which is lossy
enough to need ~2x the diffusion steps for clean audio. This script re-quantizes
the *fp32* graph to block-wise int4 (MatMulNBits, block_size=32) — comparable to
llama.cpp's Q4_K — which stays clean at low step counts while running fast on CPU.

The Qwen3 token-embedding table is large (~620 MB fp32) and is a Gather, not a
MatMul, so MatMulNBits leaves it alone. By default it is downcast to fp16
(near-lossless, ~310 MB); ``--int4-embed`` quantizes it to int4 instead (smaller,
~394 MB total, but a small quality cost at low step counts).

The source fp32 graph is the single-graph, bidirectional OmniVoice export
(embeddings + Qwen3 backbone + audio heads) from ``gluschenko/omnivoice-onnx``.
The output keeps the same inputs/outputs, so it is a drop-in replacement.

Usage:
    python script/quantize_omnivoice.py --out omnivoice.int4.onnx

Then upload both files to a HuggingFace repo and point the backend at it:
    hf upload <your-repo> omnivoice.int4.onnx      onnx/omnivoice.int4.onnx
    hf upload <your-repo> omnivoice.int4.onnx.data onnx/omnivoice.int4.onnx.data
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

SRC_REPO = "gluschenko/omnivoice-onnx"
SRC_FILE = "onnx/omnivoice.onnx"  # fp32 graph
SRC_DATA = "onnx/omnivoice.onnx_data"

EMBED_WEIGHT = "model.llm.embed_tokens.weight"


def _save(onnx, model, out: Path, data_name: str) -> None:
    """Save with external data, replacing any existing output files.

    onnx.save_model raises FileExistsError if the external data file already
    exists, so remove stale outputs first.
    """
    out.unlink(missing_ok=True)
    (out.parent / data_name).unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )


def _embed_to_fp16(model) -> None:
    """Downcast the token-embedding table to fp16 and Cast its Gather back."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    for i, t in enumerate(model.graph.initializer):
        if t.name == EMBED_WEIGHT:
            arr = numpy_helper.to_array(t).astype(np.float16)
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, t.name))
            break

    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Gather" and EMBED_WEIGHT in node.input:
            orig_out = node.output[0]
            node.output[0] = orig_out + "_fp16"
            model.graph.node.insert(
                idx + 1,
                helper.make_node(
                    "Cast",
                    [orig_out + "_fp16"],
                    [orig_out],
                    to=TensorProto.FLOAT,
                    name="embed_cast_fp32",
                ),
            )
            break


def _fix_reduce_mean(model) -> None:
    """Move ReduceMean 'axes' attribute to an input (required at opset >= 18).

    Quantizing the Gather bumps the graph to opset 21, but the exporter emitted
    ReduceMean nodes with the pre-18 axes attribute; convert them so the model
    loads.
    """
    import numpy as np
    from onnx import numpy_helper

    axes_inits: dict = {}
    for node in model.graph.node:
        if node.op_type != "ReduceMean":
            continue
        attr = next((a for a in node.attribute if a.name == "axes"), None)
        if attr is None:
            continue
        axes = tuple(attr.ints)
        if axes not in axes_inits:
            name = "rm_axes_" + "_".join(str(x) for x in axes).replace("-", "n")
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(axes, np.int64), name)
            )
            axes_inits[axes] = name
        node.input.append(axes_inits[axes])
        node.attribute.remove(attr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="omnivoice.int4.onnx", help="Output .onnx path")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--fp32", help="Local fp32 .onnx (skip download)")
    ap.add_argument(
        "--int4-embed",
        action="store_true",
        help="Quantize the token embedding to int4 too (smaller, slight quality cost)",
    )
    args = ap.parse_args()

    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        if args.fp32:
            fp32 = args.fp32
        else:
            from huggingface_hub import hf_hub_download

            print(f"Downloading fp32 graph from {SRC_REPO} ...", flush=True)
            src = hf_hub_download(SRC_REPO, SRC_FILE)
            hf_hub_download(SRC_REPO, SRC_DATA)
            # onnx rejects symlinked external data (HF cache uses symlinks),
            # so copy both into a plain directory first.
            fp32 = str(Path(tmp) / "omnivoice.onnx")
            shutil.copy(src, fp32)
            shutil.copy(
                str(Path(src).parent / Path(SRC_DATA).name),
                str(Path(tmp) / Path(SRC_DATA).name),
            )

        print("Loading fp32 model ...", flush=True)
        model = onnx.load(fp32)

        op_types = ("MatMul", "Gather") if args.int4_embed else ("MatMul",)
        print(
            f"Quantizing to int4 (block_size={args.block_size}, "
            f"ops={op_types}) ...",
            flush=True,
        )
        cfg = DefaultWeightOnlyQuantConfig(
            block_size=args.block_size,
            bits=4,
            is_symmetric=False,
            op_types_to_quantize=op_types,
        )
        quantizer = MatMulNBitsQuantizer(model, algo_config=cfg)
        quantizer.process()

        out = Path(args.out)
        data_name = out.name + ".data"
        print(f"Saving {out} (+ {data_name}) ...", flush=True)
        _save(onnx, quantizer.model.model, out, data_name)

    # Post-process on a fresh reload of the saved file (editing the quantizer's
    # live model in place does not persist reliably).
    model = onnx.load(str(out))
    if args.int4_embed:
        _fix_reduce_mean(model)  # Gather quant bumped opset to 21
    else:
        print("Downcasting token embedding to fp16 ...", flush=True)
        _embed_to_fp16(model)
    _save(onnx, model, out, data_name)

    size_mb = (out.stat().st_size + (out.parent / data_name).stat().st_size) / 1e6
    print(f"Done in {time.time() - t0:.0f}s — {out} ({size_mb:.0f} MB total)")


if __name__ == "__main__":
    main()
