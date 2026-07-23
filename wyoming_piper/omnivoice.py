"""ONNX OmniVoice text-to-speech backend.

This wraps the OmniVoice inference pipeline (text tokenizer, audio codec, and
the iterative MaskGIT decode loop from the ``omnivoice`` package) but replaces
the transformer forward pass with an ONNX graph run under onnxruntime, which is
the fastest OmniVoice path on CPU.

The ONNX graph exports only the LM forward (embeddings -> Qwen3 backbone ->
audio heads). Everything else -- tokenization, reference-audio encoding, output
decoding, and the sampling loop -- still runs through the torch pipeline, so the
full ``omnivoice`` package and torch are required.

The graph is a block-wise **int4** re-quantization of the fp32 export (see
``script/quantize_omnivoice.py``). The public per-tensor int8 export is lossy
enough to need ~2x the diffusion steps for clean audio; block-wise int4 stays
clean at low step counts (e.g. ``--omnivoice-steps 10``) while running fast on
CPU. The int4 quantization is hardcoded for now.
"""

import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

_LOGGER = logging.getLogger(__name__)

# Repo with the LM-only int4 ONNX export. Override with --omnivoice-onnx-repo.
# Produce/host this with script/quantize_omnivoice.py.
ONNX_REPO = "rhasspy/omnivoice-onnx"
ONNX_FILE = "omnivoice.int4.onnx"
ONNX_DATA_FILE = "omnivoice.int4.onnx.data"
ONNX_DIR = "omnivoice"

# Repo with the full pipeline (tokenizer + audio codec + weights).
PIPELINE_REPO = "k2-fsa/OmniVoice"

# Voice name for the built-in (no-reference) speaker. Also used for an empty
# voice name. Reserved: cannot be a cloning voice.
DEFAULT_VOICE_NAME = "default"


def advertise_language(code: str) -> str:
    """Format a language for the Wyoming Info in the BCP-47 form HA expects.

    OmniVoice IDs are already bare codes (``en``); a regional reference-voice
    directory name like ``en_US`` becomes ``en-US``.
    """
    return code.replace("_", "-")


def get_supported_languages() -> List[str]:
    """All languages OmniVoice supports, as HA-facing codes (e.g. 'en', 'zh')."""
    from omnivoice.utils.lang_map import LANG_IDS

    return sorted(advertise_language(code) for code in LANG_IDS)


def _normalize_language(language: Optional[str]) -> Optional[str]:
    """Map a Wyoming/HA language code to something OmniVoice recognizes.

    OmniVoice accepts an ID (``en``) or a full name (``English``) but not
    locale codes like ``en-US`` / ``en_US``, which it silently treats as
    language-agnostic. Pass the raw value through when it already resolves
    (this preserves the hyphenated language *names* OmniVoice supports);
    otherwise strip the region subtag (``en_US`` -> ``en``).
    """
    if not language or language.lower() == "none":
        return language

    from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

    if language in LANG_IDS or language.lower() in LANG_NAME_TO_ID:
        return language

    primary = re.split(r"[-_]", language, maxsplit=1)[0].lower()
    if primary in LANG_IDS:
        return primary

    return language  # unknown; OmniVoice will warn and go language-agnostic


@dataclass
class OmniVoiceRef:
    """A voice discovered under --omnivoice-ref-dir.

    Two kinds are supported:

    * **Cloning** — ``ref_audio`` + ``ref_text`` are set (from ``ref.wav`` and
      ``ref.txt``); the voice is cloned from the reference recording.
    * **Voice design** — ``instruct`` is set (from ``instruct.txt``); the text
      describes the desired voice style and no reference audio is used.
    """

    name: str
    language: str  # directory name; advertised as-is, normalized at synth time
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    instruct: Optional[str] = None


def scan_ref_dir(ref_dir: Union[str, Path]) -> List[OmniVoiceRef]:
    """Discover voices under ``<ref_dir>/<language>/<voice>/``.

    Each voice directory is one of two kinds:

    * **Cloning** — contains ``ref.wav`` + ``ref.txt``.
    * **Voice design** — contains ``instruct.txt`` (whose text describes the
      desired voice style). Only used when there is no ``ref.wav``/``ref.txt``.

    Voice names must be unique; duplicates are skipped with a warning.
    """
    voices: List[OmniVoiceRef] = []
    base = Path(ref_dir)
    if not base.is_dir():
        _LOGGER.warning("OmniVoice ref dir not found: %s", base)
        return voices

    seen: dict = {}
    for lang_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for voice_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            wav = voice_dir / "ref.wav"
            txt = voice_dir / "ref.txt"
            instruct_txt = voice_dir / "instruct.txt"

            is_clone = wav.is_file() and txt.is_file()
            is_instruct = instruct_txt.is_file()
            if not (is_clone or is_instruct):
                continue

            name = voice_dir.name
            if name == DEFAULT_VOICE_NAME:
                _LOGGER.warning(
                    "Ignoring voice %r: name is reserved for the built-in speaker",
                    name,
                )
                continue
            if name in seen:
                _LOGGER.warning(
                    "Duplicate OmniVoice voice %r (%s/ and %s/); keeping first",
                    name,
                    seen[name],
                    lang_dir.name,
                )
                continue

            # Prefer cloning when both a reference recording and an instruct
            # file are present.
            if is_clone:
                try:
                    ref_text = txt.read_text(encoding="utf-8").strip()
                except OSError as err:
                    _LOGGER.warning(
                        "Skipping voice %r: cannot read %s (%s)", name, txt, err
                    )
                    continue
                voice = OmniVoiceRef(
                    name=name,
                    language=lang_dir.name,
                    ref_audio=str(wav),
                    ref_text=ref_text,
                )
            else:
                try:
                    instruct = instruct_txt.read_text(encoding="utf-8").strip()
                except OSError as err:
                    _LOGGER.warning(
                        "Skipping voice %r: cannot read %s (%s)",
                        name,
                        instruct_txt,
                        err,
                    )
                    continue
                if not instruct:
                    _LOGGER.warning(
                        "Skipping voice %r: %s is empty", name, instruct_txt
                    )
                    continue
                voice = OmniVoiceRef(
                    name=name,
                    language=lang_dir.name,
                    instruct=instruct,
                )

            seen[name] = lang_dir.name
            voices.append(voice)

    _LOGGER.info("Loaded %d OmniVoice reference voice(s) from %s", len(voices), base)
    return voices


def _find_local_onnx(
    data_dirs: Optional[Iterable[Union[str, Path]]],
) -> Optional[str]:
    """Return a data-dir copy of the ONNX graph if both files are present.

    Looks for the ONNX_FILE / ONNX_DATA_FILE basenames directly in each data
    directory (the external weight data must sit next to the graph).
    """
    onnx_name = Path(ONNX_FILE).name
    data_name = Path(ONNX_DATA_FILE).name
    for data_dir in data_dirs or []:
        onnx_path = Path(data_dir) / ONNX_DIR / onnx_name
        data_path = Path(data_dir) / ONNX_DIR / data_name
        if onnx_path.exists() and data_path.exists():
            return str(onnx_path)
    return None


def ensure_omnivoice_downloaded(
    local_files_only: bool = False,
    onnx_repo: Optional[str] = None,
    data_dirs: Optional[Iterable[Union[str, Path]]] = None,
) -> str:
    """Ensure the ONNX graph and pipeline model are available.

    If the ONNX graph (and its ``.data``) are found in one of ``data_dirs``,
    that local copy is used and only the pipeline model is fetched. Otherwise
    the graph is downloaded from HuggingFace (``onnx_repo`` overrides the repo).
    The HF cache location is controlled by the ``HF_HOME`` environment variable,
    which the caller sets from ``--download-dir``.

    Returns the local path to the ONNX graph file.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    onnx_path = _find_local_onnx(data_dirs)
    if onnx_path is not None:
        _LOGGER.info("Using local OmniVoice ONNX model: %s", onnx_path)
    else:
        repo = onnx_repo or ONNX_REPO
        _LOGGER.debug("Downloading OmniVoice ONNX model from %s", repo)
        onnx_path = hf_hub_download(repo, ONNX_FILE, local_files_only=local_files_only)
        # External weight data must sit next to the .onnx graph for onnxruntime.
        hf_hub_download(repo, ONNX_DATA_FILE, local_files_only=local_files_only)

    _LOGGER.debug("Ensuring OmniVoice pipeline model is available")
    snapshot_download(PIPELINE_REPO, local_files_only=local_files_only)

    return onnx_path


class OmniVoiceModel:
    """Loaded OmniVoice model that synthesizes into a wave writer."""

    def __init__(
        self,
        onnx_path: str,
        num_step: int = 32,
        default_language: str = "English",
        local_files_only: bool = False,
    ) -> None:
        import types

        import numpy as np
        import onnxruntime as ort
        import torch
        from omnivoice.models.omnivoice import OmniVoice, OmniVoiceModelOutput

        self._np = np
        self._torch = torch
        self.num_step = num_step
        self.default_language = default_language
        self._prompt_cache: dict = {}  # ref_audio path -> VoiceClonePrompt

        _LOGGER.debug("Loading OmniVoice pipeline (torch, fp32, cpu)")
        model = OmniVoice.from_pretrained(
            PIPELINE_REPO,
            device_map="cpu",
            dtype=torch.float32,
            local_files_only=local_files_only,
        )
        model.eval()

        _LOGGER.debug("Loading ONNX LM graph: %s", onnx_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            onnx_path, sess_options, providers=["CPUExecutionProvider"]
        )
        input_names = {i.name for i in session.get_inputs()}

        def onnx_forward(
            _self,
            input_ids,  # type: ignore[no-untyped-def]
            audio_mask,
            labels=None,
            attention_mask=None,
            document_ids=None,
            position_ids=None,
        ):
            """Drop-in replacement for OmniVoice.forward using onnxruntime.

            The ONNX graph takes a 2D padding mask (bidirectional attention),
            while the torch pipeline builds a 4D block mask -- so derive the 2D
            real-token mask from the 4D mask's rows.
            """
            batch, _codebooks, seq = input_ids.shape
            if attention_mask is not None and attention_mask.dim() == 4:
                # A real key position is a row that attends to more than itself.
                real = attention_mask[:, 0, :, :].sum(-1) > 1
                attn2d = real.to(torch.int64)
            elif attention_mask is not None and attention_mask.dim() == 2:
                attn2d = attention_mask.to(torch.int64)
            else:
                attn2d = torch.ones(batch, seq, dtype=torch.int64)

            if position_ids is None:
                position_ids = (
                    torch.arange(seq, dtype=torch.int64)
                    .unsqueeze(0)
                    .expand(batch, seq)
                    .contiguous()
                )

            feeds = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "audio_mask": audio_mask.cpu().numpy().astype(bool),
                "attention_mask": attn2d.cpu().numpy().astype(np.int64),
                "position_ids": position_ids.cpu().numpy().astype(np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in input_names}
            logits = session.run(["logits"], feeds)[0]
            return OmniVoiceModelOutput(logits=torch.from_numpy(logits))

        model.forward = types.MethodType(onnx_forward, model)

        self._model = model
        self._session = session
        self.sampling_rate: int = int(model.sampling_rate)
        _LOGGER.info(
            "OmniVoice loaded (num_step=%s, sample_rate=%s)",
            num_step,
            self.sampling_rate,
        )

    def _voice_clone_prompt(self, ref_audio: str, ref_text: Optional[str]):
        """Get the (cached) voice-clone prompt for a reference audio file.

        The encoded reference (RVQ codes) is cached in memory and, next to the
        WAV, as ``ref.rvq`` -- built lazily on first use and regenerated when the
        WAV is newer than the cached file. Avoids re-encoding on every request.
        """
        torch = self._torch
        from omnivoice.models.omnivoice import VoiceClonePrompt

        cached = self._prompt_cache.get(ref_audio)
        if cached is not None:
            return cached

        wav_path = Path(ref_audio)
        rvq_path = wav_path.with_suffix(".rvq")

        prompt = None
        if rvq_path.is_file() and rvq_path.stat().st_mtime >= wav_path.stat().st_mtime:
            try:
                data = torch.load(rvq_path, map_location="cpu", weights_only=False)
                prompt = VoiceClonePrompt(
                    ref_audio_tokens=data["ref_audio_tokens"],
                    ref_text=data["ref_text"],
                    ref_rms=data["ref_rms"],
                )
                _LOGGER.debug("Loaded cached reference codes: %s", rvq_path)
            except Exception as err:  # noqa: BLE001 - stale/foreign cache: rebuild
                _LOGGER.warning("Ignoring bad %s (%s); regenerating", rvq_path, err)
                prompt = None

        if prompt is None:
            _LOGGER.debug("Encoding reference audio: %s", ref_audio)
            prompt = self._model.create_voice_clone_prompt(ref_audio, ref_text)
            try:
                torch.save(
                    {
                        "ref_audio_tokens": prompt.ref_audio_tokens,
                        "ref_text": prompt.ref_text,
                        "ref_rms": prompt.ref_rms,
                    },
                    rvq_path,
                )
                _LOGGER.debug("Cached reference codes: %s", rvq_path)
            except OSError as err:
                _LOGGER.warning("Could not write %s (%s)", rvq_path, err)

        self._prompt_cache[ref_audio] = prompt
        return prompt

    def synthesize_wav(
        self,
        text: str,
        wav_writer: wave.Wave_write,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Synthesize ``text`` and write 16-bit PCM into ``wav_writer``.

        Three mutually exclusive voice modes: ``ref_audio``/``ref_text`` select a
        cloning voice for this request (the encoded reference is cached, see
        :meth:`_voice_clone_prompt`); ``instruct`` selects voice-design mode,
        where the text describes the desired voice style; with neither, the
        built-in OmniVoice speaker is used. ``language`` overrides the default
        (OmniVoice is multilingual, so it is a per-request property).
        """
        torch = self._torch
        np = self._np

        kwargs = dict(
            text=text,
            language=_normalize_language(language or self.default_language),
            num_step=self.num_step,
        )
        if ref_audio:
            kwargs["voice_clone_prompt"] = self._voice_clone_prompt(ref_audio, ref_text)
        elif instruct:
            kwargs["instruct"] = instruct

        with torch.no_grad():
            audios = self._model.generate(**kwargs)

        audio = audios[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")

        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(self.sampling_rate)
        wav_writer.writeframes(pcm.tobytes())
