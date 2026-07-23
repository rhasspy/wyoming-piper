"""Flask web UI for managing custom Piper and OmniVoice voices.

Runs in a background thread alongside the Wyoming server and exposes a small web
UI (designed to work as a Home Assistant add-on behind ingress) for:

* **Piper** — uploading and deleting custom voices (``<voice>.onnx`` +
  ``<voice>.onnx.json``) under ``--download-dir``, with a little metadata read
  from each config file.
* **OmniVoice** — uploading reference recordings (``ref.wav`` + a required
  transcript) as cloning voices under ``--omnivoice-ref-dir/<language>/<name>/``
  and deleting whole voice directories.

The web UI always functions; each section shows a warning when its backend is
not the one the Wyoming server was started with. File changes take effect for
the Wyoming server only after the Piper integration is reloaded (or Home
Assistant is restarted), so every mutation returns a reminder to that effect.
"""

import argparse
import importlib.util
import json
import logging
import re
import shutil
import threading
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request
from werkzeug.middleware.proxy_fix import ProxyFix

_LOGGER = logging.getLogger(__name__)

# Reminder shown after any change so the user knows it is not live yet.
RELOAD_MESSAGE = (
    "Changes saved. Reload the Piper integration or restart Home Assistant "
    "for them to take effect."
)

# Safe voice / language / directory names: no path separators, no surprises.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Allow large model uploads (Piper "high" voices are ~110 MB).
_MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1 GiB


def _valid_name(name: Optional[str]) -> bool:
    """True if ``name`` is safe to use as a single path component."""
    if not name or name in (".", ".."):
        return False
    return _SAFE_NAME.match(name) is not None


# -----------------------------------------------------------------------------
# Piper custom voices
# -----------------------------------------------------------------------------


def _piper_voice_metadata(config_path: Path) -> Dict[str, Any]:
    """Read a few human-friendly fields from a Piper voice config file."""
    meta: Dict[str, Any] = {}
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError) as err:
        _LOGGER.warning("Could not read Piper config %s: %s", config_path, err)
        meta["error"] = f"Could not read config: {err}"
        return meta

    meta["dataset"] = config.get("dataset")
    audio = config.get("audio", {})
    meta["sample_rate"] = audio.get("sample_rate")
    meta["quality"] = audio.get("quality")
    language = config.get("language", {})
    meta["language"] = language.get("code") or config.get("espeak", {}).get("voice")
    num_speakers = config.get("num_speakers")
    if num_speakers is None and isinstance(config.get("speaker_id_map"), dict):
        num_speakers = len(config["speaker_id_map"]) or None
    meta["num_speakers"] = num_speakers
    return meta


def _list_piper_voices(download_dir: Path, builtin: set) -> List[Dict[str, Any]]:
    """List custom Piper voices (``*.onnx`` + ``*.onnx.json``) in download-dir."""
    voices: List[Dict[str, Any]] = []
    if not download_dir.is_dir():
        return voices

    for onnx_path in sorted(download_dir.glob("*.onnx")):
        name = onnx_path.stem  # e.g. "en_US-lessac-medium"
        config_path = onnx_path.with_name(onnx_path.name + ".json")
        if not config_path.is_file():
            # A model with no config isn't usable; skip it.
            continue

        voice: Dict[str, Any] = {
            "name": name,
            "builtin": name in builtin,
            "size_bytes": onnx_path.stat().st_size,
        }
        voice.update(_piper_voice_metadata(config_path))
        voices.append(voice)

    return voices


# -----------------------------------------------------------------------------
# OmniVoice cloned voices
# -----------------------------------------------------------------------------


def _list_omnivoice_voices(ref_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """List OmniVoice reference voices discovered under the ref dir."""
    if not ref_dir:
        return []

    from .omnivoice import scan_ref_dir

    voices: List[Dict[str, Any]] = []
    for ref in scan_ref_dir(ref_dir):
        voices.append(
            {
                "name": ref.name,
                "language": ref.language,
                "kind": "cloning" if ref.ref_audio else "voice-design",
                "ref_text": ref.ref_text,
                "instruct": ref.instruct,
            }
        )
    return voices


@lru_cache(maxsize=1)
def _omnivoice_lang_ids() -> List[str]:
    """OmniVoice's language IDs, loaded cheaply (no torch import).

    ``import omnivoice.utils.lang_map`` runs the ``omnivoice`` package
    ``__init__``, which pulls in torch/transformers (~6 s). These are only
    free-text suggestions for the voice-language field, so load ``lang_map.py``
    as a standalone module by file path instead, skipping the package init.
    Returns an empty list if OmniVoice isn't installed.
    """
    try:
        spec = importlib.util.find_spec("omnivoice")
        if spec is None or not spec.submodule_search_locations:
            return []
        lang_path = Path(spec.submodule_search_locations[0]) / "utils" / "lang_map.py"
        mod_spec = importlib.util.spec_from_file_location("_ov_lang_map", lang_path)
        if mod_spec is None or mod_spec.loader is None:
            return []
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return sorted(module.LANG_IDS)
    except Exception:  # noqa: BLE001 - suggestions are optional; never fail here
        return []


def _omnivoice_languages(default_language: str) -> List[str]:
    """Language directory suggestions for new OmniVoice voices.

    Uses OmniVoice's own language IDs when available; always includes the
    configured default so the UI still works even without OmniVoice installed.
    """
    langs = _omnivoice_lang_ids()
    if default_language and default_language not in langs:
        langs = [default_language] + langs
    return langs


# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------


def make_web_server(cli_args: argparse.Namespace) -> Flask:
    """Build the Flask app for managing voices."""
    flask_app = Flask(__name__)
    flask_app.config["MAX_CONTENT_LENGTH"] = _MAX_CONTENT_LENGTH

    # Home Assistant ingress: honor forwarded headers and the ingress path.
    flask_app.wsgi_app = ProxyFix(  # type: ignore[method-assign]
        flask_app.wsgi_app, x_proto=1, x_host=1
    )
    flask_app.wsgi_app = IngressPrefixMiddleware(  # type: ignore[method-assign]
        flask_app.wsgi_app
    )

    download_dir = Path(cli_args.download_dir)
    ref_dir = Path(cli_args.omnivoice_ref_dir) if cli_args.omnivoice_ref_dir else None
    backend = cli_args.backend

    def _builtin_voice_names() -> set:
        """Names of known built-in Piper voices (for labeling only)."""
        try:
            from .download import get_voices

            return set(get_voices(download_dir).keys())
        except Exception:  # noqa: BLE001 - listing must not fail on this
            return set()

    @flask_app.route("/", methods=["GET"])
    def index():  # type: ignore[no-untyped-def]
        return render_template_string(INDEX_HTML)

    @flask_app.route("/health", methods=["GET"])
    def health():  # type: ignore[no-untyped-def]
        return {"status": "ok"}, 200

    @flask_app.route("/api/status", methods=["GET"])
    def status():  # type: ignore[no-untyped-def]
        return jsonify(
            {
                "backend": backend,
                "piper_enabled": backend == "piper",
                "omnivoice_enabled": backend == "omnivoice",
                "omnivoice_ref_dir": str(ref_dir) if ref_dir else None,
                "download_dir": str(download_dir),
                "omnivoice_languages": _omnivoice_languages(
                    cli_args.omnivoice_language
                ),
            }
        )

    # --- Piper -------------------------------------------------------------

    @flask_app.route("/api/piper/voices", methods=["GET"])
    def piper_voices():  # type: ignore[no-untyped-def]
        return jsonify(
            {"voices": _list_piper_voices(download_dir, _builtin_voice_names())}
        )

    @flask_app.route("/api/piper/upload", methods=["POST"])
    def piper_upload():  # type: ignore[no-untyped-def]
        onnx_file = request.files.get("onnx")
        config_file = request.files.get("config")
        if onnx_file is None or not onnx_file.filename:
            return jsonify({"ok": False, "error": "Missing .onnx model file"}), 400
        if config_file is None or not config_file.filename:
            return (
                jsonify({"ok": False, "error": "Missing .onnx.json config file"}),
                400,
            )

        onnx_name = Path(onnx_file.filename).name
        if not onnx_name.endswith(".onnx"):
            return (
                jsonify({"ok": False, "error": "Model file must end with .onnx"}),
                400,
            )

        base = onnx_name[: -len(".onnx")]
        if not _valid_name(base):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            "Invalid voice name; use letters, digits, '.', '-', '_'"
                        ),
                    }
                ),
                400,
            )

        # Config must be valid JSON (and it becomes <base>.onnx.json regardless
        # of the uploaded filename).
        config_bytes = config_file.read()
        try:
            json.loads(config_bytes)
        except ValueError as err:
            return (
                jsonify({"ok": False, "error": f"Config is not valid JSON: {err}"}),
                400,
            )

        download_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = download_dir / f"{base}.onnx"
        config_path = download_dir / f"{base}.onnx.json"

        onnx_file.save(onnx_path)
        config_path.write_bytes(config_bytes)
        _LOGGER.info("Uploaded custom Piper voice: %s", base)

        return jsonify({"ok": True, "name": base, "message": RELOAD_MESSAGE})

    @flask_app.route("/api/piper/delete", methods=["POST"])
    def piper_delete():  # type: ignore[no-untyped-def]
        name = (request.form.get("name") or "").strip()
        if not _valid_name(name):
            return jsonify({"ok": False, "error": "Invalid voice name"}), 400

        onnx_path = download_dir / f"{name}.onnx"
        config_path = download_dir / f"{name}.onnx.json"
        if not onnx_path.is_file() and not config_path.is_file():
            return jsonify({"ok": False, "error": f"Voice not found: {name}"}), 404

        removed = []
        for path in (onnx_path, config_path):
            try:
                if path.is_file():
                    path.unlink()
                    removed.append(path.name)
            except OSError as err:
                return (
                    jsonify({"ok": False, "error": f"Could not delete {path}: {err}"}),
                    500,
                )

        _LOGGER.info("Deleted custom Piper voice: %s (%s)", name, ", ".join(removed))
        return jsonify({"ok": True, "message": RELOAD_MESSAGE})

    # --- OmniVoice ---------------------------------------------------------

    @flask_app.route("/api/omnivoice/voices", methods=["GET"])
    def omnivoice_voices():  # type: ignore[no-untyped-def]
        return jsonify({"voices": _list_omnivoice_voices(ref_dir)})

    @flask_app.route("/api/omnivoice/upload", methods=["POST"])
    def omnivoice_upload():  # type: ignore[no-untyped-def]
        if not ref_dir:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            "No reference directory configured " "(--omnivoice-ref-dir)"
                        ),
                    }
                ),
                400,
            )

        from .omnivoice import DEFAULT_VOICE_NAME

        name = (request.form.get("name") or "").strip()
        language = (request.form.get("language") or "").strip()
        transcript = (request.form.get("transcript") or "").strip()
        wav_file = request.files.get("wav")

        if not _valid_name(name):
            return jsonify({"ok": False, "error": "Invalid voice name"}), 400
        if name == DEFAULT_VOICE_NAME:
            return (
                jsonify({"ok": False, "error": f"'{DEFAULT_VOICE_NAME}' is reserved"}),
                400,
            )
        if not _valid_name(language):
            return jsonify({"ok": False, "error": "Invalid language"}), 400
        if not transcript:
            return jsonify({"ok": False, "error": "Transcript is required"}), 400
        if wav_file is None or not wav_file.filename:
            return jsonify({"ok": False, "error": "Missing reference WAV file"}), 400

        voice_dir = (ref_dir / language / name).resolve()
        # Guard against path traversal via crafted names.
        if ref_dir.resolve() not in voice_dir.parents:
            return jsonify({"ok": False, "error": "Invalid path"}), 400
        if voice_dir.exists():
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            f"Voice '{name}' already exists under {language}; "
                            "delete it first"
                        ),
                    }
                ),
                409,
            )

        wav_bytes = wav_file.read()
        # Validate it is a real WAV before writing anything.
        try:
            import io

            with wave.open(io.BytesIO(wav_bytes), "rb"):
                pass
        except (wave.Error, EOFError) as err:
            return (
                jsonify({"ok": False, "error": f"Not a valid WAV file: {err}"}),
                400,
            )

        voice_dir.mkdir(parents=True, exist_ok=False)
        (voice_dir / "ref.wav").write_bytes(wav_bytes)
        (voice_dir / "ref.txt").write_text(transcript, encoding="utf-8")
        _LOGGER.info("Created OmniVoice cloning voice: %s/%s", language, name)

        return jsonify({"ok": True, "name": name, "message": RELOAD_MESSAGE})

    @flask_app.route("/api/omnivoice/delete", methods=["POST"])
    def omnivoice_delete():  # type: ignore[no-untyped-def]
        if not ref_dir:
            return jsonify({"ok": False, "error": "No reference directory"}), 400

        name = (request.form.get("name") or "").strip()
        language = (request.form.get("language") or "").strip()
        if not _valid_name(name) or not _valid_name(language):
            return jsonify({"ok": False, "error": "Invalid voice or language"}), 400

        voice_dir = (ref_dir / language / name).resolve()
        if ref_dir.resolve() not in voice_dir.parents:
            return jsonify({"ok": False, "error": "Invalid path"}), 400
        if not voice_dir.is_dir():
            return jsonify({"ok": False, "error": f"Voice not found: {name}"}), 404

        try:
            shutil.rmtree(voice_dir)
        except OSError as err:
            return jsonify({"ok": False, "error": f"Could not delete: {err}"}), 500

        _LOGGER.info("Deleted OmniVoice voice: %s/%s", language, name)
        return jsonify({"ok": True, "message": RELOAD_MESSAGE})

    return flask_app


def run_web_server(flask_app: Flask, host: str, port: int) -> threading.Thread:
    """Run the Flask app in a daemon thread."""

    def _run() -> None:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        flask_app.run(host=host, port=port, use_reloader=False, threaded=True)

    thread = threading.Thread(target=_run, name="web-server", daemon=True)
    thread.start()
    _LOGGER.info("Web UI available on http://%s:%s", host, port)
    return thread


class IngressPrefixMiddleware:
    """Strip the Home Assistant ingress path prefix so routing works."""

    def __init__(self, app):  # type: ignore[no-untyped-def]
        self.app = app

    def __call__(self, environ, start_response):  # type: ignore[no-untyped-def]
        ingress_path = environ.get("HTTP_X_INGRESS_PATH", "")
        if ingress_path:
            environ["SCRIPT_NAME"] = ingress_path
            path_info = environ.get("PATH_INFO", "")
            if path_info.startswith(ingress_path):
                environ["PATH_INFO"] = path_info[len(ingress_path) :] or "/"
        return self.app(environ, start_response)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Piper &amp; OmniVoice voices</title>
<style>
  :root {
    --bg: #f5f6f8; --fg: #1c1c1c; --muted: #666; --card: #fff;
    --border: #d9dce1; --accent: #0b6bcb; --accent-fg: #fff;
    --warn-bg: #fff4e5; --warn-border: #ffb74d; --warn-fg: #7a4f01;
    --ok-bg: #e6f4ea; --ok-border: #66bb6a; --ok-fg: #1a5e2a;
    --err-bg: #fdecea; --err-border: #ef5350; --err-fg: #a32019;
    --danger: #cf222e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1b1e23; --fg: #e6e6e6; --muted: #9aa0a6; --card: #24272e;
      --border: #3a3f47; --accent: #4c9ffe; --accent-fg: #0b0f14;
      --warn-bg: #3a2f1a; --warn-border: #a9772a; --warn-fg: #ffcf8a;
      --ok-bg: #1e3524; --ok-border: #3f8a4d; --ok-fg: #9fe0ab;
      --err-bg: #3a1f1e; --err-border: #a3423f; --err-fg: #ffb3ae;
      --danger: #ff6b6b;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 0; background: var(--bg); color: var(--fg); line-height: 1.5;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
  h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
  h2 { font-size: 1.2rem; margin: 0; }
  .sub { color: var(--muted); margin: 0 0 1.5rem; }
  section {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem;
  }
  .sec-head { display: flex; align-items: center; gap: .5rem; margin-bottom: 1rem; }
  .badge {
    font-size: .72rem; text-transform: uppercase; letter-spacing: .03em;
    padding: .12rem .5rem; border-radius: 999px; border: 1px solid var(--border);
    color: var(--muted);
  }
  .badge.active { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
  .note { border-radius: 8px; padding: .6rem .8rem; margin: 0 0 1rem; font-size: .92rem; }
  .note.warn { background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn-fg); }
  .banner { position: sticky; top: 0; z-index: 5; margin: 0 0 1rem; }
  .banner .note { margin: 0; }
  .note.ok { background: var(--ok-bg); border: 1px solid var(--ok-border); color: var(--ok-fg); }
  .note.err { background: var(--err-bg); border: 1px solid var(--err-border); color: var(--err-fg); }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; }
  th, td { border-bottom: 1px solid var(--border); padding: .5rem .5rem; text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }
  td.mono, .mono { font-variant-numeric: tabular-nums; }
  .voice-name { font-weight: 600; }
  .tag { font-size: .72rem; padding: .05rem .4rem; border-radius: 4px; border: 1px solid var(--border); color: var(--muted); margin-left: .4rem; }
  .quote { color: var(--muted); font-style: italic; max-width: 32ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .empty { color: var(--muted); padding: .75rem 0; }
  button {
    font: inherit; padding: .45rem .9rem; border-radius: 7px; cursor: pointer;
    border: 1px solid var(--accent); background: var(--accent); color: var(--accent-fg);
  }
  button.secondary { background: transparent; color: var(--accent); }
  button.link-danger {
    background: transparent; border: none; color: var(--danger);
    padding: .2rem .3rem; cursor: pointer; text-decoration: underline;
  }
  button[disabled] { opacity: .5; cursor: default; }
  details { margin-top: 1rem; border-top: 1px dashed var(--border); padding-top: 1rem; }
  summary { cursor: pointer; font-weight: 600; }
  form.add { margin-top: 1rem; display: grid; gap: .75rem; }
  label { display: block; font-size: .9rem; }
  label > span { display: block; color: var(--muted); margin-bottom: .2rem; }
  input[type=text], textarea, select, input[type=file] {
    width: 100%; padding: .45rem .5rem; border: 1px solid var(--border);
    border-radius: 7px; background: var(--bg); color: var(--fg); font: inherit;
  }
  textarea { min-height: 4rem; resize: vertical; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
  @media (max-width: 560px) { .row { grid-template-columns: 1fr; } }
  .req { color: var(--danger); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Piper &amp; OmniVoice voices</h1>
  <p class="sub">Manage custom voices for this Wyoming TTS server.</p>

  <div class="banner"><div id="global-msg"></div></div>

  <!-- Piper ------------------------------------------------------------- -->
  <section id="piper">
    <div class="sec-head">
      <h2>Piper</h2>
      <span class="badge" id="piper-badge">&nbsp;</span>
    </div>
    <div id="piper-warn"></div>
    <p class="sub">Custom voices in the download directory. Each voice is a
      <code>&lt;name&gt;.onnx</code> model and its <code>&lt;name&gt;.onnx.json</code> config.</p>
    <div id="piper-list"><p class="empty">Loading…</p></div>

    <details>
      <summary>Add a custom voice</summary>
      <form class="add" id="piper-form">
        <div class="row">
          <label><span>Model file (<code>.onnx</code>) <span class="req">*</span></span>
            <input type="file" name="onnx" accept=".onnx" required></label>
          <label><span>Config file (<code>.onnx.json</code>) <span class="req">*</span></span>
            <input type="file" name="config" accept=".json" required></label>
        </div>
        <div><button type="submit">Upload voice</button></div>
      </form>
    </details>
  </section>

  <!-- OmniVoice --------------------------------------------------------- -->
  <section id="omnivoice">
    <div class="sec-head">
      <h2>OmniVoice</h2>
      <span class="badge" id="omni-badge">&nbsp;</span>
    </div>
    <div id="omni-warn"></div>
    <p class="sub">Cloned voices, each a reference recording plus its transcript.</p>
    <div id="omni-list"><p class="empty">Loading…</p></div>

    <details id="omni-add-wrap">
      <summary>Add a cloned voice</summary>
      <form class="add" id="omni-form">
        <div class="row">
          <label><span>Voice name <span class="req">*</span></span>
            <input type="text" name="name" placeholder="e.g. my_voice" required></label>
          <label><span>Language <span class="req">*</span></span>
            <input type="text" name="language" list="omni-langs" placeholder="e.g. en_US" required>
            <datalist id="omni-langs"></datalist></label>
        </div>
        <label><span>Reference WAV <span class="req">*</span></span>
          <input type="file" name="wav" accept=".wav,audio/wav" required></label>
        <label><span>Transcript of the recording <span class="req">*</span></span>
          <textarea name="transcript" placeholder="Exactly what is spoken in the WAV" required></textarea></label>
        <div><button type="submit">Upload voice</button></div>
      </form>
    </details>
  </section>
</div>

<script>
const BASE = window.location.pathname.replace(/\/+$/, "");
const api = (p) => BASE + p;
const el = (id) => document.getElementById(id);

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
}
function fmtSize(n) {
  if (n == null) return "";
  const mb = n / (1024 * 1024);
  return mb >= 1 ? mb.toFixed(1) + " MB" : (n / 1024).toFixed(0) + " KB";
}
function showGlobal(kind, msg) {
  el("global-msg").innerHTML =
    '<div class="note ' + kind + '">' + esc(msg) + '</div>';
  if (kind === "ok") window.scrollTo({ top: 0, behavior: "smooth" });
}
function sectionMsg(id, kind, msg) {
  el(id).innerHTML = msg ? '<div class="note ' + kind + '">' + esc(msg) + '</div>' : '';
}

let STATE = {};

async function loadStatus() {
  const r = await fetch(api("/api/status"));
  STATE = await r.json();

  // Piper badge + warning
  el("piper-badge").textContent = STATE.piper_enabled ? "active backend" : "not active";
  el("piper-badge").className = "badge" + (STATE.piper_enabled ? " active" : "");
  if (!STATE.piper_enabled) {
    sectionMsg("piper-warn", "warn",
      "The Piper backend is not enabled (server backend is '" + STATE.backend +
      "'). You can still manage voice files here, but this server won't use them.");
  }

  // OmniVoice badge + warning
  el("omni-badge").textContent = STATE.omnivoice_enabled ? "active backend" : "not active";
  el("omni-badge").className = "badge" + (STATE.omnivoice_enabled ? " active" : "");
  let warn = "";
  if (!STATE.omnivoice_enabled) {
    warn = "The OmniVoice backend is not enabled (server backend is '" +
      STATE.backend + "').";
  }
  if (!STATE.omnivoice_ref_dir) {
    warn += (warn ? " " : "") +
      "No reference directory is configured (--omnivoice-ref-dir), so cloned " +
      "voices cannot be managed.";
    el("omni-form").querySelectorAll("input,textarea,button")
      .forEach(e => e.disabled = true);
    el("omni-add-wrap").open = false;
  }
  if (warn) sectionMsg("omni-warn", "warn", warn);

  const dl = el("omni-langs");
  dl.innerHTML = (STATE.omnivoice_languages || [])
    .map(l => '<option value="' + esc(l) + '">').join("");
}

async function loadPiper() {
  const r = await fetch(api("/api/piper/voices"));
  const { voices } = await r.json();
  if (!voices.length) {
    el("piper-list").innerHTML = '<p class="empty">No custom voices found.</p>';
    return;
  }
  let rows = voices.map(v =>
    "<tr><td><span class='voice-name'>" + esc(v.name) + "</span>" +
      (v.builtin ? "<span class='tag'>built-in</span>" : "") + "</td>" +
    "<td>" + esc(v.dataset || "") + "</td>" +
    "<td>" + esc(v.language || "") + "</td>" +
    "<td>" + esc(v.quality || "") + "</td>" +
    "<td class='mono'>" + esc(v.sample_rate || "") + "</td>" +
    "<td class='mono'>" + esc(fmtSize(v.size_bytes)) + "</td>" +
    "<td><button class='link-danger' data-name='" + esc(v.name) +
      "'>Delete</button></td></tr>"
  ).join("");
  el("piper-list").innerHTML =
    "<table><thead><tr><th>Voice</th><th>Dataset</th><th>Language</th>" +
    "<th>Quality</th><th>Rate</th><th>Size</th><th></th></tr></thead><tbody>" +
    rows + "</tbody></table>";
  el("piper-list").querySelectorAll("button[data-name]").forEach(b =>
    b.addEventListener("click", () => deletePiper(b.dataset.name)));
}

async function loadOmni() {
  if (!STATE.omnivoice_ref_dir) {
    el("omni-list").innerHTML =
      '<p class="empty">Reference directory not configured.</p>';
    return;
  }
  const r = await fetch(api("/api/omnivoice/voices"));
  const { voices } = await r.json();
  if (!voices.length) {
    el("omni-list").innerHTML = '<p class="empty">No cloned voices found.</p>';
    return;
  }
  let rows = voices.map(v =>
    "<tr><td class='voice-name'>" + esc(v.name) + "</td>" +
    "<td>" + esc(v.language) + "</td>" +
    "<td>" + esc(v.kind) + "</td>" +
    "<td class='quote' title='" + esc(v.ref_text || v.instruct || "") + "'>" +
      esc(v.ref_text || v.instruct || "") + "</td>" +
    "<td><button class='link-danger' data-name='" + esc(v.name) +
      "' data-lang='" + esc(v.language) + "'>Delete</button></td></tr>"
  ).join("");
  el("omni-list").innerHTML =
    "<table><thead><tr><th>Voice</th><th>Language</th><th>Kind</th>" +
    "<th>Transcript / style</th><th></th></tr></thead><tbody>" +
    rows + "</tbody></table>";
  el("omni-list").querySelectorAll("button[data-name]").forEach(b =>
    b.addEventListener("click", () => deleteOmni(b.dataset.name, b.dataset.lang)));
}

async function postForm(url, formData) {
  const r = await fetch(url, { method: "POST", body: formData });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || !data.ok) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}

async function deletePiper(name) {
  if (!confirm("Delete custom voice '" + name + "'? This removes its files.")) return;
  const fd = new FormData(); fd.append("name", name);
  try {
    const d = await postForm(api("/api/piper/delete"), fd);
    showGlobal("ok", d.message);
    loadPiper();
  } catch (e) { showGlobal("err", e.message); }
}

async function deleteOmni(name, lang) {
  if (!confirm("Delete cloned voice '" + name + "'? This removes its directory.")) return;
  const fd = new FormData(); fd.append("name", name); fd.append("language", lang);
  try {
    const d = await postForm(api("/api/omnivoice/delete"), fd);
    showGlobal("ok", d.message);
    loadOmni();
  } catch (e) { showGlobal("err", e.message); }
}

el("piper-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = ev.target.querySelector("button");
  btn.disabled = true; showGlobal("", "");
  try {
    const d = await postForm(api("/api/piper/upload"), new FormData(ev.target));
    showGlobal("ok", d.message);
    ev.target.reset();
    loadPiper();
  } catch (e) { showGlobal("err", e.message); }
  finally { btn.disabled = false; }
});

el("omni-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = ev.target.querySelector("button");
  btn.disabled = true; showGlobal("", "");
  try {
    const d = await postForm(api("/api/omnivoice/upload"), new FormData(ev.target));
    showGlobal("ok", d.message);
    ev.target.reset();
    loadOmni();
  } catch (e) { showGlobal("err", e.message); }
  finally { btn.disabled = false; }
});

(async function init() {
  try {
    await loadStatus();
    await Promise.all([loadPiper(), loadOmni()]);
  } catch (e) {
    showGlobal("err", "Failed to load: " + e.message);
  }
})();
</script>
</body>
</html>
"""
