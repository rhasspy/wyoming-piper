#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import signal
from functools import partial
from pathlib import Path
from typing import Any, Dict, Set

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .download import ensure_voice_exists, find_voice, get_voices
from .handler import PiperEventHandler, get_omnivoice_voices, load_omnivoice

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="piper",
        choices=("piper", "omnivoice"),
        help="TTS backend to use (default: piper)",
    )
    parser.add_argument(
        "--voice",
        help="Default Piper voice to use (e.g., en_US-lessac-medium). "
        "Required for the piper backend.",
    )
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    #
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="piper",
        help="Enable discovery over zeroconf with optional name (default: piper)",
    )
    #
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download voices into (default: first data dir)",
    )
    #
    parser.add_argument(
        "--speaker", type=str, help="Name or id of speaker for default voice"
    )
    parser.add_argument("--noise-scale", type=float, help="Generator noise")
    parser.add_argument("--length-scale", type=float, help="Phoneme length")
    parser.add_argument(
        "--noise-w-scale", "--noise-w", type=float, help="Phoneme width noise"
    )
    parser.add_argument(
        "--sentence-silence",
        type=float,
        help="Seconds of silence to add between sentences (default: no silence)",
    )
    #
    parser.add_argument(
        "--auto-punctuation",
        default=".?!。？！．؟",
        help="Automatically add punctuation",
    )
    parser.add_argument("--samples-per-chunk", type=int, default=1024)
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable audio streaming on sentence boundaries",
    )
    #
    parser.add_argument(
        "--update-voices",
        action="store_true",
        help="Download latest voices.json during startup",
    )
    #
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CUDA if available (requires onnxruntime-gpu)",
    )
    #
    # Web UI for managing custom voices (runs alongside the Wyoming server)
    parser.add_argument(
        "--web-server",
        action="store_true",
        help="Run a web UI for managing custom Piper/OmniVoice voices "
        "(requires the 'web' optional dependencies)",
    )
    parser.add_argument(
        "--web-server-host",
        default="127.0.0.1",
        help="Host to bind the web UI to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--web-server-port",
        type=int,
        default=5000,
        help="Port for the web UI (default: 5000)",
    )
    #
    # OmniVoice backend options
    parser.add_argument(
        "--omnivoice-steps",
        type=int,
        default=32,
        help="Number of MaskGIT decode steps for the omnivoice backend "
        "(default: 32, fewer is faster)",
    )
    parser.add_argument(
        "--omnivoice-ref-dir",
        help="Directory of reference voices for cloning (omnivoice backend), "
        "organized as <language>/<voice_name>/ref.{wav,txt}. Each is advertised "
        "as a voice; requests without a voice use the built-in speaker.",
    )
    parser.add_argument(
        "--omnivoice-language",
        default="English",
        help="Language for the omnivoice backend (default: English)",
    )
    parser.add_argument(
        "--omnivoice-onnx-repo",
        help="HuggingFace repo id for the int4 ONNX graph (omnivoice backend). "
        "Overrides the built-in default.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached model files; never download "
        "(sets HuggingFace hub to offline mode)",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        # Default to first data directory
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    if args.backend == "omnivoice":
        wyoming_info, voices_info = _setup_omnivoice(args)
    else:
        if not args.voice:
            parser.error("--voice is required for the piper backend")

        wyoming_info, voices_info = _setup_piper(args)

    # Start server
    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    # Optional web UI for managing custom voices, in a background thread.
    if args.web_server:
        try:
            from .web_server import make_web_server, run_web_server
        except ImportError as err:
            parser.error(
                f"--web-server requires the 'web' optional dependencies ({err})"
            )

        run_web_server(
            make_web_server(args),
            host=args.web_server_host,
            port=args.web_server_port,
        )

    _LOGGER.info("Ready")
    server_task = asyncio.create_task(
        server.run(
            partial(
                PiperEventHandler,
                wyoming_info,
                args,
                voices_info,
            )
        )
    )
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, server_task.cancel)
    loop.add_signal_handler(signal.SIGTERM, server_task.cancel)

    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("Server stopped")


# -----------------------------------------------------------------------------


def _setup_piper(args: argparse.Namespace) -> "tuple[Info, Dict[str, Any]]":
    """Build Wyoming info and voice table for the piper backend."""
    # Load voice info
    voices_info = get_voices(args.download_dir, update_voices=args.update_voices)

    # Resolve aliases for backwards compatibility with old voice names
    aliases_info: Dict[str, Any] = {}
    for voice_info in voices_info.values():
        for voice_alias in voice_info.get("aliases", []):
            aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

    voices_info.update(aliases_info)
    voices = [
        TtsVoice(
            name=voice_name,
            description=get_description(voice_info),
            attribution=Attribution(
                name="rhasspy", url="https://github.com/rhasspy/piper"
            ),
            installed=True,
            version=None,
            languages=[
                voice_info.get("language", {}).get(
                    "code",
                    voice_info.get("espeak", {}).get("voice", voice_name.split("_")[0]),
                )
            ],
            speakers=(
                [
                    TtsVoiceSpeaker(name=speaker_name)
                    for speaker_name in voice_info["speaker_id_map"]
                ]
                if voice_info.get("speaker_id_map")
                else None
            ),
        )
        for voice_name, voice_info in voices_info.items()
        if not voice_info.get("_is_alias", False)
    ]

    custom_voice_names: Set[str] = set()
    if args.voice not in voices_info:
        custom_voice_names.add(args.voice)

    for data_dir in args.data_dir:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            continue

        for onnx_path in data_dir.glob("*.onnx"):
            custom_voice_name = onnx_path.stem
            if custom_voice_name not in voices_info:
                custom_voice_names.add(custom_voice_name)

    for custom_voice_name in custom_voice_names:
        # Add custom voice info
        custom_voice_path, custom_config_path = find_voice(
            custom_voice_name, args.data_dir
        )
        with open(custom_config_path, "r", encoding="utf-8") as custom_config_file:
            custom_config = json.load(custom_config_file)
            custom_name = custom_config.get("dataset", custom_voice_path.stem)
            custom_quality = custom_config.get("audio", {}).get("quality")
            if custom_quality:
                description = f"{custom_name} ({custom_quality})"
            else:
                description = custom_name

            lang_code = custom_config.get("language", {}).get("code")
            if not lang_code:
                lang_code = custom_config.get("espeak", {}).get("voice")
                if not lang_code:
                    lang_code = custom_voice_path.stem.split("_")[0]

            voices.append(
                TtsVoice(
                    name=custom_name,
                    description=description,
                    version=None,
                    attribution=Attribution(name="", url=""),
                    installed=True,
                    languages=[lang_code],
                )
            )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="piper",
                description="A fast, local, neural text to speech engine",
                attribution=Attribution(
                    name="rhasspy", url="https://github.com/rhasspy/piper"
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version=__version__,
                supports_synthesize_streaming=(not args.no_streaming),
            )
        ],
    )

    # Ensure default voice is downloaded
    voice_info = voices_info.get(args.voice, {})
    voice_name = voice_info.get("key", args.voice)
    assert voice_name is not None

    ensure_voice_exists(voice_name, args.data_dir, args.download_dir, voices_info)

    return wyoming_info, voices_info


# -----------------------------------------------------------------------------


def _setup_omnivoice(args: argparse.Namespace) -> "tuple[Info, Dict[str, Any]]":
    """Build Wyoming info and ensure models for the omnivoice backend.

    The HuggingFace cache is pointed at ``--download-dir`` and the model is
    downloaded there (unless ``--local-files-only`` is set).
    """
    import os

    # Point the HuggingFace cache at the download dir before any hub import.
    os.environ["HF_HOME"] = str(Path(args.download_dir).resolve())
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Download (if needed) and load the shared model + reference voices now,
    # before serving.
    load_omnivoice(args)

    from .omnivoice import (
        DEFAULT_VOICE_NAME,
        advertise_language,
        get_supported_languages,
    )

    attribution = Attribution(name="k2-fsa", url="https://github.com/k2-fsa/OmniVoice")

    # Built-in (no-reference) speaker: advertised for every supported language.
    # Used for this voice, an empty voice name, or an unknown one.
    voices = [
        TtsVoice(
            name=DEFAULT_VOICE_NAME,
            description="OmniVoice",
            version=None,
            attribution=attribution,
            installed=True,
            languages=get_supported_languages(),
        )
    ]
    # Cloning and voice-design (instruct) voices discovered under --omnivoice-ref-dir.
    for ref in get_omnivoice_voices().values():
        lang = advertise_language(ref.language)
        voices.append(
            TtsVoice(
                name=ref.name,
                description=ref.name,
                version=None,
                attribution=attribution,
                installed=True,
                languages=[lang],
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="omnivoice",
                description="High-quality multilingual voice-cloning TTS",
                attribution=attribution,
                installed=True,
                voices=voices,
                version=__version__,
                supports_synthesize_streaming=(not args.no_streaming),
            )
        ],
    )

    # voices_info is unused by the omnivoice backend.
    return wyoming_info, {}


# -----------------------------------------------------------------------------


def get_description(voice_info: Dict[str, Any]):
    """Get a human readable description for a voice."""
    name = voice_info["name"]
    name = " ".join(name.split("_"))
    quality = voice_info["quality"]

    return f"{name} ({quality})"


# -----------------------------------------------------------------------------


def run():
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
