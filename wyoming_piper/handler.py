"""Event handler for clients of the server."""

import argparse
import asyncio
import logging
import math
import tempfile
import wave
from typing import Any, Dict, Optional

from piper import PiperVoice, SynthesisConfig
from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .download import ensure_voice_exists, find_voice

_LOGGER = logging.getLogger(__name__)

# Keep the most recently used voice loaded
_VOICE: Optional[PiperVoice] = None
_VOICE_NAME: Optional[str] = None
_VOICE_LOCK = asyncio.Lock()


class PiperEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        voices_info: Dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.voices_info = voices_info
        self.is_streaming: Optional[bool] = None
        self.sbd = SentenceBoundaryDetector()
        self._synthesize: Optional[Synthesize] = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    # Ignore since this is only sent for compatibility reasons.
                    # For streaming, we expect:
                    # [synthesize-start] -> [synthesize-chunk]+ -> [synthesize]? -> [synthesize-stop]
                    return True

                # Sent outside a stream, so we must process it
                synthesize = Synthesize.from_event(event)
                self._synthesize = Synthesize(text="", voice=synthesize.voice)
                self.sbd = SentenceBoundaryDetector()
                start_sent = False
                for i, sentence in enumerate(self.sbd.add_chunk(synthesize.text)):
                    self._synthesize.text = sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(i == 0), send_stop=False
                    )
                    start_sent = True

                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Last sentence
                    await self._handle_synthesize(
                        self._synthesize, send_start=(not start_sent), send_stop=True
                    )
                else:
                    # No final sentence
                    await self.write_event(AudioStop().event())

                return True

            if self.cli_args.no_streaming:
                # Streaming is not enabled
                return True

            if SynthesizeStart.is_type(event.type):
                # Start of a stream
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                for sentence in self.sbd.add_chunk(stream_chunk.text):
                    _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                    self._synthesize.text = sentence
                    await self._handle_synthesize(self._synthesize)

                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    # Final audio chunk(s)
                    await self._handle_synthesize(self._synthesize)

                # End of audio
                await self.write_event(SynthesizeStopped().event())

                _LOGGER.debug("Text stream stopped")
                return True

            if not Synthesize.is_type(event.type):
                return True

            synthesize = Synthesize.from_event(event)
            return await self._handle_synthesize(synthesize)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(
        self, synthesize: Synthesize, send_start: bool = True, send_stop: bool = True
    ) -> bool:
        global _VOICE, _VOICE_NAME

        _LOGGER.debug(synthesize)

        raw_text = synthesize.text

        # Join multiple lines
        text = " ".join(raw_text.strip().splitlines())

        if self.cli_args.auto_punctuation and text:
            # Add automatic punctuation (important for some voices)
            has_punctuation = False
            for punc_char in self.cli_args.auto_punctuation:
                if text[-1] == punc_char:
                    has_punctuation = True
                    break

            if not has_punctuation:
                text = text + self.cli_args.auto_punctuation[0]

        # Resolve voice
        _LOGGER.debug("synthesize: raw_text=%s, text='%s'", raw_text, text)
        voice_name: Optional[str] = None
        voice_speaker: Optional[str] = None
        if synthesize.voice is not None:
            voice_name = synthesize.voice.name
            voice_speaker = synthesize.voice.speaker

        if voice_name is None:
            # Default voice
            voice_name = self.cli_args.voice

        if voice_name == self.cli_args.voice:
            # Default speaker
            voice_speaker = voice_speaker or self.cli_args.speaker

        assert voice_name is not None

        # Resolve alias
        voice_info = self.voices_info.get(voice_name, {})
        voice_name = voice_info.get("key", voice_name)
        assert voice_name is not None

        with tempfile.NamedTemporaryFile(mode="wb+", suffix=".wav") as output_file:
            async with _VOICE_LOCK:
                if voice_name != _VOICE_NAME:
                    # Load new voice
                    _LOGGER.debug("Loading voice: %s", _VOICE_NAME)
                    ensure_voice_exists(
                        voice_name,
                        self.cli_args.data_dir,
                        self.cli_args.download_dir,
                        self.voices_info,
                    )
                    model_path, config_path = find_voice(
                        voice_name, self.cli_args.data_dir
                    )
                    _VOICE = PiperVoice.load(
                        model_path, config_path, use_cuda=self.cli_args.use_cuda
                    )
                    _VOICE_NAME = voice_name

                assert _VOICE is not None

                syn_config = SynthesisConfig()
                if voice_speaker is not None:
                    syn_config.speaker_id = _VOICE.config.speaker_id_map.get(
                        voice_speaker
                    )
                    if syn_config.speaker_id is None:
                        try:
                            # Try to interpret as an id
                            syn_config.speaker_id = int(voice_speaker)
                        except ValueError:
                            pass

                    if syn_config.speaker_id is None:
                        _LOGGER.warning(
                            "No speaker '%s' for voice '%s'", voice_speaker, voice_name
                        )

                if self.cli_args.length_scale is not None:
                    syn_config.length_scale = self.cli_args.length_scale

                if self.cli_args.noise_scale is not None:
                    syn_config.noise_scale = self.cli_args.noise_scale

                if self.cli_args.noise_w_scale is not None:
                    syn_config.noise_w_scale = self.cli_args.noise_w_scale

                wav_writer: wave.Wave_write = wave.open(output_file, "wb")
                with wav_writer:
                    _VOICE.synthesize_wav(text, wav_writer, syn_config)

            output_file.seek(0)

            wav_file: wave.Wave_read = wave.open(output_file, "rb")
            with wav_file:
                rate = wav_file.getframerate()
                width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()

                if send_start:
                    await self.write_event(
                        AudioStart(
                            rate=rate,
                            width=width,
                            channels=channels,
                        ).event(),
                    )

                # Audio
                audio_bytes = wav_file.readframes(wav_file.getnframes())
                bytes_per_sample = width * channels
                bytes_per_chunk = bytes_per_sample * self.cli_args.samples_per_chunk
                num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))

                # Split into chunks
                for i in range(num_chunks):
                    offset = i * bytes_per_chunk
                    chunk = audio_bytes[offset : offset + bytes_per_chunk]

                    await self.write_event(
                        AudioChunk(
                            audio=chunk,
                            rate=rate,
                            width=width,
                            channels=channels,
                        ).event(),
                    )

            if send_stop:
                await self.write_event(AudioStop().event())

        return True
