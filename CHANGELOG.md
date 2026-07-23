# Changelog

## 2.4.0

- Add experimental `--backend omnivoice` for [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS via onnxruntime
    - `--omnivoice-steps` to configure MaskGIT decode steps (default: 32)
    - `--omnivoice-ref-dir` for custom voices: a directory of voices organized
      as `<language>/<voice_name>/`, loaded at startup and advertised as voices.
      Each voice directory is either a cloning voice (`ref.{wav,txt}`) or a
      voice-design voice (`instruct.txt`, whose text describes the desired voice
      style). A `default` voice is also advertised for every supported language;
      it (or an empty/unknown voice name) uses the built-in speaker.
    - `--omnivoice-language` to set the default synthesis language (default:
      English); per-request language codes (`en_US`, `en-US`) are also honored
    - `--omnivoice-onnx-repo` to override the HuggingFace repo for the ONNX graph
    - block-wise int4 quantization is hardcoded for now (clean audio at low step
      counts, e.g. `--omnivoice-steps 10`); reproduce with
      `script/quantize_omnivoice.py`
    - the ONNX model is used from a `--data-dir` if present there, otherwise
      downloaded into `--download-dir` (used as the HuggingFace cache)
    - install with the `omnivoice` optional dependencies
- Add `--local-files-only` to run the HuggingFace loader in offline mode
- Add `--web-server` for a web UI (runs alongside the Wyoming server) to manage custom and cloned voices

## 2.3.1

- Fix publishing

## 2.3.0

- Add `--sentence-silence` (seconds of silence after each sentence)

## 2.2.2

- Bump `piper-tts` to 1.4.1 (wheel fix)

## 2.2.1

- Fix `zeroconf` dependency
- Don't download `g2pW` model into Docker container

## 2.2.0

- Bump `piper-tts` to 1.4.0
- Add `zh` optional dependencies for new Chinese voices
- Add `zeroconf` optional dependencies

## 2.1.2

- Add `--data-dir /data` to Docker run script

## 2.1.1

- Fix issue with streaming

## 2.1.0

- Add `--zeroconf` option for discovery

## 2.0.0

- Use [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl/) library instead of piper binary
- Use [sentence-stream](https://github.com/OHF-Voice/sentence-stream) library instead of internal code
- Add `--use-cuda` to enable GPU acceleration (requires `onnxruntime-gpu`)
- Ignore file sizes and hashes when downloading voices
- Default streaming to be on (remove `--streaming`) and add `--no-streaming` to disable
- Remove `--piper` and `--max-piper-procs` (no longer needed)
- Add Docker build here
- Add publish workflow
- Add alias `--noise-w-scale` for `--noise-w` to align with piper1-gpl

## 1.6.3

- Bump wyoming to 1.7.2 to fix error with event data

## 1.6.2

- Remove asterisks at the start of a line (markdown list)

## 1.6.1

- Split sentences on numbered lists and remove asterisks surrounding words

## 1.6.0

- Add support for streaming audio on sentence boundaries (`--streaming`)

## 1.5.4

- Merge downloaded voices.json on top of embedded
- Use "main" instead of "v1.0.0" in HF URL

## 1.5.3

- Migrate to pyproject.toml
- Update voices.json

## 1.5.0

- Send speakers in `info` message
- Update voices.json with new voices
- Add tests to CI

## 1.4.0

- Fix use of UTF-8 characters in URLs
- Try harder to avoid re-downloading files
- Update voices.json

## 1.3.3

- Initial release
