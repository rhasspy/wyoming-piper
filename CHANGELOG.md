# Changelog

## 2.0.0

- Use [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl/) library instead of piper binary
- Use [sentence-stream](https://github.com/OHF-Voice/sentence-stream) library instead of internal code
- Add `--use-cuda` to enable GPU acceleration (requires `onnxruntime-gpu`)
- Ignore file sizes and hashes when downloading voices
- Add Docker build here
- Add publish workflow
- Remove `--piper` and `--max-piper-procs` (no longer needed)
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
