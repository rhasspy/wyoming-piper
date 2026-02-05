#!/usr/bin/env bash
cd /usr/src
exec .venv/bin/python3 -m wyoming_piper \
    --uri 'tcp://0.0.0.0:10200' \
    --data-dir /data "$@"
