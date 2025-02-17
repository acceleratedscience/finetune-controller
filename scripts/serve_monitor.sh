#!/bin/bash

echo ">> installing dependencies.."
uv sync
echo ">> initializing environment.."
source .venv/bin/activate
echo ">> starting debug server.."
python -m app.monitor_main
