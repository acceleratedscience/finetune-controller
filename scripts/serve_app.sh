#!/bin/bash

echo ">> installing dependencies.."
uv sync
echo ">> initializing environment.."
source .venv/bin/activate
echo ">> starting debug server.."
uvicorn app.main:app --reload
