# syntax=docker/dockerfile:1
FROM python:3.11-slim

# espeak-ng: phonemization Kokoro/misaki uses. libsndfile1: runtime lib for soundfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first so Kokoro doesn't pull the much larger CUDA build.
# Keeps the image small enough to build and introspect quickly.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "kokoro>=0.8.4" \
    "mcp[cli]>=1.3.0" \
    "soundfile>=0.13.1" \
    "httpx>=0.27.0"

COPY tts-mcp.py ./

# MCP clients launch the server over stdio.
ENV MCP_TRANSPORT=stdio

CMD ["python", "tts-mcp.py"]
