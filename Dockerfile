# syntax=docker/dockerfile:1
FROM python:3.11-slim

# espeak-ng provides the phonemization Kokoro/misaki uses for several languages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
RUN pip install --no-cache-dir \
    "kokoro>=0.8.4" \
    "mcp[cli]>=1.3.0" \
    "soundfile>=0.13.1" \
    "httpx>=0.27.0"

COPY tts-mcp.py ./

# MCP clients launch the server over stdio.
ENV MCP_TRANSPORT=stdio

CMD ["python", "tts-mcp.py"]
