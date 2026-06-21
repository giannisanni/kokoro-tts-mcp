# Kokoro TTS MCP Server

A Model Context Protocol (MCP) server that provides text-to-speech capabilities using the Kokoro TTS engine. This server exposes TTS functionality through MCP tools, making it easy to integrate speech synthesis into your applications.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager

## Installation

1. First, install the `uv` package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone this repository and install dependencies:

```bash
uv venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
uv pip install .
```

## Features

- Text-to-speech synthesis with customizable voices
- Adjustable speech speed
- Support for saving audio to files or direct playback
- Cross-platform audio playback support (Windows, macOS, Linux)
- Optional OpenAI-compatible remote backend (e.g. [kokoro-fastapi](https://github.com/remsky/Kokoro-FastAPI)) to offload synthesis to a GPU box

## Usage

The server provides a single MCP tool `generate_speech` with the following parameters:

- `text` (required): The text to convert to speech
- `voice` (optional): Voice to use for synthesis (default: "af_heart")
- `speed` (optional): Speech speed multiplier (default: 1.0)
- `save_path` (optional): Directory to save audio files
- `play_audio` (optional): Whether to play the audio immediately (default: False)

### Example Usage

```python
from mcp.client import Client

async with Client() as client:
    await client.connect("kokoro-tts")
    
    # Generate and play speech
    result = await client.call_tool(
        "generate_speech",
        {
            "text": "Hello, world!",
            "voice": "af_heart",
            "speed": 1.0,
            "play_audio": True
        }
    )
```

## Remote backend (OpenAI-compatible)

By default the server runs Kokoro locally. If you already run an OpenAI-compatible
TTS endpoint such as [kokoro-fastapi](https://github.com/remsky/Kokoro-FastAPI)
(handy for running on a GPU), point the server at it with environment variables —
no local `torch`/`kokoro` needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `KOKORO_BASE_URL` | _(unset)_ | OpenAI-compatible base URL, e.g. `http://localhost:8880/v1`. When set, synthesis is sent here instead of running locally. |
| `KOKORO_API_KEY` | `not-needed` | Bearer token, if your endpoint requires one. |
| `KOKORO_MODEL` | `kokoro` | Model name passed to the endpoint. |

Under the hood this calls `POST {KOKORO_BASE_URL}/audio/speech` with the standard
OpenAI payload (`model`, `input`, `voice`, `speed`, `response_format: wav`).

## Docker

```bash
docker build -t kokoro-tts-mcp .
docker run --rm -i kokoro-tts-mcp
```

To use a remote backend instead of bundling Kokoro:

```bash
docker run --rm -i -e KOKORO_BASE_URL=http://host.docker.internal:8880/v1 kokoro-tts-mcp
```

## Dependencies

- kokoro >= 0.8.4
- mcp[cli] >= 1.3.0
- soundfile >= 0.13.1
- httpx >= 0.27.0

## Platform Support

Audio playback is supported on:
- Windows (using `start`)
- macOS (using `afplay`)
- Linux (using `aplay`)

## MCP Configuration

Add the following configuration to your MCP settings file:

```json
{
  "mcpServers": {
    "kokoro-tts": {
      "command": "/Users/giannisan/pinokio/bin/miniconda/bin/uv",
      "args": [
        "--directory",
        "/Users/giannisan/Documents/Cline/MCP/kokoro-tts-mcp",
        "run",
        "tts-mcp.py"
      ]
    }
  }
}
```

## License

[MIT](LICENSE) © Gianni Sanrochman
