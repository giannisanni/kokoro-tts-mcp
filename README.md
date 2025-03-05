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

## Dependencies

- kokoro >= 0.8.4
- mcp[cli] >= 1.3.0
- soundfile >= 0.13.1

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

[Add your license information here]
