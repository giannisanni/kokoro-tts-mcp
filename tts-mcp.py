import sys
import os
import logging
import subprocess
import tempfile
from typing import List, Optional

import httpx
import soundfile as sf
from mcp.server.fastmcp import FastMCP
from pathlib import Path

# Disable ALL logging
logging.disable(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.captureWarnings(True)

# Optional remote backend. When KOKORO_BASE_URL is set, requests are sent to an
# OpenAI-compatible TTS endpoint (e.g. kokoro-fastapi: https://github.com/remsky/Kokoro-FastAPI)
# instead of running Kokoro locally. This lets you offload synthesis to a GPU box
# and run this MCP server without torch/kokoro installed.
REMOTE_BASE_URL: Optional[str] = os.getenv("KOKORO_BASE_URL")  # e.g. http://localhost:8880/v1
REMOTE_API_KEY: str = os.getenv("KOKORO_API_KEY", "not-needed")
REMOTE_MODEL: str = os.getenv("KOKORO_MODEL", "kokoro")

mcp = FastMCP("kokoro-tts")

# The local pipeline (and its heavy torch/kokoro imports) is loaded lazily so the
# server can run in remote-only mode without those dependencies installed.
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline

        _pipeline = KPipeline(lang_code="a")
    return _pipeline


def _play_audio(path: Path):
    """Silent audio playback"""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]  # Windows-only; no shell.
        elif sys.platform == "darwin":
            subprocess.call(["afplay", str(path)])
        else:
            subprocess.call(["aplay", str(path)])
    except Exception:
        pass


async def _generate_remote(
    text: str, voice: str, speed: float, save_path: Optional[Path], play_audio: bool
) -> List[dict]:
    """Synthesize via an OpenAI-compatible /audio/speech endpoint."""
    url = REMOTE_BASE_URL.rstrip("/") + "/audio/speech"
    headers = {"Authorization": f"Bearer {REMOTE_API_KEY}"} if REMOTE_API_KEY else {}
    payload = {
        "model": REMOTE_MODEL,
        "input": text,
        "voice": voice,
        "speed": speed,
        "response_format": "wav",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        audio_bytes = resp.content

    with tempfile.TemporaryDirectory() as tmp_dir:
        out = (save_path / "segment_0.wav") if save_path else (Path(tmp_dir) / "segment_0.wav")
        out.write_bytes(audio_bytes)
        if play_audio:
            _play_audio(out)

    return [{"text": text}]


def _generate_local(
    text: str, voice: str, speed: float, save_path: Optional[Path], play_audio: bool
) -> List[dict]:
    """Synthesize with a locally loaded Kokoro pipeline."""
    voice_arg = voice
    if isinstance(voice, str) and Path(voice).exists():
        import torch

        try:
            voice_arg = torch.load(voice, weights_only=True)
        except Exception:
            raise ValueError("Invalid voice tensor")

    pipeline = _get_pipeline()
    try:
        generator = pipeline(text, voice=voice_arg, speed=speed, split_pattern=r"\n+")
    except Exception:
        raise RuntimeError("TTS failed")

    results: List[dict] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (graphemes, _, audio) in enumerate(generator):
            audio_numpy = audio.cpu().numpy()

            if save_path:
                sf.write(save_path / f"segment_{i}.wav", audio_numpy, 24000)

            if play_audio:
                temp_path = Path(tmp_dir) / f"segment_{i}.wav"
                sf.write(temp_path, audio_numpy, 24000)
                _play_audio(temp_path)

            results.append({"text": graphemes})

    return results


@mcp.tool()
async def generate_speech(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    save_path: str = None,
    play_audio: bool = False,
) -> List[dict]:
    save_dir: Optional[Path] = None
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    if REMOTE_BASE_URL:
        return await _generate_remote(text, voice, speed, save_dir, play_audio)
    return _generate_local(text, voice, speed, save_dir, play_audio)


if __name__ == "__main__":
    try:
        mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
    except Exception:
        sys.exit(1)
