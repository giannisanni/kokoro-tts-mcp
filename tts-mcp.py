import sys
import os
import logging
import subprocess
import tempfile
from typing import List
import torch
import soundfile as sf
from kokoro import KPipeline
from mcp.server.fastmcp import FastMCP
from pathlib import Path

# Disable ALL logging
logging.disable(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
logging.captureWarnings(True)

# Initialize components
mcp = FastMCP("kokoro-tts")
pipeline = KPipeline(lang_code='a')

def _play_audio(path: Path):
    """Silent audio playback"""
    try:
        if sys.platform == "win32":
            subprocess.call(["start", str(path)], shell=True)
        elif sys.platform == "darwin":
            subprocess.call(["afplay", str(path)])
        else:
            subprocess.call(["aplay", str(path)])
    except:
        pass

@mcp.tool()
async def generate_speech(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    save_path: str = None,
    play_audio: bool = False
) -> List[dict]:
    results = []
    
    voice_tensor = None
    if isinstance(voice, str) and Path(voice).exists():
        try:
            voice_tensor = torch.load(voice, weights_only=True)
        except:
            raise ValueError("Invalid voice tensor")

    if save_path:
        (save_path := Path(save_path)).mkdir(parents=True, exist_ok=True)

    try:
        generator = pipeline(text, voice=voice_tensor or voice, speed=speed, split_pattern=r'\n+')
    except:
        raise RuntimeError("TTS failed")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (graphemes, _, audio) in enumerate(generator):
            audio_numpy = audio.cpu().numpy()
            
            if save_path:
                sf.write(save_path/f'segment_{i}.wav', audio_numpy, 24000)
            
            if play_audio:
                # Fixed temp_path definition
                temp_path = Path(tmp_dir) / f'segment_{i}.wav'
                sf.write(temp_path, audio_numpy, 24000)
                _play_audio(temp_path)

            results.append({'text': graphemes})
    
    return results

if __name__ == "__main__":
    try:
        mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))
    except:
        sys.exit(1)
