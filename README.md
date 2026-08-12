# StemSeparator

Standalone **Demucs stem separation** app — the heavy ML piece split out of
[Pitch2MIDI](https://github.com/prasanthkjp-max/Pitch2MIDI) so the main app
stays small and updates stay fast.

Splits a full mix into stems (**vocals / drums / bass / other**, or 6/8 stems
with the fine-tuned models) so each can be transcribed separately — much
better MIDI results on dense, multi-instrument music than a single pass on
the mix.

## Workflow

1. **StemSeparator**: drop an audio file → pick a model → Separate
2. Output: `<song>_vocals.wav`, `<song>_drums.wav`, `<song>_bass.wav`, `<song>_other.wav`
3. **Pitch2MIDI**: drop those stem WAVs into the batch queue → Convert All

## Install (from source)

Requires Python 3.10+ and FFmpeg on PATH (Demucs uses it for non-WAV input).

```bash
python -m venv venv
venv\Scripts\activate
# CPU-only torch (smaller exe, no GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python main.py
```

For GPU (CUDA) support, install a CUDA torch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Build the exe

```bash
pip install pyinstaller
pyinstaller --clean build.spec   # outputs to dist/StemSeparator.exe
```

## GPU acceleration (on-demand CUDA)

The packaged exe ships **CPU-only torch** (~195 MB) so it runs on any machine
and stays under GitHub's 2 GB release-asset limit. If an NVIDIA GPU is
detected, the app shows an **Enable GPU acceleration…** button:

1. One-time download of CUDA torch + demucs (~2.5 GB) into
   `%LOCALAPPDATA%\StemSeparator\cuda-env` (requires a system Python 3.11+)
2. Separation then runs on the GPU — roughly **5-10x faster**
3. The CPU fallback stays available; the exe itself never grows

## Models

| Model | Stems | Notes |
|---|---|---|
| `htdemucs` | 4 | default, fast |
| `htdemucs_ft` | 4 | fine-tuned, better quality, slower |
| `htdemucs_6s` | 6 | adds guitar + piano |
| `htdemucs_8s` | 8 | adds keys + choir |

First run downloads the model weights (~80–300 MB) from Hugging Face.

## Notes

- Cancellation is cooperative: an in-flight Demucs pass runs to completion,
  but stem writes stop at the next checkpoint.
- The app auto-detects CUDA vs CPU and shows the device in the UI.
