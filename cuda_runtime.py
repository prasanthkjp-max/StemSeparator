"""On-demand CUDA runtime for GPU-accelerated Demucs separation.

The packaged exe ships CPU-only torch (~195 MB) so it runs on any machine and
stays under GitHub's 2 GB release-asset limit. When an NVIDIA GPU is present,
the app can set up a separate CUDA environment (torch cu126 + demucs, ~2.5 GB,
one-time download from PyPI) in ``%LOCALAPPDATA%\\StemSeparator\\cuda-env`` and
run separation there via a subprocess — roughly 5-10x faster on GPU.

The CUDA environment is a real Python venv (created with the system Python),
so it cannot be imported in-process: the frozen exe bundles CPU torch. Instead
``run_separation`` spawns ``cuda_worker.py`` inside that venv and parses a
simple line protocol from its stdout:

    PROGRESS|<0..1>|<message>
    RESULT|<stem>|<wav path>
    ERROR|<message>
    DONE
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
# Pin torch + torchaudio to the SAME version: torchaudio must match torch
# exactly or ``import torchaudio`` fails (which silently disables demucs).
# The cu126 index currently ships torch up to 2.13.0 but torchaudio only to
# 2.11.0, so unpinned installs produce a broken pair (torch 2.13 + torchaudio
# 2.11) — exactly the bug that shipped in v1.1.0.
TORCH_VERSION = "2.11.0"
DEMUCS_VERSION = "4.1.0"

ProgressCb = Callable[[float, str], None]
LogCb = Callable[[str], None]


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def cuda_env_dir() -> Path:
    """Directory of the CUDA venv (override with STEMSEP_CUDA_ENV for tests)."""
    override = os.environ.get("STEMSEP_CUDA_ENV")
    if override:
        return Path(override)
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    return base / "StemSeparator" / "cuda-env"


def _resource(name: str) -> Path:
    """Locate a bundled script — works in source and in the frozen exe."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / name  # noqa: SLF001 - PyInstaller convention
    return Path(__file__).parent / name


def gpu_available() -> bool:
    """True if an NVIDIA GPU is present (nvidia-smi ships with the driver)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_creation_flags(),
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:  # noqa: BLE001 - any failure means "no GPU we can use"
        return False


def gpu_name() -> str:
    """NVIDIA GPU model name, or empty string."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_creation_flags(),
        )
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


_HEALTH_CHECK = (
    "import torch, torchaudio; from demucs.api import Separator; print(torch.cuda.is_available())"
)


def is_setup() -> bool:
    """True if the CUDA venv is healthy: torch+torchaudio match and demucs imports."""
    pyexe = cuda_env_dir() / "Scripts" / "python.exe"
    if not pyexe.is_file():
        return False
    try:
        r = subprocess.run(
            [str(pyexe), "-c", _HEALTH_CHECK],
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=_creation_flags(),
        )
        return r.returncode == 0 and r.stdout.strip() == "True"
    except Exception:  # noqa: BLE001
        return False


def find_system_python() -> str | None:
    """Locate a system Python able to create venvs (needed for setup).

    Prefers Python 3.11 — the same version the exe is built with — so the
    CUDA venv matches the frozen app's ABI.
    """
    candidates = [
        ["py", "-3.11"],
        ["py", "-3"],
        ["py"],
        ["python"],
        ["python3"],
    ]
    for cmd in candidates:
        try:
            r = subprocess.run(
                [*cmd, "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=_creation_flags(),
            )
            if r.returncode == 0:
                return cmd[0]
        except Exception:  # noqa: BLE001
            continue
    return None


def _run_streamed(cmd: list[str], log_cb: LogCb) -> None:
    """Run a command, streaming its output to ``log_cb``; raise on failure."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_cb(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def setup(progress_cb: ProgressCb, log_cb: LogCb) -> None:
    """Create the CUDA venv and install torch cu126 + demucs (one-time)."""
    env_dir = cuda_env_dir()
    env_dir.mkdir(parents=True, exist_ok=True)
    pyexe = env_dir / "Scripts" / "python.exe"

    py = find_system_python()
    if py is None:
        raise RuntimeError(
            "No system Python found. Install Python 3.11+ from python.org "
            "to enable GPU acceleration."
        )

    if not pyexe.is_file():
        progress_cb(0.05, "Creating CUDA environment…")
        _run_streamed([py, "-m", "venv", str(env_dir)], log_cb)

    progress_cb(0.10, "Upgrading pip…")
    _run_streamed([str(pyexe), "-m", "pip", "install", "--upgrade", "pip"], log_cb)

    progress_cb(0.15, "Downloading CUDA torch (~2.5 GB, one-time)…")
    _run_streamed(
        [
            str(pyexe),
            "-m",
            "pip",
            "install",
            f"torch=={TORCH_VERSION}",
            f"torchaudio=={TORCH_VERSION}",
            "--index-url",
            TORCH_INDEX,
        ],
        log_cb,
    )

    progress_cb(0.75, "Installing demucs…")
    _run_streamed(
        [str(pyexe), "-m", "pip", "install", f"demucs=={DEMUCS_VERSION}", "soundfile"],
        log_cb,
    )

    # Copy the worker scripts into the venv so the subprocess can import them.
    shutil.copy(_resource("separator.py"), env_dir / "separator.py")
    shutil.copy(_resource("cuda_worker.py"), env_dir / "cuda_worker.py")

    progress_cb(0.95, "Verifying CUDA + demucs…")
    r = subprocess.run(
        [str(pyexe), "-c", _HEALTH_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
        creationflags=_creation_flags(),
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Environment check failed — demucs could not import:\n{r.stderr.strip()[-500:]}"
        )
    if r.stdout.strip() != "True":
        raise RuntimeError(
            "CUDA verification failed — torch could not see the GPU. "
            "Check that the NVIDIA driver is up to date."
        )
    progress_cb(1.0, "CUDA ready.")


def run_separation(
    audio_path: str,
    output_dir: str,
    model: str,
    progress_cb: ProgressCb,
    log_cb: LogCb,
) -> dict[str, str]:
    """Run separation in the CUDA venv subprocess; return {stem: wav_path}."""
    env_dir = cuda_env_dir()
    pyexe = env_dir / "Scripts" / "python.exe"
    if not pyexe.is_file():
        raise RuntimeError("CUDA environment not set up. Enable GPU acceleration first.")

    # Run the worker from its copy INSIDE the venv (not the exe's extraction
    # dir): the extraction dir is full of Python 3.11 binaries that conflict
    # with the venv's Python version. Refresh the copies so they always match
    # the exe, even if the venv predates this version.
    shutil.copy(_resource("separator.py"), env_dir / "separator.py")
    shutil.copy(_resource("cuda_worker.py"), env_dir / "cuda_worker.py")

    proc = subprocess.Popen(
        [str(pyexe), str(env_dir / "cuda_worker.py"), audio_path, output_dir, model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
    )
    assert proc.stdout is not None

    results: dict[str, str] = {}
    error: str | None = None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("PROGRESS|"):
            _, p, msg = line.split("|", 2)
            with contextlib.suppress(ValueError):
                progress_cb(float(p), msg)
        elif line.startswith("RESULT|"):
            _, stem, path = line.split("|", 2)
            results[stem] = path
        elif line.startswith("ERROR|"):
            error = line[len("ERROR|") :]
        else:
            log_cb(line)

    proc.wait()
    if error:
        raise RuntimeError(error)
    if proc.returncode != 0:
        raise RuntimeError(f"CUDA worker exited with code {proc.returncode}")
    return results
