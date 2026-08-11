"""
Stem separation using Demucs (optional dependency).

Splits a full mix into stems (vocals / drums / bass / other) so each can be
transcribed separately — dramatically better MIDI results on dense,
multi-instrument music (e.g. Indian film songs) than a single pass on the mix.

Demucs + PyTorch are heavy (~3 GB) and are NOT bundled in the packaged exe;
the feature degrades gracefully with a clear message when unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

try:
    import torch
    from demucs.api import Separator, save_audio

    DEMUCS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    DEMUCS_AVAILABLE = False


def demucs_available() -> bool:
    """True if Demucs + PyTorch are importable."""
    return DEMUCS_AVAILABLE


def best_device() -> str:
    """Prefer CUDA when a GPU is present, else CPU."""
    if DEMUCS_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class StemSeparator:
    """Separate an audio file into stems with Demucs.

    Parameters
    ----------
    model : str
        Demucs model name (default ``htdemucs`` — 4 stems).
    device : str, optional
        ``"cuda"`` or ``"cpu"``. Auto-detected when None.
    """

    STEMS = ("vocals", "drums", "bass", "other")

    def __init__(self, model: str = "htdemucs", device: Optional[str] = None) -> None:
        if not DEMUCS_AVAILABLE:
            raise RuntimeError("Demucs is not installed. Install it with: pip install demucs")
        self.model = model
        self.device = device or best_device()
        self._separator = None
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation.

        Cooperative only: the demucs API exposes no hook to interrupt an
        in-flight ``separate_audio_file()``, so that call still runs to
        completion. Cancellation takes effect at the next checkpoint — before
        inference starts, and between stem writes — which is enough to stop the
        (slow) disk writes and to avoid queueing stems the user no longer wants.
        """
        self._cancelled = True

    def _get_separator(self):
        if self._separator is None:
            self._separator = Separator(
                model=self.model,
                device=self.device,
                shifts=1,
                progress=False,
            )
        return self._separator

    def separate(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> dict[str, str]:
        """Separate ``audio_path`` into stems.

        Parameters
        ----------
        audio_path : str | Path
            Input audio file.
        output_dir : str | Path
            Directory for the stem WAV files.
        progress_callback : callable, optional
            Called with ``(progress: float, message: str)``.

        Returns
        -------
        dict
            ``{stem_name: wav_path}`` for each separated stem.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._cancelled:
            return {}

        if progress_callback:
            progress_callback(0.05, "Loading Demucs model…")
        sep = self._get_separator()

        if self._cancelled:
            return {}

        if progress_callback:
            progress_callback(0.15, "Separating stems…")
        _, separated = sep.separate_audio_file(str(audio_path))

        results: dict[str, str] = {}
        stem_count = max(1, len(separated))
        for i, (stem, tensor) in enumerate(separated.items()):
            if self._cancelled:
                break
            out = output_dir / f"{audio_path.stem}_{stem}.wav"
            save_audio(tensor, str(out), samplerate=sep.samplerate)
            results[str(stem)] = str(out)
            if progress_callback:
                progress_callback(
                    0.15 + 0.8 * (i + 1) / stem_count,
                    f"Saved {stem} stem",
                )

        if progress_callback:
            progress_callback(1.0, "Separation cancelled." if self._cancelled else "Separation complete.")
        return results
