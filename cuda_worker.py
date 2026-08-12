"""CUDA worker — runs inside the CUDA venv as a subprocess.

The main app (frozen with CPU torch) cannot import CUDA torch in-process, so
separation runs here, in the CUDA environment, and reports progress over
stdout using the line protocol parsed by ``cuda_runtime.run_separation``:

    PROGRESS|<0..1>|<message>
    RESULT|<stem>|<wav path>
    ERROR|<message>
    DONE
"""

from __future__ import annotations

import sys

# Fail fast with a clear diagnostic instead of separator.py's generic
# "Demucs is not installed" (which masks torch/torchaudio version mismatch).
try:
    import torch  # noqa: F401 - importing validates the torch/torchaudio pair
    import torchaudio  # noqa: F401 - must match torch's version exactly
    from demucs.api import Separator, save_audio  # noqa: F401

    from separator import StemSeparator
except Exception as exc:  # noqa: BLE001 - report any failure to the parent
    print(f"ERROR|Environment broken: {exc}", flush=True)
    sys.exit(1)


def main() -> int:
    if len(sys.argv) != 4:
        print("ERROR|usage: cuda_worker.py <audio> <outdir> <model>", flush=True)
        return 2
    audio, outdir, model = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        sep = StemSeparator(model=model, device="cuda")
        results = sep.separate(
            audio,
            outdir,
            progress_callback=lambda p, m: print(f"PROGRESS|{p}|{m}", flush=True),
        )
        for stem, path in results.items():
            print(f"RESULT|{stem}|{path}", flush=True)
        print("DONE", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - report any failure to the parent
        print(f"ERROR|{exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
