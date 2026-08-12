"""StemSeparator main window — drag & drop audio, pick a model, separate.

The heavy Demucs work runs in a QThread so the UI stays responsive; progress
is reported via signals. Output stems land in the chosen folder as
``<song>_vocals.wav``, ``<song>_drums.wav``, ``<song>_bass.wav``,
``<song>_other.wav`` — ready to drop into Pitch2MIDI's batch queue.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import cuda_runtime
from separator import StemSeparator, best_device, demucs_available

MODELS = [
    "htdemucs",  # 4 stems, default
    "htdemucs_ft",  # 4 stems, fine-tuned (better quality, slower)
    "htdemucs_6s",  # 6 stems (vocals/drums/bass/other/guitar/piano)
    "htdemucs_8s",  # 8 stems (adds keys/choir)
]


class CudaSetupWorker(QThread):
    """Set up the on-demand CUDA environment off the UI thread."""

    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            cuda_runtime.setup(
                progress_cb=lambda p, m: self.progress.emit(p, m),
                log_cb=self.log.emit,
            )
            self.finished_ok.emit()
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            self.failed.emit(str(exc))


class CudaSeparationWorker(QThread):
    """Run separation in the CUDA venv subprocess off the UI thread."""

    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)  # {stem_name: wav_path}
    failed = pyqtSignal(str)

    def __init__(self, audio_path: str, output_dir: str, model: str) -> None:
        super().__init__()
        self._audio_path = audio_path
        self._output_dir = output_dir
        self._model = model

    def run(self) -> None:
        try:
            results = cuda_runtime.run_separation(
                self._audio_path,
                self._output_dir,
                self._model,
                progress_cb=lambda p, m: self.progress.emit(p, m),
                log_cb=self.log.emit,
            )
            self.finished_ok.emit(results)
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            self.failed.emit(str(exc))


class SeparationWorker(QThread):
    """Run Demucs separation off the UI thread."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(dict)  # {stem_name: wav_path}
    failed = pyqtSignal(str)

    def __init__(self, audio_path: str, output_dir: str, model: str, device: str) -> None:
        super().__init__()
        self._audio_path = audio_path
        self._output_dir = output_dir
        self._model = model
        self._device = device
        self._separator: StemSeparator | None = None

    def cancel(self) -> None:
        if self._separator is not None:
            self._separator.cancel()

    def run(self) -> None:
        try:
            sep = StemSeparator(model=self._model, device=self._device)
            self._separator = sep
            results = sep.separate(
                self._audio_path,
                self._output_dir,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(results)
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StemSeparator — Demucs")
        self.resize(640, 480)
        self.setAcceptDrops(True)

        self._worker: SeparationWorker | None = None
        self._cuda_setup_worker: CudaSetupWorker | None = None
        self._cuda_worker: CudaSeparationWorker | None = None
        self._last_output_dir: Path | None = None

        self._build_ui()
        self._update_device_label()
        self._set_running(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        form = QFormLayout()

        # Input file
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Drop an audio file here, or browse…")
        self._input_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_edit, 1)
        input_row.addWidget(browse_btn)
        form.addRow("Audio file:", input_row)

        # Output folder
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Where the stem WAVs go (default: next to the audio)")
        self._output_edit.setReadOnly(True)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._on_browse_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self._output_edit, 1)
        out_row.addWidget(out_btn)
        form.addRow("Output folder:", out_row)

        # Model
        self._model_combo = QComboBox()
        for m in MODELS:
            self._model_combo.addItem(m)
        self._model_combo.setToolTip(
            "htdemucs: 4 stems (default)\n"
            "htdemucs_ft: fine-tuned, better quality, slower\n"
            "htdemucs_6s: adds guitar + piano\n"
            "htdemucs_8s: adds keys + choir"
        )
        form.addRow("Model:", self._model_combo)

        # Device
        self._device_label = QLabel()
        form.addRow("Device:", self._device_label)

        # GPU acceleration (on-demand CUDA)
        self._cuda_btn = QPushButton("Enable GPU acceleration…")
        self._cuda_btn.setToolTip(
            "Detected an NVIDIA GPU. Downloads CUDA torch + demucs (~2.5 GB, "
            "one-time) for 5-10x faster separation."
        )
        self._cuda_btn.clicked.connect(self._on_enable_cuda)
        self._cuda_btn.setVisible(False)
        form.addRow("", self._cuda_btn)

        root.addLayout(form)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        self._status = QLabel("Ready.")
        root.addWidget(self._status)

        # Buttons
        self._start_btn = QPushButton("Separate")
        self._start_btn.clicked.connect(self._on_start)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._open_btn = QPushButton("Open output folder")
        self._open_btn.clicked.connect(self._on_open_output)
        self._open_btn.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._open_btn)
        root.addLayout(btn_row)

        # Log
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        root.addWidget(self._log_view, 1)

        self._log("Drop an audio file or click Browse to start.")

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        urls = event.mimeData().urls()
        if urls:
            self._set_input(urls[0].toLocalFile())

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose audio file",
            "",
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.aiff *.wma *.opus);;All files (*)",
        )
        if path:
            self._set_input(path)

    def _on_browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if folder:
            self._output_edit.setText(folder)

    def _set_input(self, path: str) -> None:
        if not path:
            return
        p = Path(path)
        if not p.is_file():
            QMessageBox.warning(self, "StemSeparator", f"File not found:\n{p}")
            return
        self._input_edit.setText(str(p))
        # Default output folder: next to the audio, in a "stems" subfolder.
        if not self._output_edit.text():
            self._output_edit.setText(str(p.parent / "stems"))
        self._log(f"Input: {p}")

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._cuda_worker is not None and self._cuda_worker.isRunning():
            return
        audio = self._input_edit.text().strip()
        if not audio:
            QMessageBox.information(self, "StemSeparator", "Choose an audio file first.")
            return
        out = self._output_edit.text().strip() or str(Path(audio).parent / "stems")
        model = self._model_combo.currentText()

        self._last_output_dir = Path(out)
        self._progress.setValue(0)
        self._set_running(True)

        if cuda_runtime.is_setup():
            self._log(f"Starting separation on GPU — model: {model}")
            self._cuda_worker = CudaSeparationWorker(audio, out, model)
            self._cuda_worker.progress.connect(self._on_progress)
            self._cuda_worker.log.connect(self._log)
            self._cuda_worker.finished_ok.connect(self._on_finished)
            self._cuda_worker.failed.connect(self._on_failed)
            self._cuda_worker.start()
        else:
            device = best_device()
            self._log(f"Starting separation on CPU — model: {model}")
            self._worker = SeparationWorker(audio, out, model, device)
            self._worker.progress.connect(self._on_progress)
            self._worker.finished_ok.connect(self._on_finished)
            self._worker.failed.connect(self._on_failed)
            self._worker.start()

    def _on_enable_cuda(self) -> None:
        if self._cuda_setup_worker is not None and self._cuda_setup_worker.isRunning():
            return
        reply = QMessageBox.question(
            self,
            "Enable GPU acceleration",
            "This downloads CUDA torch + demucs (~2.5 GB, one-time) and sets up "
            "a GPU environment for 5-10x faster separation.\n\nContinue?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._cuda_btn.setEnabled(False)
        self._cuda_btn.setText("Setting up GPU environment…")
        self._set_running(True)
        self._cuda_setup_worker = CudaSetupWorker()
        self._cuda_setup_worker.progress.connect(self._on_progress)
        self._cuda_setup_worker.log.connect(self._log)
        self._cuda_setup_worker.finished_ok.connect(self._on_cuda_setup_done)
        self._cuda_setup_worker.failed.connect(self._on_cuda_setup_failed)
        self._cuda_setup_worker.start()

    def _on_cuda_setup_done(self) -> None:
        self._cuda_setup_worker = None
        self._set_running(False)
        self._cuda_btn.setVisible(False)
        self._update_device_label()
        self._status.setText("GPU acceleration ready.")
        self._log("GPU acceleration enabled — separation will use CUDA.")

    def _on_cuda_setup_failed(self, error_msg: str) -> None:
        self._cuda_setup_worker = None
        self._set_running(False)
        self._cuda_btn.setEnabled(True)
        self._cuda_btn.setText("Enable GPU acceleration…")
        self._status.setText("GPU setup failed.")
        self._log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "StemSeparator", f"GPU setup failed:\n{error_msg}")

    def _on_cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._log("Cancelling… (finishes current stem write)")
        elif self._cuda_worker is not None and self._cuda_worker.isRunning():
            self._cuda_worker.terminate()
            self._log("Cancelling…")
        elif self._cuda_setup_worker is not None and self._cuda_setup_worker.isRunning():
            self._cuda_setup_worker.terminate()
            self._log("Cancelling GPU setup…")

    def _on_progress(self, progress: float, msg: str) -> None:
        self._progress.setValue(int(progress * 100))
        self._status.setText(msg)
        self._log(msg)

    def _on_finished(self, results: dict) -> None:
        self._set_running(False)
        self._worker = None
        self._cuda_worker = None
        if not results:
            self._status.setText("Cancelled.")
            self._log("Separation cancelled — no stems written.")
            return
        self._status.setText(f"Done — {len(results)} stems.")
        self._open_btn.setEnabled(True)
        for stem, path in sorted(results.items()):
            self._log(f"  {stem}: {path}")
        self._log("Stems ready — drop them into Pitch2MIDI's batch queue to transcribe.")

    def _on_failed(self, error_msg: str) -> None:
        self._set_running(False)
        self._worker = None
        self._cuda_worker = None
        self._status.setText("Failed.")
        self._log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "StemSeparator", f"Separation failed:\n{error_msg}")

    def _on_open_output(self) -> None:
        if self._last_output_dir is not None and self._last_output_dir.is_dir():
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(self._last_output_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self._last_output_dir)])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_device_label(self) -> None:
        if not demucs_available():
            self._device_label.setText("Demucs not installed — see README")
            return
        if cuda_runtime.is_setup():
            self._device_label.setText(f"GPU ({cuda_runtime.gpu_name()}) — CUDA")
            return
        if cuda_runtime.gpu_available():
            self._device_label.setText(
                f"CPU — {cuda_runtime.gpu_name()} detected, enable CUDA for 5-10x speed"
            )
            self._cuda_btn.setVisible(True)
        else:
            self._device_label.setText("CPU (no NVIDIA GPU detected)")
            self._cuda_btn.setVisible(False)

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._model_combo.setEnabled(not running)
        if not running and self._cuda_setup_worker is None:
            self._cuda_btn.setEnabled(True)

    def _log(self, msg: str) -> None:
        self._log_view.appendPlainText(msg)
