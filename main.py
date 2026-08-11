"""StemSeparator — standalone Demucs stem separation app.

Splits a full mix into stems (vocals / drums / bass / other) so each can be
transcribed separately in Pitch2MIDI (or any other tool).

Run from source:
    python main.py

Build the exe:
    pyinstaller --clean build.spec
"""

import sys

from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("StemSeparator")
    app.setOrganizationName("Pitch2MIDI")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
