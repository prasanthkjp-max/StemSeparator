# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for StemSeparator (standalone Demucs app)."""

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Worker scripts copied into the on-demand CUDA venv at setup time.
        ('separator.py', '.'),
        ('cuda_worker.py', '.'),
    ],
    hiddenimports=[
        'demucs',
        'demucs.api',
        'demucs.pretrained',
        'demucs.apply',
        'demucs.audio',
        'demucs.hdemucs',
        'demucs.htdemucs',
        'demucs.htdemucs_pack',
        'demucs.repitch',
        'demucs.separate',
        'demucs.states',
        'demucs.utils',
        'demucs.valid',
        'demucs.wwave',
        'torch',
        'torchaudio',
        'julius',
        'einops',
        'soundfile',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "numpy.testing",
        "pytest",
        "IPython",
        "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='StemSeparator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=None,
)
