# -*- mode: python ; coding: utf-8 -*-
import os

def get_datas():
    datas = []
    folder = "ffmpeg_bin"
    # Walk the ffmpeg_bin folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            full_path = os.path.join(root, file)
            # Compute the relative folder path inside ffmpeg_bin.
            rel_path = os.path.relpath(root, folder)
            # Set target folder: if rel_path is '.', then use just 'ffmpeg_bin'
            if rel_path == '.':
                target = folder
            else:
                target = os.path.join(folder, rel_path)
            datas.append((full_path, target))
    return datas

a = Analysis(
    ['AI-PatternGenerator.py'],
    pathex=[],
    binaries=[],
    datas=get_datas(),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AI-PatternGenerator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True
)