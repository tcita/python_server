# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

a = Analysis(
    ['server.py'],
    pathex=[],  # 使用项目的根路径
    binaries=[],
    datas=[
        ('trained/best_genome.pkl', 'trained'),  # 确保文件路径正确
        ('trained/move_predictor.pth', 'trained')  # 确保文件路径正确
    ],
    hiddenimports=[],  # 如果需要，可以在这里添加其他需要的模块
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
    name='server',
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
)
