
block_cipher = None


a = Analysis(
    ['server.py'],  # 你的主程序入口文件
    pathex=[],
    binaries=[],
    datas=[
        ('trained', 'trained')  # 这是关键部分：将 'trained' 文件夹及其内容复制到打包后的 'trained' 目录中
    ],
    hiddenimports=[
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree',
        'sklearn.tree._utils',
        'waitress' # 明确告诉 PyInstaller 包含 waitress
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CardGameAI_Server', # 生成的 .exe 文件名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,         # 这是个控制台程序，所以设为 True。双击 exe 会显示命令行窗口。
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CardGameAI_Server', # 最终生成的文件夹名 (如果不是 --onefile 模式)
)