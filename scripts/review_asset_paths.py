"""Relocate frozen Windows asset locators without changing manifest contents."""
import os
from pathlib import Path, PureWindowsPath

ASSET_ROOT=Path(os.environ.get('PINA_ASSET_ROOT',r'E:\3dconformergenerationcode'))

def asset_path(value, root=None):
    root=Path(root) if root is not None else ASSET_ROOT
    text=PureWindowsPath(str(value)).as_posix() if '\\' in str(value) else str(value)
    marker='/3dconformergenerationcode/'
    if marker in text:
        return root/Path(text.split(marker,1)[1])
    return Path(value)
