"""PyInstaller hook: Panda3D loads its OpenGL display plugin by name at run time, so import analysis never sees it. Without it the 3D viewer dies with "No graphics pipe is available"."""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

binaries = [lib for lib in collect_dynamic_libs("panda3d") if "pandagl" in lib[0]]
datas = collect_data_files("panda3d")  # etc/Config.prc names the display plugin; models/ holds the default fonts
