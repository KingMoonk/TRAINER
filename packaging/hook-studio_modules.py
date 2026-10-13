"""Collect every dynamically discovered model module in frozen builds.

The registry imports studio_modules with pkgutil.walk_packages at runtime, which static
analysis cannot follow. PyInstaller's submodule collector keeps new module files available
without maintaining a second list in the release workflow.
"""
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("studio_modules")
