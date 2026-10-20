"""
Runtime patch for Starlette StaticFiles case sensitivity on Windows
Fixes upstream issue where os.path.realpath normalizes 'Lib' with capital 'L'
while os.path.commonpath uses lowercase 'lib', causing Starlette to mistakenly
reject static assets as directory traversal and return index.html instead of JS/CSS bundles.
"""

import os
import sys

try:
    import starlette.staticfiles as sf

    _orig_lookup = sf.StaticFiles.lookup_path

    def _patched_lookup_path(self, path: str):
        for directory in self.all_directories:
            joined_path = os.path.join(directory, path)
            if self.follow_symlink:
                full_path = os.path.abspath(joined_path)
            else:
                full_path = os.path.realpath(joined_path)
            directory = os.path.realpath(directory)
            # Use os.path.normcase for Windows case-insensitive path comparisons
            if os.path.normcase(os.path.commonpath([full_path, directory])) != os.path.normcase(directory):
                continue
            try:
                return full_path, os.stat(full_path)
            except (FileNotFoundError, NotADirectoryError):
                continue
        return "", None

    sf.StaticFiles.lookup_path = _patched_lookup_path
except ImportError:
    pass
