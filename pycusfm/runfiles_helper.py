# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path


def get_runfiles_dir():
    """Get the Bazel runfiles directory if running under Bazel, otherwise return None."""
    # Check if we're running under Bazel
    if 'RUNFILES_DIR' in os.environ:
        runfiles_dir = Path(os.environ['RUNFILES_DIR'])
        # RUNFILES_DIR might point to the .runfiles directory itself or _main
        if runfiles_dir.name == '_main':
            return runfiles_dir
        else:
            return runfiles_dir / '_main'

    # Try to detect runfiles directory from the current file location
    # In Bazel, Python files are typically at: <runfiles>/_main/<package>/<file>
    current_file = Path(__file__).resolve()

    # Look for .runfiles directory in the path
    for parent in current_file.parents:
        if parent.name.endswith('.runfiles'):
            # Check if we're already in _main or need to add it
            if '_main' in str(current_file):
                # Find the _main directory
                for p in current_file.parents:
                    if p.name == '_main' and p.parent == parent:
                        return p
            return parent / '_main'
        # Also check if we're already in a _main directory under runfiles
        if parent.name == '_main':
            if parent.parent.name.endswith('.runfiles'):
                return parent

    return None


def get_binary_dir():
    """Get the binary directory, handling both Bazel and non-Bazel environments."""
    runfiles_dir = get_runfiles_dir()

    if runfiles_dir:
        # In Bazel, binaries are in the runfiles under their original paths
        # We need to return a custom object that can locate binaries
        return BazelBinaryDir(runfiles_dir)

    # Fallback for non-Bazel environments
    # Check if there's a symlink in the package directory
    package_dir = Path(__file__).parent
    bin_path = package_dir / 'bin'

    if bin_path.exists():
        return bin_path

    # If no symlink, go up to repository root and use build/tools
    # Assuming structure: <repo_root>/visual_mapping/runfiles_helper.py
    repo_root = package_dir.parent
    return repo_root / 'build' / 'tools'


class BazelBinaryDir:
    """Helper class to locate binaries in Bazel runfiles structure."""

    def __init__(self, runfiles_root):
        self.runfiles_root = Path(runfiles_root)
        # Known binary location directories in runfiles
        self.binary_search_dirs = [
            self.runfiles_root / 'tools' / 'visual' / 'cusfm',
            self.runfiles_root / 'tools' / 'visual' / 'cuvgl',
            self.runfiles_root / 'tools' / 'visual' / 'metrics',
            self.runfiles_root / 'tools' / 'visual' / 'utils',
        ]

        # Add cuVSLAM external repository paths for x86_64 and aarch64
        # These are downloaded via http_file in MODULE.bazel
        # External repos in Bazel 6+ are prefixed with +_repo_rules2+
        self.cuvslam_paths = [
            self.runfiles_root.parent /
            '+_repo_rules2+cuvslam_launcher_x86_64' / 'file',
            self.runfiles_root.parent /
            '+_repo_rules2+cuvslam_launcher_aarch64' / 'file',
        ]

    def __truediv__(self, binary_name):
        """Override / operator to locate binaries in runfiles."""
        # Try each known binary location
        for search_dir in self.binary_search_dirs:
            binary_path = search_dir / binary_name
            if binary_path.exists():
                return str(binary_path)

        # Check cuVSLAM external repository locations
        for cuvslam_dir in self.cuvslam_paths:
            cuvslam_binary = cuvslam_dir / binary_name
            if cuvslam_binary.exists():
                return str(cuvslam_binary)

        # Fallback: return expected path even if it doesn't exist
        # This will allow proper error messages
        return str(
            self.runfiles_root / 'tools' / 'visual' / 'utils' / binary_name)

    def __str__(self):
        # Return the runfiles_root path
        # Note: This is only for display purposes; actual binary location
        # should use __truediv__ operator via BazelCommandRunner
        return str(self.runfiles_root)


def get_model_dir():
    """Get the model directory, handling both Bazel and non-Bazel environments."""
    runfiles_dir = get_runfiles_dir()

    if runfiles_dir:
        return runfiles_dir / 'models'

    # Fallback for non-Bazel environments
    # Check if there's a symlink in the package directory
    package_dir = Path(__file__).parent
    model_path = package_dir / 'models'

    if model_path.exists():
        return model_path

    # If no symlink, go up to repository root
    # Assuming structure: <repo_root>/visual_mapping/runfiles_helper.py
    repo_root = package_dir.parent
    return repo_root / 'models'


def get_config_dir():
    """Get the config directory, handling both Bazel and non-Bazel environments."""
    runfiles_dir = get_runfiles_dir()

    if runfiles_dir:
        return runfiles_dir / 'configs' / 'isaac'

    # Fallback for non-Bazel environments
    # Check if there's a symlink in the package directory
    package_dir = Path(__file__).parent
    config_path = package_dir / 'configs'

    if config_path.exists():
        return config_path / 'isaac'

    # If no symlink, go up to repository root
    # Assuming structure: <repo_root>/visual_mapping/runfiles_helper.py
    repo_root = package_dir.parent
    return repo_root / 'configs' / 'isaac'
