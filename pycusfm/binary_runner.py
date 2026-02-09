# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import os
import pkg_resources
import argparse
import sys
from pathlib import Path
from .command_runner import CommandRunner

try:
    from .runfiles_helper import get_binary_dir, get_config_dir
    HAVE_RUNFILES_HELPER = True
except ImportError:
    HAVE_RUNFILES_HELPER = False


def setup_env(package_path=None):
    if package_path is None:
        if os.environ.get('PYCUSFM_DIR'):
            package_path = Path(os.environ.get('PYCUSFM_DIR'))
        else:
            try:
                package_path = Path(
                    pkg_resources.resource_filename('pycusfm', ''))
            except (TypeError, AttributeError, ImportError, Exception):
                # Fallback: get the project root directory (parent of the pycusfm package directory)
                package_path = Path(__file__).parent
                print(
                    f"pkg_resources failed, using fallback path: {package_path}"
                )

    return package_path


def binary_runner(binary_name, args, package_path=None):
    # Try to use Bazel runfiles helper first
    if HAVE_RUNFILES_HELPER:
        try:
            binary_dir = get_binary_dir()
            config_dir = get_config_dir()
        except Exception:
            # Fall back to traditional method if runfiles detection fails
            package_path = setup_env(package_path)
            binary_dir = os.path.join(package_path, "bin")
            config_dir = os.path.join(package_path, "configs/isaac")
    else:
        package_path = setup_env(package_path)
        binary_dir = os.path.join(package_path, "bin")
        config_dir = os.path.join(package_path, "configs/isaac")
    dry_run = False
    add_tensorrt_path = True

    runner = CommandRunner(
        binary_dir=binary_dir,
        config_dir=config_dir,
        dry_run=dry_run,
        add_tensorrt_path=add_tensorrt_path,
        runtime_log_dir=None)

    runner.run_binary(binary_name, args)


def main():
    parser = argparse.ArgumentParser(
        description='Run pycusfm binary with arguments')
    parser.add_argument('--binary_name', help='Name of the binary to run')
    parser.add_argument(
        '--args', help='Arguments string to pass to the binary')

    # Parse command line arguments
    parsed_args = parser.parse_args()

    # Split the args string into a list
    args_list = parsed_args.args.split() if parsed_args.args else []

    package_path = setup_env()
    binary_runner(parsed_args.binary_name, args_list, package_path)


if __name__ == "__main__":
    main()
