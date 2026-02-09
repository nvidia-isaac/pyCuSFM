# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import os
from .command_runner import CommandRunner
from .runfiles_helper import BazelBinaryDir


class BazelCommandRunner(CommandRunner):
    """CommandRunner that works with Bazel's runfiles structure."""

    def __init__(
            self,
            binary_dir,
            config_dir,
            dry_run=False,
            add_tensorrt_path=True,
            runtime_log_dir=None):
        # Store binary_dir before calling parent __init__
        self._bazel_binary_dir = binary_dir if isinstance(
            binary_dir, BazelBinaryDir) else None

        # For parent __init__, convert BazelBinaryDir to string to avoid os.path operations
        if isinstance(binary_dir, BazelBinaryDir):
            # Use a dummy string path that won't be used
            super().__init__(
                str(binary_dir), config_dir, dry_run, add_tensorrt_path,
                runtime_log_dir)
            # Override with the actual BazelBinaryDir object
            self.binary_dir = self._bazel_binary_dir
        else:
            super().__init__(
                binary_dir, config_dir, dry_run, add_tensorrt_path,
                runtime_log_dir)

    def run_binary(
            self,
            binary_name,
            args,
            logs_file_path=None,
            ensure_logs_directory_exists=False,
            exit_on_failure=True):
        if not self.binary_dir:
            print("Error: CommandRunner is not initialized.")
            if exit_on_failure:
                import sys
                sys.exit(1)
            return 1

        # Check if binary_dir is a BazelBinaryDir instance
        if isinstance(self.binary_dir, BazelBinaryDir):
            # Use the __truediv__ operator to locate the binary
            binary_path = self.binary_dir / binary_name
        else:
            # Standard path join for non-Bazel environments
            binary_path = os.path.join(str(self.binary_dir), binary_name)

        # Set up environment with LD_LIBRARY_PATH
        env = os.environ.copy()

        # Set up lib_dir path (always needed)
        self._setup_lib_dir_path(env)

        # Set up TensorRT library paths (optional)
        self._setup_tensorrt_library_path(env)

        # Set up cuVSLAM library path for Bazel runfiles
        if isinstance(self.binary_dir, BazelBinaryDir):
            self._setup_cuvslam_library_path(env)

        # --- Runtime logging ---
        import time
        start_time = time.time()
        result = self.run_command(
            binary_path,
            args,
            logs_file_path,
            ensure_logs_directory_exists,
            env=env,
            exit_on_failure=exit_on_failure,
        )
        end_time = time.time()
        runtime = end_time - start_time

        # Log runtime if enabled and not dry_run
        if self.runtime_log_dir and not self.dry_run:
            os.makedirs(self.runtime_log_dir, exist_ok=True)
            runtime_csv_path = os.path.join(
                self.runtime_log_dir, "runtime.csv")
            command = f"{binary_name} {' '.join(args)}"

            # Use lock to prevent race conditions when multiple threads write to the same CSV file
            with self._runtime_log_lock:
                # Check if CSV file exists and has headers
                file_exists = os.path.exists(runtime_csv_path)

                try:
                    import csv
                    with open(runtime_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)

                        # Write header if file doesn't exist
                        if not file_exists:
                            writer.writerow(['command', 'runtime_seconds'])

                        # Write the runtime data
                        writer.writerow([command, runtime])

                except Exception as e:
                    print(f"Warning: Could not write runtime data to CSV: {e}")

        return result

    def _setup_cuvslam_library_path(self, env):
        """Set up cuVSLAM library path in the environment for Bazel runfiles.

        Args:
            env: Environment dictionary to modify
        """
        if not isinstance(self.binary_dir, BazelBinaryDir):
            return

        from pathlib import Path

        # Find cuVSLAM library in external repositories
        # External repos in Bazel 6+ are prefixed with +_repo_rules2+
        runfiles_root = self.binary_dir.runfiles_root
        cuvslam_lib_paths = [
            runfiles_root.parent / '+_repo_rules2+cuvslam_lib_x86_64' / 'file',
            runfiles_root.parent / '+_repo_rules2+cuvslam_lib_aarch64' /
            'file',
        ]

        for lib_path in cuvslam_lib_paths:
            if lib_path.exists():
                lib_dir = str(
                    lib_path
                )  # lib_path already points to the 'file/' directory
                if 'LD_LIBRARY_PATH' in env:
                    env['LD_LIBRARY_PATH'] = f"{lib_dir}:{env['LD_LIBRARY_PATH']}"
                else:
                    env['LD_LIBRARY_PATH'] = lib_dir
                print(f"Added cuVSLAM library path: {lib_dir}")
                break
