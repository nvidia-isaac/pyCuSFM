# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class CommandRunner:

    def __init__(self, binary_dir: str, config_dir: str, dry_run=False, add_tensorrt_path=True, runtime_log_dir=None):
        self.binary_dir = binary_dir
        self.config_dir = config_dir
        self.dry_run = dry_run
        self.add_tensorrt_path = add_tensorrt_path
        self.runtime_log_dir = runtime_log_dir

        # Add lib_dir for packaged libraries
        # This will be relative to the binary_dir in the installed package
        if binary_dir and os.path.exists(os.path.dirname(binary_dir)):
            self.lib_dir = os.path.join(os.path.dirname(binary_dir), 'lib')
        else:
            self.lib_dir = None

        # Lock for thread-safe runtime logging
        self._runtime_log_lock = threading.Lock()

    def get_config_path(self, config_name):
        return os.path.join(self.config_dir, config_name)

    def get_config_folder(self):
        return self.config_dir

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            if self.dry_run:
                print(f"Dry run: Would create directory {directory}")
            else:
                os.makedirs(directory)
                print(f"Created local directory: {directory}")
        else:
            print(f"Local directory already exists: {directory}")

    def parent_directory_exists(self, file_path):
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            if self.dry_run:
                print(
                    f"Dry run: Parent directory {parent_dir} does not exist.")
                return False
            else:
                print(f"Parent directory {parent_dir} does not exist.")
                return False
        return True

    def _setup_lib_dir_path(self, env):
        """Set up lib_dir path in the environment.

        Args:
            env: Environment dictionary to modify

        Returns:
            None, modifies env in place
        """
        # Skip using packaged libraries to prefer system libraries
        if os.getenv('USE_SYSTEM_PROTOBUF', 'false').lower() == 'true':
            print("Skipping packaged libraries, using system protobuf")
            return

        if self.lib_dir and os.path.exists(self.lib_dir):
            print(f"Using packaged libraries from: {self.lib_dir}")
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = f"{self.lib_dir}:{env['LD_LIBRARY_PATH']}"
            else:
                env['LD_LIBRARY_PATH'] = self.lib_dir

    def _setup_tensorrt_library_path(self, env):
        """Set up TensorRT library paths in the environment.

        Args:
            env: Environment dictionary to modify

        Returns:
            None, modifies env in place
        """
        if not self.add_tensorrt_path:
            return

        # Get system TensorRT path
        system_tensorrt_path = None
        if 'TRT_LIB_PATH' in env:
            system_tensorrt_path = env['TRT_LIB_PATH']
        else:
            # add /usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH
            # if on aarch64, add /usr/lib/aarch64-linux-gnu
            if os.uname().machine == 'aarch64':
                system_tensorrt_path = "/usr/lib/aarch64-linux-gnu"
            else:
                system_tensorrt_path = "/usr/lib/x86_64-linux-gnu"
            print("TRT_LIB_PATH is not set, defaulting to system libraries: ",
                  system_tensorrt_path)

        # Add system TensorRT path to LD_LIBRARY_PATH
        if 'LD_LIBRARY_PATH' not in env:
            env['LD_LIBRARY_PATH'] = f"{system_tensorrt_path}"
        else:
            env['LD_LIBRARY_PATH'] = f"{system_tensorrt_path}:{env['LD_LIBRARY_PATH']}"

        # Check if TensorRT libraries might already be in the path
        tensorrt_in_path = False
        if 'LD_LIBRARY_PATH' in env:
            tensorrt_paths = [
                path for path in env['LD_LIBRARY_PATH'].split(':')
                if os.path.exists(path) and any(
                    os.path.exists(os.path.join(path, f"libnvinfer{suffix}"))
                    for suffix in ['.so', '.so.10', '.so.10.3.0'])
            ]
            if tensorrt_paths:
                print(f"TensorRT libraries already found in: {tensorrt_paths}")
                tensorrt_in_path = True

        # Optionally add tensor rt libs from python package tensorrt_libs if not already in path
        if not tensorrt_in_path:
            from pkg_resources import resource_filename
            tensorrt_libs = resource_filename('tensorrt_libs', '')
            print(f"Found TensorRT libraries at: {tensorrt_libs}")
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = f"{tensorrt_libs}:{env['LD_LIBRARY_PATH']}"
            else:
                env['LD_LIBRARY_PATH'] = tensorrt_libs

    def run_command(
            self,
            command_name,
            args,
            logs_file_path=None,
            ensure_logs_directory_exists=False,
            env=None,
            exit_on_failure=True):
        command = [command_name] + args

        if logs_file_path:
            log_dir = os.path.dirname(logs_file_path)
            if ensure_logs_directory_exists:
                self.ensure_directory_exists(log_dir)
            elif not self.parent_directory_exists(logs_file_path):
                print(f"Error: Log directory {log_dir} does not exist.")
                if exit_on_failure:
                    sys.exit(1)
                return 1

            command_str = " ".join(command)
            print(f"Running.. {command_str}")
            print(f'Logging to.. {logs_file_path}')

            # Redirect both stdout and stderr to the log file while also printing to console
            with open(logs_file_path, 'w') as log_file:
                process = subprocess.Popen(
                    command_str,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    text=True, env=env)

                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()  # Ensure immediate writing to file

                process.wait()
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env)

            for line in iter(process.stdout.readline, ""):
                print(line, end="")

            process.wait()

        # Print detailed exit information
        print(f"Process completed with return code: {process.returncode}")
        if process.returncode != 0:
            error_message = (
                f"Error: Command failed with status {process.returncode}: {' '.join(command)}"
            )
            print(error_message)
            if logs_file_path:
                with open(logs_file_path, "a") as log_file:
                    log_file.write(error_message + "\n")
                # Also print the last few lines of the log to help debug
                print(f"Last few lines of log file {logs_file_path}:")
                try:
                    with open(logs_file_path, 'r') as log_file:
                        lines = log_file.readlines()
                        for line in lines[-10:]:  # Print last 10 lines
                            print(f"  {line.rstrip()}")
                except Exception as e:
                    print(f"  Could not read log file: {e}")
            if exit_on_failure:
                sys.exit(1)
            return process.returncode
        else:
            print(f"Command completed successfully with exit code 0")

        return 0

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
                sys.exit(1)
            return 1

        binary_path = os.path.join(self.binary_dir, binary_name)

        # Set up environment with LD_LIBRARY_PATH
        env = os.environ.copy()

        # Set up lib_dir path (always needed)
        self._setup_lib_dir_path(env)

        # Set up TensorRT library paths (optional)
        self._setup_tensorrt_library_path(env)

        # --- Runtime logging ---
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
            runtime_csv_path = os.path.join(self.runtime_log_dir, "runtime.csv")
            command = f"{binary_name} {' '.join(args)}"

            # Use lock to prevent race conditions when multiple threads write to the same CSV file
            with self._runtime_log_lock:
                # Check if CSV file exists and has headers
                file_exists = os.path.exists(runtime_csv_path)

                try:
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

    def run_wget(self, wget_args, logs_file_path=None):
        wget_command = ["wget"] + wget_args

        if self.dry_run:
            print(f"Dry run: Would run wget command {' '.join(wget_command)}")
            return 0  # Return 0 to simulate successful execution

        if logs_file_path:
            command_str = " ".join(
                wget_command) + f" 2>&1 | tee {logs_file_path}"
            process = subprocess.Popen(
                command_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True,
            )
        else:
            process = subprocess.Popen(
                wget_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True)

        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            error_message = (
                f"Error: Command failed with status {process.returncode}: {' '.join(wget_command)}"
            )
            print(error_message)
            if logs_file_path:
                with open(logs_file_path, "a") as log_file:
                    log_file.write(error_message + "\n")
            sys.exit(1)

    def run_python(self, script_path, args, logs_file_path=None):
        command = ["python", script_path] + args
        if self.dry_run:
            print(f"Dry run: Would run command {command}")
            return

        if logs_file_path:
            command_str = " ".join(command) + f" 2>&1 | tee {logs_file_path}"
            print(f"Running.. {command_str}")
            process = subprocess.Popen(
                command_str,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True,
            )
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True)

        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            error_message = (
                f"Error: Command failed with status {process.returncode}: {' '.join(command)}"
            )
            print(error_message)
            if logs_file_path:
                with open(logs_file_path, "a") as log_file:
                    log_file.write(error_message + "\n")
            sys.exit(1)

    def remove_directory(self, dir_path):
        if self.dry_run:
            print(f"Dry run: Would remove directory {dir_path}")
        else:
            subprocess.run(
                ["rm", "-rf", dir_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            print(f"Removed directory {dir_path}")

    def run_binaries_parallel(self, commands, max_workers=4):
        print(f"Starting parallel execution of {len(commands)} commands with max_workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.run_binary, cmd["binary_name"], cmd["args"],
                    cmd["log_file"], False, False) for cmd in commands  # exit_on_failure=False
            ]

            print(f"Submitted {len(futures)} tasks to executor")

            failed_commands = []
            completed_count = 0

            for future in as_completed(futures):
                completed_count += 1
                print(f"Command {completed_count}/{len(futures)} completed")

                try:
                    result = future.result()
                    if result != 0:
                        failed_commands.append(result)
                        print(f"Command failed with return code {result}")
                    else:
                        print(f"Command succeeded")
                except Exception as e:
                    print(f"Command failed with exception: {e}")
                    failed_commands.append(-1)

            print(f"All {completed_count} commands completed")

            if failed_commands:
                return False
            else:
                return True
