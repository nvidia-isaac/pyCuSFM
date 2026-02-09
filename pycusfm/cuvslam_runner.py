# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import yaml
import pkg_resources
from pathlib import Path
from .command_runner import CommandRunner
from .split_repeated_poses import RepeatedPoseSplitter
from .constants import (
    kFRAME_META_FILE, kFRAME_META_FILE_CUVSLAM,
    kSLAM_POSES_FILE, kODOM_POSES_FILE
)

logger = logging.getLogger(__name__)


def get_default_config_dir(package_path=None):
    if package_path is None:
        try:
            package_path = Path(pkg_resources.resource_filename('visual_mapping', ''))
        except Exception:
            package_path = Path(__file__).parent.parent

    try:
        from .runfiles_helper import get_config_dir
        return str(get_config_dir())
    except (ImportError, Exception):
        return str(package_path / 'configs/isaac')


class CuVSlamRunner:
    def __init__(self, binary_dir, config_dir=None, config_name=None, dry_run=False, log_dir=None):
        """
        Initialize CuVSlamRunner.

        Args:
            binary_dir: Directory containing cuvslam binaries.
            config_dir: Directory containing configuration files (passed to CommandRunner).
            config_name: Name of YAML configuration file (relative to config_dir, or absolute path).
            dry_run: If True, print commands instead of executing.
            log_dir: Directory for execution logs.
        """
        self.config_name = config_name
        self.dry_run = dry_run
        self.log_dir = log_dir

        if config_dir is None:
            config_dir = get_default_config_dir()

        # Initialize CommandRunner
        self.runner = CommandRunner(
            binary_dir=binary_dir,
            config_dir=config_dir,
            dry_run=dry_run,
            add_tensorrt_path=False,
            runtime_log_dir=log_dir
        )

    def load_config(self):
        """Load cuvslam configuration from YAML file."""
        if not self.config_name:
            return {}

        # Resolve path using CommandRunner logic (relative to config_dir, or absolute)
        config_path = self.runner.get_config_path(self.config_name)

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'cuvslam' in config:
                    return config['cuvslam']
                return {}
        except Exception as e:
            logger.warning(
                f"Failed to load cuvslam config from {config_path}: {e}"
            )
            return {}

    def _add_parameters(self, command, config, log_dir):
        """Add cuVSLAM parameters to command list."""
        if not config:
            return command

        # Core SLAM settings
        if 'cfg_enable_slam' in config:
            enable_slam = str(config["cfg_enable_slam"]).lower()
            command.append(f'--cfg_enable_slam={enable_slam}')
        if 'ros_frame_conversion' in config:
            ros_frame = str(config["ros_frame_conversion"]).lower()
            command.append(f'--ros_frame_conversion={ros_frame}')
        if 'print_format' in config:
            command.append(f'--print_format={config["print_format"]}')

        # Performance & throttling
        if 'max_fps' in config:
            command.append(f'--max_fps={config["max_fps"]}')
        if 'verbosity' in config:
            command.append(f'--verbosity={config["verbosity"]}')

        # Image processing & quality
        if 'cfg_denoising' in config:
            denoising = str(config["cfg_denoising"]).lower()
            command.append(f'--cfg_denoising={denoising}')
        if 'cfg_max_frame_delta_s' in config:
            command.append(f'--cfg_max_frame_delta_s={config["cfg_max_frame_delta_s"]}')
        if 'cfg_horizontal' in config:
            horizontal = str(config["cfg_horizontal"]).lower()
            command.append(f'--cfg_horizontal={horizontal}')

        # Image masking
        if 'border_bottom' in config:
            command.append(f'--border_bottom={config["border_bottom"]}')
        if 'border_left' in config:
            command.append(f'--border_left={config["border_left"]}')
        if 'border_right' in config:
            command.append(f'--border_right={config["border_right"]}')
        if 'border_top' in config:
            command.append(f'--border_top={config["border_top"]}')

        # SLAM algorithm configuration
        if 'cfg_multicam_mode' in config:
            command.append(f'--cfg_multicam_mode={config["cfg_multicam_mode"]}')
        if 'cfg_odom_mode' in config:
            command.append(f'--cfg_odom_mode={config["cfg_odom_mode"]}')
        if 'cfg_planar' in config:
            planar = str(config["cfg_planar"]).lower()
            command.append(f'--cfg_planar={planar}')
        if 'cfg_slam_max_map_size' in config:
            command.append(f'--cfg_slam_max_map_size={config["cfg_slam_max_map_size"]}')
        if 'cfg_sync_slam' in config:
            sync_slam = str(config["cfg_sync_slam"]).lower()
            command.append(f'--cfg_sync_slam={sync_slam}')

        # Processing & error handling
        if 'ignore_tracking_errors' in config:
            ignore_errors = str(config["ignore_tracking_errors"]).lower()
            command.append(f'--ignore_tracking_errors={ignore_errors}')
        if 'repeat' in config:
            command.append(f'--repeat={config["repeat"]}')
        if 'start_frame' in config:
            command.append(f'--start_frame={config["start_frame"]}')
        if 'cache_uncompressed' in config:
            cache = str(config["cache_uncompressed"]).lower()
            command.append(f'--cache_uncompressed={cache}')

        # Depth camera settings
        if 'cfg_enable_depth_stereo_tracking' in config:
            depth_tracking = str(config["cfg_enable_depth_stereo_tracking"]).lower()
            command.append(f'--cfg_enable_depth_stereo_tracking={depth_tracking}')
        if 'cfg_depth_camera' in config:
            command.append(f'--cfg_depth_camera={config["cfg_depth_camera"]}')
        if 'cfg_depth_scale_factor' in config:
            command.append(f'--cfg_depth_scale_factor={config["cfg_depth_scale_factor"]}')

        # Export & debug settings
        if 'cfg_enable_export' in config:
            enable_export = str(config["cfg_enable_export"]).lower()
            command.append(f'--cfg_enable_export={enable_export}')

        if 'debug_dump' in config:
            debug_dump_val = config["debug_dump"]
            # If boolean true, use the log directory
            if isinstance(debug_dump_val, bool) and debug_dump_val and log_dir:
                command.append(f'--debug_dump={log_dir}')
            elif debug_dump_val and not isinstance(debug_dump_val, bool):
                # If string path provided
                command.append(f'--debug_dump={debug_dump_val}')

        if 'shuttle' in config:
            shuttle = str(config["shuttle"]).lower()
            command.append(f'--shuttle={shuttle}')

        # Localization settings
        if 'loc_input_map' in config:
            command.append(f'--loc_input_map={config["loc_input_map"]}')
        if 'loc_input_hints' in config:
            command.append(f'--loc_input_hints={config["loc_input_hints"]}')
        if 'loc_hint_ts_format' in config:
            command.append(f'--loc_hint_ts_format={config["loc_hint_ts_format"]}')
        if 'loc_hint_noise' in config:
            command.append(f'--loc_hint_noise={config["loc_hint_noise"]}')
        if 'loc_random_rot' in config:
            random_rot = str(config["loc_random_rot"]).lower()
            command.append(f'--loc_random_rot={random_rot}')
        if 'loc_retries' in config:
            command.append(f'--loc_retries={config["loc_retries"]}')
        if 'loc_skip_frames' in config:
            command.append(f'--loc_skip_frames={config["loc_skip_frames"]}')
        if 'loc_start_frame' in config:
            command.append(f'--loc_start_frame={config["loc_start_frame"]}')
        if 'localize_forever' in config:
            localize_forever = str(config["localize_forever"]).lower()
            command.append(f'--localize_forever={localize_forever}')
        if 'localize_wait' in config:
            localize_wait = str(config["localize_wait"]).lower()
            command.append(f'--localize_wait={localize_wait}')
        if 'print_nan_on_failure' in config:
            print_nan = str(config["print_nan_on_failure"]).lower()
            command.append(f'--print_nan_on_failure={print_nan}')

        # Camera selection
        if 'cameras' in config and config["cameras"]:
            command.append(f'--cameras={config["cameras"]}')

        return command

    def run(self, input_dir, output_dir, override_frames_meta_file="", use_slam_pose=True):
        """Run CUVSLAM processing to generate pose estimates.

        Args:
            input_dir (str): Directory containing input images and metadata.
            output_dir (str): Output directory for CUVSLAM results.
            override_frames_meta_file (str, optional): Override frames metadata file path.
            use_slam_pose (bool): Whether to use SLAM poses or ODOM poses for the output.

        Returns:
            str: Path to keyframe metadata file with pose information.
        """
        # Determine paths
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        if override_frames_meta_file:
            keyframe_file_to_use = override_frames_meta_file
        else:
            keyframe_file_to_use = os.path.join(input_dir, kFRAME_META_FILE)

        # 1. Convert keyframe metadata to EDEX format
        self.runner.run_binary(
            'keyframe_metadata_to_edex_main', [
                "--keyframe_metadata_file", keyframe_file_to_use,
                "--output_edex_dir", input_dir
            ],
            logs_file_path=os.path.join(log_dir, 'keyframe_metadata_to_edex_main.txt')
        )

        # 2. Run cuvslam_api_launcher
        logger.info("Running cuvslam_api_launcher")
        cuvslam_config = self.load_config()

        # Check repeat count
        repeat_count = cuvslam_config.get('repeat', 1)

        # Adjust output filenames if repeating
        odom_filename = kODOM_POSES_FILE
        slam_filename = kSLAM_POSES_FILE
        if repeat_count > 1:
            odom_filename = "odom_poses_repeated.tum"
            slam_filename = "slam_poses_repeated.tum"

        cmd = [
            "--dataset", input_dir,
            "--output_map", os.path.join(output_dir, "cuvslam_map"),
            "--print_odom_poses", os.path.join(output_dir, odom_filename),
            "--print_slam_poses", os.path.join(output_dir, slam_filename),
        ]

        # Add default flags if not present in config to maintain backward compatibility
        if not cuvslam_config.get('print_format'):
            cmd.extend(["--print_format", "tum"])

        if 'ros_frame_conversion' not in cuvslam_config:
            cmd.append("--ros_frame_conversion=true")

        if 'cfg_enable_slam' not in cuvslam_config:
            cmd.append("--cfg_enable_slam")

        cmd = self._add_parameters(cmd, cuvslam_config, log_dir)

        self.runner.run_binary(
            'cuvslam_api_launcher',
            cmd,
            logs_file_path=os.path.join(log_dir, 'cuvslam_api_launcher.txt')
        )

        # 3. Post-process if repeating
        if repeat_count > 1:
            logger.info(f"Repeat count {repeat_count} > 1. Splitting repeated poses...")
            try:
                splitter = RepeatedPoseSplitter(
                    poses_dir=Path(output_dir),
                    edx_dir=Path(input_dir),
                    repeat_count=repeat_count
                )
                splitter.process_all_files()
            except Exception as e:
                logger.error(f"Failed to split repeated poses: {e}")
                raise

        # 4. Update poses in frames_meta.json
        use_pose_file = kSLAM_POSES_FILE if use_slam_pose else kODOM_POSES_FILE
        pose_file_path = os.path.join(output_dir, use_pose_file)

        if not os.path.isfile(pose_file_path) and not self.dry_run:
            raise RuntimeError(f"Expected pose file not found: {pose_file_path}")

        output_keyframe_metadata_file = os.path.join(output_dir, kFRAME_META_FILE_CUVSLAM)

        self.runner.run_binary(
            'update_keyframe_pose_main', [
                "--input_file", keyframe_file_to_use,
                "--output_file", output_keyframe_metadata_file,
                "--tum_pose_file", pose_file_path,
            ],
            logs_file_path=os.path.join(log_dir, 'update_keyframe_pose_main.txt')
        )

        return output_keyframe_metadata_file
