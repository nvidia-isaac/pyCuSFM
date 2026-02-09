# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path
from .cuvslam_runner import CuVSlamRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run cuVSLAM standalone'
    )

    parser.add_argument(
        '--input_dir',
        required=True,
        help='Input directory containing images and metadata'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for cuVSLAM results'
    )
    parser.add_argument(
        '--config_file',
        default=None,
        help='Path to YAML configuration file for cuVSLAM'
    )
    parser.add_argument(
        '--binary_dir',
        default=None,
        help='Directory containing binaries'
    )
    parser.add_argument(
        '--override_frames_meta_file',
        default="",
        help='Override frames meta file path'
    )
    parser.add_argument(
        '--use_odom_pose',
        action='store_true',
        help='Use odometry poses instead of SLAM poses'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print commands without executing'
    )

    return parser.parse_args()


def get_default_binary_dir(package_path=None):
    if package_path is None:
        import pkg_resources
        try:
            package_path = Path(pkg_resources.resource_filename('visual_mapping', ''))
        except Exception:
            package_path = Path(__file__).parent

    # Try runfiles helper for Bazel
    try:
        from .runfiles_helper import get_binary_dir
        return get_binary_dir()
    except (ImportError, Exception):
        return str(package_path / 'bin')


def main():
    args = parse_args()

    binary_dir = args.binary_dir
    if not binary_dir:
        binary_dir = get_default_binary_dir()

    runner = CuVSlamRunner(
        binary_dir=binary_dir,
        config_dir=None,
        config_name=args.config_file,
        dry_run=args.dry_run,
        log_dir=str(Path(args.output_dir) / 'logs')
    )

    try:
        output_meta_file = runner.run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            override_frames_meta_file=args.override_frames_meta_file,
            use_slam_pose=not args.use_odom_pose
        )
        print(f"Successfully ran cuVSLAM. Result metadata: {output_meta_file}")
    except Exception as e:
        print(f"Error running cuVSLAM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
