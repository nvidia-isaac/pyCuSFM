# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility to split repeated pose files from cuvslam into individual run files.

This module:
1. Loads stereo.edx to determine original sequence length
2. Creates timestamp mapping: repeat_timestamps -> original_timestamps
3. Splits poses between runs using this mapping
4. Saves individual run files with original timestamps
"""

import json
import logging
import pathlib
import sys
from typing import Tuple

import numpy as np


class RepeatedPoseSplitter:
    """Clean class to split repeated pose files using timestamp mapping."""

    def __init__(self, poses_dir: pathlib.Path, edx_dir: pathlib.Path, repeat_count: int):
        self.poses_dir = poses_dir
        self.edx_dir = edx_dir
        self.repeat_count = repeat_count
        self.logger = logging.getLogger(__name__)

        # Load frame count once
        self.original_frame_count = self._load_frame_count()

        # Timestamp mapping: repeated_timestamp -> original_timestamp
        self.timestamp_mapping = {}

    def _load_frame_count(self) -> int:
        """Load original frame count from stereo.edx."""
        stereo_edx_path = self.edx_dir / 'stereo.edex'

        if not stereo_edx_path.exists():
            self.logger.error(f"stereo.edex not found: {stereo_edx_path}")
            raise FileNotFoundError(f"stereo.edex not found: {stereo_edx_path}")

        try:
            self.logger.info(f"Loading stereo.edex from: {stereo_edx_path}")
            with open(stereo_edx_path, 'r') as f:
                data = json.load(f)

            metadata = data[0]
            frame_start = metadata.get('frame_start', 0)
            frame_end = metadata.get('frame_end', 0)
            total_frames = frame_end - frame_start + 1

            self.logger.info(
                f"Original sequence: {total_frames} frames (frames {frame_start}-{frame_end})")
            return total_frames

        except Exception as e:
            self.logger.error(f"Failed to parse stereo.edex: {e}")
            raise

    def _load_tum_file(self, file_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load TUM file and return timestamps and poses."""
        try:
            data = np.loadtxt(file_path)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 8:
                raise ValueError(f"TUM file must have at least 8 columns, got {data.shape[1]}")

            timestamps = data[:, 0]
            poses = data[:, 1:]
            return timestamps, poses
        except Exception as e:
            raise ValueError(f"Failed to load TUM file {file_path}: {e}")

    def _save_tum_file(self, file_path: pathlib.Path, timestamps: np.ndarray, poses: np.ndarray):
        """Save timestamps and poses to TUM file."""
        if len(timestamps) != len(poses):
            raise ValueError("Timestamps and poses must have same length")

        data = np.column_stack([timestamps, poses])
        with open(file_path, 'w') as f:
            for row in data:
                f.write(f"{row[0]:.6f} {' '.join(f'{x:.6f}' for x in row[1:])}\n")

    def _create_timestamp_mapping(self, timestamps: np.ndarray) -> tuple:
        """Create simple timestamp mapping.

        Returns: (run_ranges, timestamp_mapping)
        - run_ranges: {run_idx: {'start_ts': float, 'end_ts': float}}
        - timestamp_mapping: {repeated_timestamp: original_timestamp}
        """

        self.logger.info("Creating simple timestamp mapping:")

        # Validate: should have exact poses for clean splitting
        expected_total = self.original_frame_count * self.repeat_count
        if len(timestamps) != expected_total:
            self.logger.error(
                f"Invalid pose count: got {len(timestamps)}, expected {expected_total}")
            self.logger.error(f"  ({self.original_frame_count} frames x {self.repeat_count} runs)")
            raise ValueError("Pose count validation failed")

        # Extract run ranges and create mapping
        run_ranges = {}
        timestamp_mapping = {}

        # Extract first run as original pattern
        original_timestamps = timestamps[:self.original_frame_count]

        for run_idx in range(self.repeat_count):
            start_idx = run_idx * self.original_frame_count
            end_idx = (run_idx + 1) * self.original_frame_count

            run_timestamps = timestamps[start_idx:end_idx]

            # Validate run has correct count
            if len(run_timestamps) != self.original_frame_count:
                self.logger.error(f"Run {run_idx}: got {len(run_timestamps)} poses, "
                                  f"expected {self.original_frame_count}")
                raise ValueError(f"Run {run_idx} validation failed")

            # Store run range
            run_ranges[run_idx] = {
                'start_ts': run_timestamps[0],
                'end_ts': run_timestamps[-1]
            }

            # Create timestamp mapping: repeated -> original
            for i, repeated_ts in enumerate(run_timestamps):
                original_ts = original_timestamps[i]
                timestamp_mapping[repeated_ts] = original_ts

            self.logger.info(f"  Run {run_idx}: {self.original_frame_count} poses")
            self.logger.info(f"    Time range: {run_timestamps[0]:.6f} - {run_timestamps[-1]:.6f}")

        self.logger.info(f"Created mapping for {len(run_ranges)} runs")
        return run_ranges, timestamp_mapping

    def process_regular_poses(self, base_name: str) -> bool:
        """Process regular poses (odom_poses, slam_poses) using index-based splitting."""

        repeated_file = self.poses_dir / f"{base_name}_repeated.tum"
        if not repeated_file.exists():
            self.logger.warning(f"File not found: {repeated_file}")
            return False

        self.logger.info(f"Processing {base_name}_repeated.tum...")

        try:
            # Load repeated poses
            timestamps, poses = self._load_tum_file(repeated_file)
            self.logger.info(f"  Loaded {len(poses)} poses")

            # Validate pose count
            expected_poses = self.original_frame_count * self.repeat_count
            if abs(len(poses) - expected_poses) > expected_poses * 0.1:  # 10% tolerance
                self.logger.error(
                    f"  Pose count mismatch: got {len(poses)}, expected ~{expected_poses}")
                return False

            # Create timestamp mapping
            run_ranges, timestamp_mapping = self._create_timestamp_mapping(timestamps)

            # Split into runs using index boundaries
            first_run_timestamps = timestamps[:self.original_frame_count]

            for run_idx in range(self.repeat_count):
                start_idx = run_idx * self.original_frame_count
                end_idx = min(start_idx + self.original_frame_count, len(poses))

                run_poses = poses[start_idx:end_idx]
                run_timestamps = first_run_timestamps[:len(run_poses)]  # Use original pattern

                # Save run file
                run_file = self.poses_dir / f"{base_name}_{run_idx}.tum"
                self._save_tum_file(run_file, run_timestamps, run_poses)
                self.logger.info(f"  Saved {base_name}_{run_idx}.tum ({len(run_poses)} poses)")

            # Copy last run to main file
            if self.repeat_count > 0:
                last_run_start = (self.repeat_count - 1) * self.original_frame_count
                last_run_end = min(last_run_start + self.original_frame_count, len(poses))

                last_poses = poses[last_run_start:last_run_end]
                last_timestamps = first_run_timestamps[:len(last_poses)]

                main_file = self.poses_dir / f"{base_name}.tum"
                self._save_tum_file(main_file, last_timestamps, last_poses)
                self.logger.info(f"  Copied last run to {base_name}.tum ({len(last_poses)} poses)")

            return True

        except Exception as e:
            self.logger.error(f"Error processing {base_name}: {e}")
            return False

    def process_keyframe_poses(self) -> bool:
        """Process keyframe poses using run ranges and exact timestamp mapping."""

        repeated_file = self.poses_dir / "keyframe_pose_repeated.tum"
        if not repeated_file.exists():
            self.logger.warning("File not found: keyframe_pose_repeated.tum")
            return False

        self.logger.info("Processing keyframe_pose_repeated.tum...")

        try:
            # Load keyframe poses
            kf_timestamps, kf_poses = self._load_tum_file(repeated_file)
            self.logger.info(f"  Loaded {len(kf_poses)} keyframe poses")

            # Get run ranges and timestamp mapping from odom poses
            odom_repeated_file = self.poses_dir / "odom_poses_repeated.tum"
            if not odom_repeated_file.exists():
                self.logger.error("Need odom_poses_repeated.tum for run ranges")
                return False

            odom_timestamps, _ = self._load_tum_file(odom_repeated_file)
            run_ranges, timestamp_mapping = self._create_timestamp_mapping(odom_timestamps)

            # Split keyframes into runs based on timestamp ranges
            keyframe_runs = {}

            for kf_ts, kf_pose in zip(kf_timestamps, kf_poses):
                # Find which run this keyframe belongs to
                found_run = None
                for run_idx, run_range in run_ranges.items():
                    if run_range['start_ts'] <= kf_ts <= run_range['end_ts']:
                        found_run = run_idx
                        break

                if found_run is not None:
                    if found_run not in keyframe_runs:
                        keyframe_runs[found_run] = {'timestamps': [], 'poses': []}

                    # Map to original timestamp using exact mapping
                    if kf_ts in timestamp_mapping:
                        original_ts = timestamp_mapping[kf_ts]
                        keyframe_runs[found_run]['timestamps'].append(original_ts)
                        keyframe_runs[found_run]['poses'].append(kf_pose)
                    else:
                        self.logger.error(f"No exact timestamp match for keyframe {kf_ts:.6f}")
                        return False
                else:
                    self.logger.error(f"Keyframe timestamp {kf_ts:.6f} doesn't belong to any run")
                    return False

            # Save individual run files
            for run_idx in sorted(keyframe_runs.keys()):
                run_data = keyframe_runs[run_idx]
                run_timestamps = np.array(run_data['timestamps'])
                run_poses = np.array(run_data['poses'])

                # Sort by timestamp
                sorted_indices = np.argsort(run_timestamps)
                run_timestamps = run_timestamps[sorted_indices]
                run_poses = run_poses[sorted_indices]

                run_file = self.poses_dir / f"keyframe_pose_{run_idx}.tum"
                self._save_tum_file(run_file, run_timestamps, run_poses)
                self.logger.info(f"  Saved keyframe_pose_{run_idx}.tum ({len(run_poses)} poses)")

            # Use latest run as main keyframe file
            if keyframe_runs:
                latest_run_idx = max(keyframe_runs.keys())
                latest_data = keyframe_runs[latest_run_idx]
                latest_timestamps = np.array(latest_data['timestamps'])
                latest_poses = np.array(latest_data['poses'])

                # Sort by timestamp
                sorted_indices = np.argsort(latest_timestamps)
                latest_timestamps = latest_timestamps[sorted_indices]
                latest_poses = latest_poses[sorted_indices]

                main_file = self.poses_dir / "keyframe_pose.tum"
                self._save_tum_file(main_file, latest_timestamps, latest_poses)
                self.logger.info(
                    f"Saved keyframe_pose.tum:run {latest_run_idx} :: ({len(latest_poses)})")

                return True
            else:
                self.logger.error("No keyframe runs were created")
                return False

        except Exception as e:
            self.logger.error(f"Error processing keyframes: {e}")
            return False

    def process_all_files(self) -> bool:
        """Main entry point - process all repeated files."""

        self.logger.info("=== Split Repeated Poses Started ===")
        self.logger.info(f"Poses dir: {self.poses_dir}")
        self.logger.info(f"EDX dir: {self.edx_dir}")
        self.logger.info(f"Repeat count: {self.repeat_count}")
        self.logger.info(f"Original frame count: {self.original_frame_count}")

        success_count = 0

        # Process regular poses
        for pose_type in ['odom_poses', 'slam_poses']:
            if self.process_regular_poses(pose_type):
                success_count += 1

        # Process keyframe poses
        if self.process_keyframe_poses():
            success_count += 1

        self.logger.info(f"=== Processing Complete: {success_count}/3 files processed ===")
        return success_count > 0
