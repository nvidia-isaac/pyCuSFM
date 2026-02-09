#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate frames_meta.json from poses.csv and config.yaml files.

This script reads:
1. A CSV pose file with columns: image_id,qw,qx,qy,qz,tx,ty,tz,camera_params_id,synced_sample_id,timestamp_us,image_name
2. A YAML config file with camera parameters, session mapping, stereo pairs, etc.

And generates a frames_meta.json file compatible with cuSFM/cuVSLAM.

Usage:
    python generate_frame_meta.py \\
        --poses /path/to/poses.csv \\
        --config /path/to/config.yaml \\
        --output /path/to/frames_meta.json

Example:
    python scripts/visual/tools/generate_frame_meta.py \\
        --poses results/cusfm_output_distort_txt/frame_meta_generation/poses.csv \\
        --config results/cusfm_output_distort_txt/frame_meta_generation/config.yaml \\
        --output results/cusfm_output_distort_txt/frame_meta_generation/frames_meta.json
"""

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def parse_pose_file(pose_path: str) -> List[Dict]:
    """
    Parse CSV pose file with columns:
    image_id,qw,qx,qy,qz,tx,ty,tz,camera_params_id,synced_sample_id,timestamp_us,image_name

    Args:
        pose_path: Path to CSV pose file

    Returns:
        List of pose dictionaries
    """
    poses = []

    with open(pose_path, 'r', newline='') as f:
        reader = csv.DictReader(f)

        # Validate required columns
        required_columns = {'image_id', 'qw', 'qx', 'qy', 'qz',
                            'tx', 'ty', 'tz', 'camera_params_id', 'image_name'}
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {pose_path} has no header row")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"CSV file {pose_path} missing required columns: {missing_columns}"
            )

        for row in reader:
            try:
                image_id = row['image_id']
                qw = float(row['qw'])
                qx = float(row['qx'])
                qy = float(row['qy'])
                qz = float(row['qz'])
                tx = float(row['tx'])
                ty = float(row['ty'])
                tz = float(row['tz'])
                image_name = row['image_name']

                # Required column with special handling for '-'
                camera_params_id = row['camera_params_id']
                # Handle '-' as missing camera_params_id (camera without calibration)
                if camera_params_id == '-':
                    camera_params_id = None

                # Optional columns
                synced_sample_id = row.get('synced_sample_id', '')
                timestamp_us = row.get('timestamp_us', '')

                # Handle empty synced_sample_id and timestamp_us
                if synced_sample_id == '':
                    synced_sample_id = None
                if timestamp_us == '':
                    timestamp_us = None

                poses.append({
                    'image_id': image_id,
                    'qw': qw,
                    'qx': qx,
                    'qy': qy,
                    'qz': qz,
                    'tx': tx,
                    'ty': ty,
                    'tz': tz,
                    'camera_params_id': camera_params_id,
                    'synced_sample_id': synced_sample_id,
                    'timestamp_us': timestamp_us,
                    'image_name': image_name
                })

            except (ValueError, KeyError) as e:
                print(f"Warning: Failed to parse row: {row}, error: {e}")
                continue

    return poses


def load_yaml_config(config_path: str) -> Dict:
    """
    Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'initial_pose_type' not in config:
        print("Warning: initial_pose_type not found, using 'EGO_MOTION'")
        config['initial_pose_type'] = 'EGO_MOTION'

    return config


def quaternion_to_axis_angle(qw: float, qx: float, qy: float,
                             qz: float) -> Dict:
    """
    Convert quaternion to axis-angle representation.

    Args:
        qw, qx, qy, qz: Quaternion components (scalar-first)

    Returns:
        Dictionary with 'x', 'y', 'z', 'angle_degrees'
    """
    # scipy expects [x, y, z, w] format
    rotation = R.from_quat([qx, qy, qz, qw])

    # Get rotation vector (axis * angle_radians)
    rotvec = rotation.as_rotvec()
    angle_rad = np.linalg.norm(rotvec)

    if angle_rad > 1e-10:
        axis = rotvec / angle_rad
        angle_degrees = np.degrees(angle_rad)
    else:
        # No rotation - use default axis
        axis = np.array([0.0, 0.0, 1.0])
        angle_degrees = 0.0

    return {
        'x': float(axis[0]),
        'y': float(axis[1]),
        'z': float(axis[2]),
        'angle_degrees': float(angle_degrees)
    }


def extract_timestamp_from_image_name(image_name: str) -> Optional[str]:
    """
    Extract timestamp from image name.

    Expected format: camera_name/timestamp.extension
    Example: back_stereo_camera_right/1707938736636230528.jpeg

    Args:
        image_name: Image filename with path

    Returns:
        Timestamp in microseconds as string, or None if extraction fails
    """
    filename = os.path.basename(image_name)
    stem = os.path.splitext(filename)[0]

    try:
        timestamp_ns = int(stem)
        timestamp_us = timestamp_ns // 1000
        return str(timestamp_us)
    except ValueError:
        numbers = re.findall(r'\d+', stem)
        if numbers:
            longest = max(numbers, key=len)
            timestamp_ns = int(longest)
            timestamp_us = timestamp_ns // 1000
            return str(timestamp_us)

    return None


def parse_session_camera_mapping(config: Dict) -> Dict[str, str]:
    """
    Parse session to camera params ID mapping.

    Format in config:
        session_camera_params_id_mapping:
          session_0: 0, 1, 2, 3, 4, 5, 6, 7
          session_1: 8, 9, 10, 11

    Returns:
        Dictionary mapping camera_params_id to session_name
    """
    camera_params_id_to_session_name = {}
    mapping = config.get('session_camera_params_id_mapping', {})

    if not mapping:
        print("Warning: session_camera_params_id_mapping not found")
        return camera_params_id_to_session_name

    # Track all camera_params_ids to check for duplicates
    all_camera_ids = set()

    for session_name, camera_ids_str in mapping.items():
        # Extract session number from session name (e.g., "session_0" -> "0")
        session_num_match = re.search(r'\d+', str(session_name))
        session_num = session_num_match.group() if session_num_match else str(session_name)

        # Parse camera IDs from the comma-separated string
        if isinstance(camera_ids_str, str):
            camera_ids = [cid.strip() for cid in camera_ids_str.split(',')]
        elif isinstance(camera_ids_str, list):
            camera_ids = [str(cid).strip() for cid in camera_ids_str]
        else:
            camera_ids = [str(camera_ids_str).strip()]

        for camera_id in camera_ids:
            if not camera_id:
                continue

            # Check for duplicates across sessions
            if camera_id in all_camera_ids:
                raise ValueError(
                    f"Camera params ID '{camera_id}' appears in multiple sessions. "
                    "Each camera_params_id must be unique across all sessions."
                )

            all_camera_ids.add(camera_id)
            camera_params_id_to_session_name[camera_id] = session_num

    return camera_params_id_to_session_name


def validate_intrinsic(intrinsic: List[float]) -> bool:
    """
    Validate that intrinsic is a 3x3 matrix (9 values).

    Args:
        intrinsic: List of intrinsic matrix values

    Returns:
        True if valid, raises ValueError otherwise
    """
    if len(intrinsic) != 9:
        raise ValueError(
            f"Intrinsic matrix must have exactly 9 values (3x3), got {len(intrinsic)}"
        )
    return True


def determine_projection_model_type(distortion_coefficients: Optional[List[float]],
                                    camera_id: str = "") -> str:
    """
    Determine camera projection model type based on distortion coefficients.

    - If distortion_coefficients is None, empty, or all zeros -> PINHOLE
    - If distortion_coefficients has exactly 4 values -> OPENCV_FISHEYE
    - If distortion_coefficients has exactly 8 values -> DISTORTED_PINHOLE
    - Other cases -> Error and exit

    Args:
        distortion_coefficients: List of distortion coefficients
        camera_id: Camera ID for error messages

    Returns:
        "PINHOLE", "OPENCV_FISHEYE", or "DISTORTED_PINHOLE"
    """
    if not distortion_coefficients:
        return "PINHOLE"

    # Check if all coefficients are zero
    all_zeros = all(abs(coeff) <= 1e-15 for coeff in distortion_coefficients)
    if all_zeros:
        return "PINHOLE"

    num_coeffs = len(distortion_coefficients)

    if num_coeffs == 4:
        return "OPENCV_FISHEYE"
    elif num_coeffs == 8:
        return "DISTORTED_PINHOLE"
    else:
        print(f"Error: Camera '{camera_id}' has {num_coeffs} distortion coefficients. "
              f"Expected 4 (OPENCV_FISHEYE) or 8 (DISTORTED_PINHOLE).")
        exit(1)


def build_camera_params(config: Dict,
                        camera_params_id_to_session_name: Dict[str, str]) -> Dict[str, Dict]:
    """
    Build camera_params_id_to_camera_params from config.

    Args:
        config: Configuration dictionary
        camera_params_id_to_session_name: Mapping of camera ID to session name

    Returns:
        Dictionary of camera parameters
    """
    camera_params_id_to_camera_params = {}

    # Get camera params from config
    camera_params_config = config.get('camera_params', {})

    # Get common parameters
    image_width = config.get('image_width', 1920)
    image_height = config.get('image_height', 1200)
    frequency = config.get('frequency', 30)

    for camera_id_str, cam_config in camera_params_config.items():
        camera_id = str(camera_id_str)

        # Validate that this camera_id has a session mapping
        if camera_id not in camera_params_id_to_session_name:
            print(
                f"Warning: Camera params ID '{camera_id}' not found in session mapping")

        # Get sensor name
        sensor_name = cam_config.get('sensor_name', f'camera_{camera_id}')

        # Get and validate extrinsic (QW, QX, QY, QZ, TX, TY, TZ)
        extrinsic = cam_config.get(
            'extrinsic', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if len(extrinsic) != 7:
            raise ValueError(
                f"Camera {camera_id}: extrinsic must have 7 values "
                f"[QW, QX, QY, QZ, TX, TY, TZ], got {len(extrinsic)}"
            )

        qw, qx, qy, qz = extrinsic[0], extrinsic[1], extrinsic[2], extrinsic[3]
        tx, ty, tz = extrinsic[4], extrinsic[5], extrinsic[6]

        # Convert quaternion to axis-angle for sensor_to_vehicle_transform
        axis_angle = quaternion_to_axis_angle(qw, qx, qy, qz)

        # Get and validate intrinsic (3x3 = 9 values)
        intrinsic = cam_config.get(
            'intrinsic', [1000.0, 0, 960, 0, 1000.0, 600, 0, 0, 1])
        validate_intrinsic(intrinsic)

        # Get distortion coefficients
        distortion_coefficients = cam_config.get('distortion_coefficients', [])

        # Determine projection model type
        projection_model_type = determine_projection_model_type(
            distortion_coefficients, camera_id)

        # Use global frequency (can be overridden per camera)
        cam_frequency = cam_config.get('frequency', frequency)

        # Build sensor_meta_data
        sensor_meta_data = {
            "sensor_type": "CAMERA",
            "sensor_name": sensor_name,
            "frequency": cam_frequency,
            "sensor_to_vehicle_transform": {
                "axis_angle": axis_angle,
                "translation": {
                    "x": tx,
                    "y": ty,
                    "z": tz
                }
            }
        }

        # Add sensor_id if camera_id is numeric
        try:
            sensor_id = int(camera_id)
            sensor_meta_data["sensor_id"] = sensor_id
        except ValueError:
            pass

        # Build calibration parameters based on projection model type
        # - PINHOLE: only needs projection_matrix
        # - DISTORTED_PINHOLE: needs camera_matrix and distortion_coefficients
        # Note: All cameras must use the same image dimensions (global config)
        calibration_parameters = {
            "image_width": image_width,
            "image_height": image_height,
        }

        if projection_model_type == "PINHOLE":
            # PINHOLE model: use projection_matrix (3x4)
            fx, _, cx = intrinsic[0], intrinsic[1], intrinsic[2]
            _, fy, cy = intrinsic[3], intrinsic[4], intrinsic[5]
            projection_matrix = [
                fx, 0.0, cx, 0.0,
                0.0, fy, cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            calibration_parameters["projection_matrix"] = {
                "data": projection_matrix,
                "row_count": 3,
                "column_count": 4
            }
        else:
            # DISTORTED_PINHOLE or OPENCV_FISHEYE: use camera_matrix and distortion_coefficients
            calibration_parameters["camera_matrix"] = {
                "data": intrinsic,
                "row_count": 3,
                "column_count": 3
            }
            calibration_parameters["distortion_coefficients"] = {
                "data": distortion_coefficients if distortion_coefficients else [],
                "row_count": 1,
                "column_count": len(distortion_coefficients) if distortion_coefficients else 0
            }

        # Build final camera params structure
        camera_params_id_to_camera_params[camera_id] = {
            "sensor_meta_data": sensor_meta_data,
            "calibration_parameters": calibration_parameters,
            "camera_projection_model_type": projection_model_type
        }

    return camera_params_id_to_camera_params


def build_stereo_pairs(config: Dict) -> List[Dict]:
    """
    Build stereo pair configuration from config.

    Format in config:
        stereo_pair:
          - [0, 1, 0.15]  # left_id, right_id, baseline_meters
          - [2, 3, 0.15]

    Returns:
        List of stereo pair dictionaries
    """
    stereo_pairs = []
    stereo_config = config.get('stereo_pair', [])

    for pair in stereo_config:
        if isinstance(pair, list) and len(pair) >= 3:
            left_id = str(pair[0])
            right_id = str(pair[1])
            baseline = float(pair[2])

            stereo_pair_entry = {
                "left_camera_param_id": left_id,
                "right_camera_param_id": right_id,
                "baseline_meters": baseline
            }
            stereo_pairs.append(stereo_pair_entry)
        elif isinstance(pair, dict):
            # Already in dict format
            stereo_pairs.append(pair)

    return stereo_pairs


def find_stereo_camera_dirs(base_dir: str) -> Tuple[str, str]:
    """
    Find left/right camera directories by keyword matching.

    Args:
        base_dir: Base directory containing camera folders

    Returns:
        Tuple of (left_camera_dir, right_camera_dir)
    """
    left_camera_dir = None
    right_camera_dir = None

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            item_lower = item.lower()
            if "left" in item_lower:
                left_camera_dir = item_path
            elif "right" in item_lower:
                right_camera_dir = item_path

    if not left_camera_dir:
        raise ValueError(f"Cannot find 'left' camera directory in {base_dir}")
    if not right_camera_dir:
        raise ValueError(f"Cannot find 'right' camera directory in {base_dir}")

    return left_camera_dir, right_camera_dir


def auto_detect_synced_sample_ids(poses: List[Dict], threshold_ms: float = 0.05) -> Dict[int, str]:
    """
    Automatically detect synced_sample_id for all cameras based on timestamps.

    Groups frames from different cameras captured within a time threshold.

    Args:
        poses: List of pose dictionaries with timestamp_us and camera_params_id
        threshold_ms: Time threshold in milliseconds for considering frames synchronized

    Returns:
        Dictionary mapping pose index to synced_sample_id string
    """
    # Convert threshold to microseconds
    threshold_us = int(threshold_ms * 1000)

    # Build list of (timestamp_us, camera_id, pose_index)
    frame_info = []
    for idx, pose in enumerate(poses):
        timestamp_us = int(pose['timestamp_us'])
        camera_id = str(pose.get('camera_params_id', ''))
        frame_info.append((timestamp_us, camera_id, idx))

    # Sort by timestamp
    frame_info.sort(key=lambda x: x[0])

    # Group frames within threshold
    synced_map = {}
    used_indices = set()
    synced_id_counter = 1

    i = 0
    while i < len(frame_info):
        if frame_info[i][2] in used_indices:
            i += 1
            continue

        base_timestamp, base_camera, base_idx = frame_info[i]

        # Find all frames within threshold with different cameras
        sync_group = [(base_timestamp, base_camera, base_idx)]
        cameras_in_group = {base_camera}

        # Look ahead for frames within threshold
        j = i + 1
        while j < len(frame_info):
            curr_timestamp, curr_camera, curr_idx = frame_info[j]

            # Stop if beyond threshold
            if curr_timestamp - base_timestamp > threshold_us:
                break

            # Add if different camera and not already used
            if curr_camera not in cameras_in_group and curr_idx not in used_indices:
                sync_group.append((curr_timestamp, curr_camera, curr_idx))
                cameras_in_group.add(curr_camera)

            j += 1

        # Assign synced_sample_id if we have multiple cameras
        if len(sync_group) > 1:
            synced_sample_id = str(synced_id_counter)
            for _, _, idx in sync_group:
                synced_map[idx] = synced_sample_id
                used_indices.add(idx)
            synced_id_counter += 1
        else:
            # Single frame, mark as unsynced
            synced_map[base_idx] = "0"
            used_indices.add(base_idx)

        i += 1

    return synced_map


def generate_frames_meta(poses_path: str,
                         config_path: str,
                         output_path: str) -> None:
    """
    Generate frames_meta.json from poses.txt and config.yaml.

    Args:
        poses_path: Path to pose file
        config_path: Path to YAML config file
        output_path: Output path for frames_meta.json
    """
    print(f"Reading poses from {poses_path} (CSV format)...")
    poses = parse_pose_file(poses_path)
    print(f"  Found {len(poses)} poses")

    print(f"Reading config from {config_path}...")
    config = load_yaml_config(config_path)

    # 1. Parse initial_pose_type
    initial_pose_type = config.get('initial_pose_type', 'EGO_MOTION')
    print(f"  Initial pose type: {initial_pose_type}")

    # 2. Parse session to camera mapping
    print("Parsing session camera mapping...")
    camera_params_id_to_session_name = parse_session_camera_mapping(config)
    print(
        f"  Found {len(camera_params_id_to_session_name)} camera-to-session mappings")

    # 3. Build camera parameters
    print("Building camera parameters...")
    camera_params_id_to_camera_params = build_camera_params(
        config, camera_params_id_to_session_name
    )
    print(f"  Built {len(camera_params_id_to_camera_params)} camera configs")

    # Validate that all camera IDs in session mapping have camera params
    for cam_id in camera_params_id_to_session_name:
        if cam_id not in camera_params_id_to_camera_params:
            print(f"Warning: Camera ID '{cam_id}' in session mapping "
                  f"but not in camera_params")

    # 4. Build stereo pairs
    print("Building stereo pairs...")
    stereo_pairs = build_stereo_pairs(config)
    print(f"  Found {len(stereo_pairs)} stereo pairs")

    # 5. Create keyframes metadata from poses
    print("Creating keyframes metadata...")

    # First pass: extract/validate timestamps and check if we need auto-sync
    need_auto_sync = False
    for i, pose in enumerate(poses):
        # Extract or validate timestamp
        if pose.get('timestamp_us'):
            timestamp_us = pose['timestamp_us']
        else:
            timestamp_us = extract_timestamp_from_image_name(
                pose['image_name'])
            if timestamp_us is None:
                print(
                    f"Error: Cannot extract timestamp from image name: {pose['image_name']}")
                print(
                    f"  When 'timestamp_us' is missing, image names must contain timestamp")
                print(f"  Expected format: camera_name/timestamp_nanoseconds.extension")
                print(f"  Example: front_camera_left/1707938736136246532.jpeg")
                exit(1)
            pose['timestamp_us'] = timestamp_us  # Store back for auto-sync

        # Check if synced_sample_id is missing or invalid
        synced_sample_id = pose.get('synced_sample_id')
        if synced_sample_id is None or synced_sample_id == "" or synced_sample_id == "0":
            need_auto_sync = True

    # Perform auto-sync detection if needed
    synced_map = {}
    if need_auto_sync:
        print("  Auto-detecting synced_sample_ids based on timestamps...")
        synced_map = auto_detect_synced_sample_ids(poses, threshold_ms=0.05)
        num_synced_groups = len(
            set(v for v in synced_map.values() if v != "0"))
        print(
            f"  Detected {num_synced_groups} synchronized groups across all cameras")

    # Second pass: build keyframes with sync IDs
    keyframes_metadata = []
    for i, pose in enumerate(poses):
        # Convert quaternion to axis-angle for camera_to_world
        axis_angle = quaternion_to_axis_angle(
            pose['qw'], pose['qx'], pose['qy'], pose['qz']
        )

        # Use the timestamp we extracted/validated in first pass
        timestamp_us = pose['timestamp_us']

        # Determine synced_sample_id
        if need_auto_sync:
            synced_sample_id = synced_map.get(i, "0")
        else:
            synced_sample_id = pose.get('synced_sample_id', "0")
            if synced_sample_id is None:
                synced_sample_id = "0"

        # Build keyframe entry - preserve exact field order
        keyframe = {'id': pose['image_id']}

        # Add camera_params_id only if it was present in the original
        if pose['camera_params_id'] is not None:
            keyframe['camera_params_id'] = pose['camera_params_id']

        keyframe['timestamp_microseconds'] = timestamp_us
        keyframe['image_name'] = pose['image_name']
        keyframe['camera_to_world'] = {
            'axis_angle': axis_angle,
            'translation': {
                'x': pose['tx'],
                'y': pose['ty'],
                'z': pose['tz']
            }
        }
        keyframe['synced_sample_id'] = synced_sample_id

        keyframes_metadata.append(keyframe)

    # Build output structure
    output_data = {
        'keyframes_metadata': keyframes_metadata,
        'initial_pose_type': initial_pose_type,
        'camera_params_id_to_session_name': camera_params_id_to_session_name,
        'camera_params_id_to_camera_params': camera_params_id_to_camera_params,
        'stereo_pair': stereo_pairs
    }

    # Write output
    print(f"Writing output to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Successfully generated frames_meta.json")
    print(f"  Total keyframes: {len(keyframes_metadata)}")
    print(f"  Total cameras: {len(camera_params_id_to_camera_params)}")
    print(f"  Stereo pairs: {len(stereo_pairs)}")
    unique_synced_ids = len(set(kf.get('synced_sample_id', '0')
                            for kf in keyframes_metadata))
    print(f"  Unique synced sample IDs: {unique_synced_ids}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate frames_meta.json from poses.csv or image folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODE 1: Generate from poses.csv (with real camera poses)
=========================================================
    python generate_frame_meta.py \\
        --poses /path/to/poses.csv \\
        --config /path/to/config.yaml \\
        --output /path/to/frames_meta.json

MODE 2: Generate from image folders (with identity poses for mapping)
======================================================================
    python generate_frame_meta.py \\
        --images /path/to/image_folder \\
        --config /path/to/config.yaml \\
        --output /path/to/frames_meta.json \\
        [--use-pseudo-timestamps]

Notes:
  - Mode 1: For datasets with known camera poses (from COLMAP, SLAM, etc.)
  - Mode 2: For raw stereo image pairs, generates identity poses for cuSFM mapping
  - Mode 2 supports ONLY single stereo pairs (left/right folders)
        """)

    # Mutually exclusive input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--poses',
        type=str,
        help='Path to CSV pose file (Mode 1)')
    input_group.add_argument(
        '--images',
        type=str,
        help='Path to image folder containing left/right subdirs (Mode 2)')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file')

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for frames_meta.json')

    # Mode 2 specific options
    parser.add_argument(
        '--use-pseudo-timestamps',
        action='store_true',
        help='Use sequential pseudo timestamps at 30Hz (Mode 2 only)')

    args = parser.parse_args()

    # Validate config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Route to appropriate mode
    if args.poses:
        # Mode 1: Poses mode
        if not os.path.exists(args.poses):
            print(f"Error: Pose file not found: {args.poses}")
            return 1

        generate_frames_meta(
            poses_path=args.poses,
            config_path=args.config,
            output_path=args.output
        )

    elif args.images:
        # Mode 2: Images mode
        if not os.path.isdir(args.images):
            print(f"Error: Image directory not found: {args.images}")
            return 1

        generate_frames_meta_from_images(
            images_dir=args.images,
            config_path=args.config,
            output_path=args.output,
            use_pseudo_timestamps=args.use_pseudo_timestamps
        )

    return 0


def generate_frames_meta_from_images(images_dir: str,
                                     config_path: str,
                                     output_path: str,
                                     use_pseudo_timestamps: bool = False) -> None:
    """
    Generate frames_meta.json from image folders for stereo pair.

    This mode generates identity poses (all zeros) suitable for initial mapping with cuSFM.

    Args:
        images_dir: Directory containing left/right camera folders
        config_path: Path to YAML config file
        output_path: Output path for frames_meta.json
        use_pseudo_timestamps: Use sequential pseudo timestamps (30Hz) instead of filename-based
    """
    print(f"\n{'='*70}")
    print(f"  IMAGES MODE: Generating from stereo image folders")
    print(f"{'='*70}\n")

    # Find camera directories
    print(f"Finding stereo camera directories in {images_dir}...")
    left_camera_dir, right_camera_dir = find_stereo_camera_dirs(images_dir)
    print(f"  Left camera:  {os.path.basename(left_camera_dir)}")
    print(f"  Right camera: {os.path.basename(right_camera_dir)}")

    # Load config
    print(f"\nReading config from {config_path}...")
    config = load_yaml_config(config_path)

    # Apply defaults for images mode
    config.setdefault('initial_pose_type', 'EGO_MOTION')
    config.setdefault('image_width', 1920)
    config.setdefault('image_height', 1200)
    config.setdefault('frequency', 30)

    # Auto-generate session mapping if missing
    if 'session_camera_params_id_mapping' not in config:
        config['session_camera_params_id_mapping'] = {'session_0': '0, 1'}
        print("  Auto-generated session mapping: {session_0: '0, 1'}")

    # Apply per-camera defaults
    for cam_id in ['0', '1']:
        if cam_id not in config.get('camera_params', {}):
            config.setdefault('camera_params', {})[cam_id] = {}

        cam_config = config['camera_params'][cam_id]
        cam_config.setdefault(
            'sensor_name', f'stereo_camera_{"left" if cam_id == "0" else "right"}')
        cam_config.setdefault('extrinsic', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cam_config.setdefault('distortion_coefficients', [])

        # Intrinsic is required, provide a default if missing
        if 'intrinsic' not in cam_config:
            print(f"Warning: No intrinsic for camera {cam_id}, using default")
            cam_config['intrinsic'] = [1000.0, 0, 960, 0, 1000.0, 600, 0, 0, 1]

    # Auto-generate stereo_pair if missing
    if 'stereo_pair' not in config:
        # Extract baseline from camera 1's extrinsic tx
        baseline = abs(config['camera_params']['1']['extrinsic'][4])
        if baseline == 0:
            baseline = 0.12  # Default baseline
        config['stereo_pair'] = [[0, 1, baseline]]
        print(f"  Auto-generated stereo pair with baseline: {baseline}m")

    # Parse config
    initial_pose_type = config.get('initial_pose_type', 'EGO_MOTION')
    print(f"  Initial pose type: {initial_pose_type}")

    camera_params_id_to_session_name = parse_session_camera_mapping(config)
    camera_params_id_to_camera_params = build_camera_params(
        config, camera_params_id_to_session_name)
    stereo_pairs = build_stereo_pairs(config)

    # Scan images
    print(f"\nScanning images...")
    left_images = sorted([f for f in os.listdir(left_camera_dir)
                         if f.endswith(('.jpeg', '.jpg', '.png'))])
    right_images = sorted([f for f in os.listdir(right_camera_dir)
                          if f.endswith(('.jpeg', '.jpg', '.png'))])

    print(f"  Found {len(left_images)} images in left camera")
    print(f"  Found {len(right_images)} images in right camera")

    # Generate keyframes with identity poses
    print(f"\nGenerating keyframes with identity poses...")
    keyframes_metadata = []
    poses_for_sync = []  # For auto-sync detection
    entry_id = 0

    if use_pseudo_timestamps:
        # Pseudo timestamp mode: sequential 30Hz timestamps
        print("  Using pseudo timestamps (30Hz)")
        frame_interval_us = int(1_000_000 / 30)
        start_timestamp_us = 0

        max_pairs = max(len(left_images), len(right_images))

        for i in range(max_pairs):
            pseudo_timestamp_us = start_timestamp_us + (i * frame_interval_us)

            # Left camera
            if i < len(left_images):
                pose = {
                    'image_id': str(entry_id),
                    'camera_params_id': '0',
                    'timestamp_us': str(pseudo_timestamp_us),
                    'image_name': f"{os.path.basename(left_camera_dir)}/{left_images[i]}"
                }
                poses_for_sync.append(pose)
                entry_id += 1

            # Right camera
            if i < len(right_images):
                pose = {
                    'image_id': str(entry_id),
                    'camera_params_id': '1',
                    'timestamp_us': str(pseudo_timestamp_us),
                    'image_name': f"{os.path.basename(right_camera_dir)}/{right_images[i]}"
                }
                poses_for_sync.append(pose)
                entry_id += 1
    else:
        # Timestamp from filename mode
        print("  Extracting timestamps from filenames")

        # Collect all images with timestamps
        for filename in left_images:
            try:
                timestamp_ns = int(os.path.splitext(filename)[0])
                timestamp_us = timestamp_ns // 1000
                pose = {
                    'image_id': str(entry_id),
                    'camera_params_id': '0',
                    'timestamp_us': str(timestamp_us),
                    'image_name': f"{os.path.basename(left_camera_dir)}/{filename}"
                }
                poses_for_sync.append(pose)
                entry_id += 1
            except ValueError:
                print(
                    f"Warning: Cannot parse timestamp from {filename}, skipping")

        for filename in right_images:
            try:
                timestamp_ns = int(os.path.splitext(filename)[0])
                timestamp_us = timestamp_ns // 1000
                pose = {
                    'image_id': str(entry_id),
                    'camera_params_id': '1',
                    'timestamp_us': str(timestamp_us),
                    'image_name': f"{os.path.basename(right_camera_dir)}/{filename}"
                }
                poses_for_sync.append(pose)
                entry_id += 1
            except ValueError:
                print(
                    f"Warning: Cannot parse timestamp from {filename}, skipping")

        # Sort by timestamp
        poses_for_sync.sort(key=lambda x: int(x['timestamp_us']))

    # Auto-detect synced_sample_ids
    print("  Auto-detecting synced_sample_ids...")
    synced_map = auto_detect_synced_sample_ids(
        poses_for_sync, threshold_ms=0.05)
    num_synced_groups = len(set(v for v in synced_map.values() if v != "0"))
    print(f"  Detected {num_synced_groups} synchronized pairs")

    # Build keyframes with identity transforms
    for i, pose_info in enumerate(poses_for_sync):
        keyframe = {
            'id': pose_info['image_id'],
            'camera_params_id': pose_info['camera_params_id'],
            'timestamp_microseconds': pose_info['timestamp_us'],
            'image_name': pose_info['image_name'],
            'camera_to_world': {
                'axis_angle': {'x': 0.0, 'y': 0.0, 'z': 1.0, 'angle_degrees': 0.0},
                'translation': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            },
            'synced_sample_id': synced_map.get(i, "0")
        }
        keyframes_metadata.append(keyframe)

    # Build output structure
    output_data = {
        'keyframes_metadata': keyframes_metadata,
        'initial_pose_type': initial_pose_type,
        'camera_params_id_to_session_name': camera_params_id_to_session_name,
        'camera_params_id_to_camera_params': camera_params_id_to_camera_params,
        'stereo_pair': stereo_pairs
    }

    # Write output
    print(f"\nWriting output to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ✓ Successfully generated frames_meta.json (IMAGES MODE)")
    print(f"{'='*70}")
    print(f"  Total keyframes: {len(keyframes_metadata)}")
    print(f"  Total cameras: {len(camera_params_id_to_camera_params)}")
    print(f"  Stereo pairs: {len(stereo_pairs)}")
    print(f"  Synchronized pairs: {num_synced_groups}")
    print(
        f"  Mode: {'Pseudo-timestamps (30Hz)' if use_pseudo_timestamps else 'Timestamp from filename'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    exit(main())
