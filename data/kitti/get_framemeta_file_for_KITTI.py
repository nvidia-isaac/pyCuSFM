# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""
KITTI Dataset to Keyframe Metadata Converter

This script converts KITTI dataset format to keyframe metadata JSON format.
It reads KITTI dataset files (times.txt, calib.txt, image_0/, image_1/) and 
generates a keyframe_meta.json file with camera parameters and frame metadata.

Usage:
    python get_framemeta_file_for_KITTI.py <dataset_dir> [--output-name OUTPUT_NAME]

Examples:
    # Convert KITTI dataset to keyframe_meta.json
    python get_framemeta_file_for_KITTI.py /path/to/kitti/sequence_00
    
    # Convert with custom output filename
    python get_framemeta_file_for_KITTI.py /path/to/kitti/sequence_00 --output-name custom_keyframe.json
    
Required files in dataset directory:
    - times.txt: Timestamp file
    - calib.txt: Camera calibration file
    - image_0/: Left camera images directory
    - image_1/: Right camera images directory

Output:
    - keyframe_meta.json: Generated metadata file in the dataset directory
"""

import os
import json
from datetime import datetime
import numpy as np
from PIL import Image
import subprocess
import argparse


def read_timestamps(times_file):
    """Read timestamps from times.txt file"""
    timestamps = []
    with open(times_file, 'r') as f:
        for line in f:
            # Convert seconds to microseconds
            timestamp = float(line.strip()) * 1e6
            timestamps.append(int(timestamp))
    return timestamps


def read_calib(calib_file):
    """Read calibration data from calib.txt file"""
    calib_data = {}
    with open(calib_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(':', 1)
            calib_data[key] = np.array([float(x)
                                       for x in value.strip().split()])

    # Extract projection matrices for left and right cameras
    P0 = calib_data['P0'].reshape(3, 4)
    P1 = calib_data['P1'].reshape(3, 4)

    return P0, P1


def get_image_files(image_dir):
    """Get sorted list of image files"""
    files = os.listdir(image_dir)
    files = [f for f in files if f.endswith('.png')]
    files.sort()
    return files


def get_image_size(image_dir):
    """Get image dimensions from first image in directory"""
    first_image = os.listdir(image_dir)[0]
    size = Image.open(os.path.join(image_dir, first_image)).size
    return size


def create_camera_params(P0, P1):
    """Create camera parameters configuration"""
    # Extract camera parameters from projection matrices
    baseline_0 = -P0[0, 3] / P0[0, 0]  # left camera baseline
    baseline_1 = -P1[0, 3] / P1[0, 0]  # right camera baseline

    # Extract principal points and focal lengths
    principal_0 = (P0[0, 2], P0[1, 2])
    principal_1 = (P1[0, 2], P1[1, 2])
    focal_0 = (P0[0, 0], P0[1, 1])
    focal_1 = (P1[0, 0], P1[1, 1])

    camera_params = {
        "0": {
            "sensor_meta_data": {
                "sensor_id": 0,
                "sensor_type": "CAMERA",
                "sensor_name": "image_0",
                "frequency": 10,
                "sensor_to_vehicle_transform": {
                    "axis_angle": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "angle_degrees": 0.0
                    },
                    "translation": {
                        "x": baseline_0,
                        "y": 0.0,
                        "z": 0.0
                    }
                }
            },
            "calibration_parameters": {
                "image_width": None,
                "image_height": None,
                "projection_matrix": {
                    "data": P0.flatten().tolist(),
                    "row_count": 3,
                    "column_count": 4
                }
            }
        },
        "1": {
            "sensor_meta_data": {
                "sensor_id": 1,
                "sensor_type": "CAMERA",
                "sensor_name": "image_1",
                "frequency": 10,
                "sensor_to_vehicle_transform": {
                    "axis_angle": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "angle_degrees": 0.0
                    },
                    "translation": {
                        "x": baseline_1,
                        "y": 0.0,
                        "z": 0.0
                    }
                }
            },
            "calibration_parameters": {
                "image_width": None,
                "image_height": None,
                "projection_matrix": {
                    "data": P1.flatten().tolist(),
                    "row_count": 3,
                    "column_count": 4
                }
            }
        }
    }
    return camera_params, baseline_0, baseline_1


def kitti_to_keyframe_metadata(dataset_dir):
    """Generate frames_meta.json file for stereo dataset"""
    # Check required files
    times_file = os.path.join(dataset_dir, 'times.txt')
    calib_file = os.path.join(dataset_dir, 'calib.txt')
    image_0_dir = os.path.join(dataset_dir, 'image_0')
    image_1_dir = os.path.join(dataset_dir, 'image_1')

    # Check each required file/directory individually
    missing_files = []
    if not os.path.exists(times_file):
        missing_files.append(f"times.txt (expected: {times_file})")
    if not os.path.exists(calib_file):
        missing_files.append(f"calib.txt (expected: {calib_file})")
    if not os.path.exists(image_0_dir):
        missing_files.append(f"image_0/ directory (expected: {image_0_dir})")
    if not os.path.exists(image_1_dir):
        missing_files.append(f"image_1/ directory (expected: {image_1_dir})")

    if missing_files:
        error_msg = f"Missing required files or directories:\n" + \
            "\n".join(f"  - {f}" for f in missing_files)
        raise FileNotFoundError(error_msg)

    # Read data
    timestamps = read_timestamps(times_file)
    P0, P1 = read_calib(calib_file)
    images_0 = get_image_files(image_0_dir)
    images_1 = get_image_files(image_1_dir)

    if not (len(images_0) == len(images_1) == len(timestamps)):
        raise ValueError(
            "Number of images does not match number of timestamps")

    # Get image size
    image_size = get_image_size(image_0_dir)
    print(f"Image size: {image_size}")

    # Create camera parameters
    camera_params, baseline_0, baseline_1 = create_camera_params(P0, P1)

    # Set image dimensions
    for cam_id in camera_params:
        camera_params[cam_id]["calibration_parameters"]["image_width"] = image_size[0]
        camera_params[cam_id]["calibration_parameters"]["image_height"] = image_size[1]

    # Calculate baseline distance between cameras
    # Baseline is the distance between camera centers (typically in meters)
    baseline_meters = abs(baseline_1 - baseline_0)

    frame_meta = {
        "keyframes_metadata": [],
        "camera_params_id_to_camera_params": camera_params,
        "camera_params_id_to_session_name": {
            "0": "0",
            "1": "0"
        },
        "stereo_pair": [
            {
                "left_camera_param_id": "0",
                "right_camera_param_id": "1",
                "baseline_meters": baseline_meters
            }
        ]
    }

    for idx, (img0, img1, timestamp) in enumerate(zip(images_0, images_1, timestamps)):
        frame_meta["keyframes_metadata"].extend([
            {
                "id": str(idx * 2),
                "camera_params_id": "0",
                "timestamp_microseconds": str(timestamp),
                "image_name": f"image_0/{img0}",
                "synced_sample_id": str(idx)
            },
            {
                "id": str(idx * 2 + 1),
                "camera_params_id": "1",
                "timestamp_microseconds": str(timestamp),
                "image_name": f"image_1/{img1}",
                "synced_sample_id": str(idx)
            }
        ])

    output_file = os.path.join(dataset_dir, 'keyframe_meta.json')
    with open(output_file, 'w') as f:
        json.dump(frame_meta, f, indent=2)

    print(
        f"Generated keyframe_meta.json with {len(frame_meta['keyframes_metadata'])} records")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert KITTI dataset to keyframe metadata JSON')
    parser.add_argument('dataset_dir', help='Path to KITTI dataset directory')
    parser.add_argument('--output-name', default='keyframe_meta.json',
                        help='Output filename (default: keyframe_meta.json)')

    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist")
        exit(1)

    try:
        output_file = kitti_to_keyframe_metadata(args.dataset_dir)
        print(f"Successfully generated: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
