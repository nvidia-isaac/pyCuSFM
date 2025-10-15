# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""
Constants used throughout the visual mapping package.

This module contains standardized directory and file names to ensure
consistency across the codebase. Always import and use these constants
instead of hardcoding directory or file names.
"""

# Directory name constants
kKEYFRAME_DIR = 'keyframes'
kPOSE_GRAPH_DIR = 'pose_graph'
kMATCHES_DIR = 'matches'
kMATCHES_TASK_DIR = 'tasks'
kMAP_DIR = 'kpmap'
kOPEN_MAP_DIR = 'sparse'
kCUVGL_MAP_DIR = 'cuvgl_map'
kVOC_DIR = 'vocabulary'
kASSOCIATIONS_DIR = 'associations'
kCUVSLAM_OUTPUT_DIR = "cuvslam_output"
kOUTPUT_POSES_DIR = "output_poses"

# File name constants
kFRAME_META_FILE = 'frames_meta.json'
kFRAME_META_FILE_CUVSLAM = 'frames_meta_cuvslam.json'
kSLAM_POSES_FILE = 'slam_poses.tum'
kODOM_POSES_FILE = 'odom_poses.tum'
kMERGED_POSE_FILE = 'merged_pose_file.tum'
kEXTRACT_POSE_LOG_FILE = 'extract_pose_from_map_main_log.txt'
kABSOLUTE_ERROR_FILE = 'absolute_error.json'
