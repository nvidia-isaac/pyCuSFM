# PyCuSFM Tutorial

This tutorial guides you through using PyCuSFM for Structure from Motion (SfM) reconstruction.

## Table of Contents

- [Raw Data Requirements](#raw-data-requirements)
- [Command Line Interface](#command-line-interface)
- [Quick Start Guide](#quick-start-guide)
- [Multi-track Input or Localization Mode](#multi-track-input-or-localization-mode)
- [KITTI Dataset Example](#kitti-dataset-example)
- [Bundle Adjustment Runner](#bundle-adjustment-runner)

## Raw Data Requirements

Your input data folder should contain:

- **Image files**: Camera images in supported formats (JPEG, PNG, etc.)
- **`frames_meta.json`**: Metadata file following the `KeyframesMetadataCollection` protobuf format

For a cleaner input format, you can generate `frames_meta.json` from `poses.csv` or stereo image folders using [`pycusfm/generate_frame_meta.py`](README_auxiliary_tools.md#tool-5-keyframe-metadata-generator).

### frames_meta.json Structure

The `frames_meta.json` file is the core metadata file that follows the `KeyframesMetadataCollection` protobuf definition. It contains:

<details>
<summary><strong>Required Top-Level Fields</strong></summary>

- **`keyframes_metadata`**: Array of keyframe metadata objects
- **`initial_pose_type`**: Pose interpretation type (EGO_MOTION, ALIGNMENT, GPS_IMU, MAP_POSE, SESSION_EGO_MOTION)
- **`camera_params_id_to_camera_params`**: Map of camera parameter IDs to camera calibration data

</details>

<details>
<summary><strong>Optional Top-Level Fields</strong></summary>

- **`reference_latlngalt`**: GPS reference point for absolute positioning
- **`stereo_pair`**: Array defining stereo camera pairs with baseline distances
- **`detector_type`**: Feature detector type (e.g., SIFT, ORB, ALIKED)
- **`descriptor_type`**: Feature descriptor type
- **`vehicle_trajectory_files`**: Relative paths to trajectory files
- **`track_id_to_track_name`**: Mapping of track IDs to track names
- **`camera_params_id_to_session_name`**: Map of camera parameter IDs to session names (optional for single-track inputs)

</details>

<details>
<summary><strong>Keyframe Metadata Structure</strong></summary>

Each entry in `keyframes_metadata` contains:

- **`id`**: Unique keyframe identifier (string)
- **`camera_params_id`**: Reference to camera parameters (string)
- **`timestamp_microseconds`**: Timestamp in microseconds (string)
- **`image_name`**: Relative path to image file (string)
- **`camera_to_world`**: 6DOF camera pose with axis-angle rotation and translation
  - **`axis_angle`**: Rotation as axis-angle representation with angle in degrees
    - `x`, `y`, `z`: Rotation axis components
    - `angle_degrees`: Rotation angle in degrees
  - **`translation`**: 3D position in world coordinates
    - `x`, `y`, `z`: Translation components in meters
- **`synced_sample_id`**: Synchronization ID for multi-camera setups (string)

</details>

<details>
<summary><strong>Camera Parameters Structure</strong></summary>

Each camera in `camera_params_id_to_camera_params` includes:

- **`sensor_meta_data`**:
  - `sensor_id`: Unique sensor identifier
  - `sensor_type`: Type (typically "CAMERA")
  - `sensor_name`: Human-readable camera name
  - `frequency`: Frame rate in Hz
  - `sensor_to_vehicle_transform`: Transform from camera to vehicle coordinate frame
- **`calibration_parameters`**:
  - `image_width`, `image_height`: Image dimensions
  - `camera_matrix`: 3x3 intrinsic camera matrix
  - `distortion_coefficients`: Distortion parameters
  - `rectification_matrix`: Rectification matrix (for stereo cameras)
  - `projection_matrix`: Projection matrix

</details>

<details>
<summary><strong>Stereo Pair Configuration</strong></summary>

For stereo camera setups, `stereo_pair` defines:

- **`left_camera_param_id`**: ID of left camera
- **`right_camera_param_id`**: ID of right camera
- **`baseline_meters`**: Baseline distance between cameras in meters

</details>


<details>
<summary><strong>Initial Pose Types</strong></summary>


The `initial_pose_type` field determines how PyCuSFM interprets the camera poses in `camera_to_world`:

- **`EGO_MOTION`**: Relative poses within a single track/sequence. Camera poses are relative to the first frame of the track.
- **`SESSION_EGO_MOTION`**: Relative poses within a session that may contain multiple tracks.
- **`ALIGNMENT`**: High-weight global pose constraints used for alignment between different sequences.
- **`GPS_IMU`**: Absolute position constraints from GPS/IMU sensors in global coordinates.
- **`MAP_POSE`**: Constant poses that don't change during optimization (e.g., previously mapped locations).

The most common use case is `EGO_MOTION` for sequential visual odometry data.
</details>

<details>
<summary><strong>Example frames_meta.json</strong></summary>

Here's a minimal example showing the required structure:

```json
{
  "keyframes_metadata": [
    {
      "id": "2356",
      "camera_params_id": "6",
      "timestamp_microseconds": "1707938736136246",
      "image_name": "right_stereo_camera_left/1707938736136246532.jpeg",
      "camera_to_world": {
        "axis_angle": {
          "x": 0.008544724537343526,
          "y": -0.7085686596387489,
          "z": 0.7055901375872029,
          "angle_degrees": 177.9004863048234
        },
        "translation": {
          "x": 4.607160850493191,
          "y": -0.0672254120438331,
          "z": 0.24059434306041566
        }
      },
      "synced_sample_id": "318"
    }
  ],
  "initial_pose_type": "EGO_MOTION",
  "camera_params_id_to_session_name": {
    "6": "0"
  },
  "camera_params_id_to_camera_params": {
    "6": {
      "sensor_meta_data": {
        "sensor_id": 6,
        "sensor_type": "CAMERA",
        "sensor_name": "right_stereo_camera_left",
        "frequency": 30,
        "sensor_to_vehicle_transform": {
          "axis_angle": { "x": 0, "y": 0, "z": 0, "angle_degrees": 0 },
          "translation": { "x": 0.093139, "y": -0.075002, "z": 0.34439 }
        }
      },
      "calibration_parameters": {
        "image_width": 1920,
        "image_height": 1200,
        "camera_matrix": {
          "data": [961.123, 0, 952.127, 0, 958.858, 591.744, 0, 0, 1]
        },
        "distortion_coefficients": {
          "data": [-0.173, 0.027, 0, 0, 0]
        }
      }
    }
  },
  "stereo_pair": [
    {
      "left_camera_param_id": "6",
      "right_camera_param_id": "7",
      "baseline_meters": 0.1499989761475276
    }
  ]
}
```

</details>

### Important Notes

1. **Coordinate Systems**:
   - `camera_to_world` transforms points from camera coordinate frame to world coordinate frame
   - `sensor_to_vehicle_transform` transforms from camera to vehicle coordinate frame
   - Rotations use axis-angle representation with angle in degrees

2. **Image Paths**: All image paths in `image_name` are relative to the directory containing `frames_meta.json`

3. **Timestamp Format**: Timestamps must be in microseconds as strings (not integers)

4. **Optional Rolling Shutter Support**:
   - Use `start_camera_to_world` for rolling shutter cameras (pose of first row)
   - `camera_to_world` becomes the pose of the last row
   - Include `camera_params_id_to_distorted_row_indices` for row distortion data

5. **Multi-Camera Synchronization**: Use `synced_sample_id` to group frames captured simultaneously across different cameras

### Example Data Structure

The example data in `data/r2b_galileo` demonstrates the expected format with 4 stereo camera pairs and multiple samples:

<details>
<summary><strong>📁 Directory Structure</strong></summary>

```
├── frames_meta.json
├── back_stereo_camera_left/
│   ├── 1707938736136244532.jpeg
│   ├── 1707938736169573532.jpeg
│   └── ... (30 total images)
├── back_stereo_camera_right/
│   ├── 1707938736136244532.jpeg
│   ├── 1707938736169573532.jpeg
│   └── ... (30 total images)
├── front_stereo_camera_left/
│   ├── 1707938736136247532.jpeg
│   ├── 1707938736169575532.jpeg
│   └── ... (29 total images)
├── front_stereo_camera_right/
│   ├── 1707938736136247532.jpeg
│   ├── 1707938736169575532.jpeg
│   └── ... (29 total images)
├── left_stereo_camera_left/
│   ├── 1707938736169592532.jpeg
│   ├── 1707938736202921532.jpeg
│   └── ... (29 total images)
├── left_stereo_camera_right/
│   ├── 1707938736169592532.jpeg
│   ├── 1707938736202921532.jpeg
│   └── ... (29 total images)
├── right_stereo_camera_left/
│   ├── 1707938736136246532.jpeg
│   ├── 1707938736169575532.jpeg
│   └── ... (28 total images)
└── right_stereo_camera_right/
    ├── 1707938736136246532.jpeg
    ├── 1707938736169575532.jpeg
    └── ... (28 total images)
```

</details>

### Rosbag Conversion

To convert a rosbag to the required mapping data format, follow the [ISAAC Mapping ROS tutorial](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_mapping_and_localization/isaac_mapping_ros/index.html). This will help you:

1. Set up the ISAAC Mapping ROS package
2. Use the `rosbag_to_mapping_data` binary
3. Provide initial pose estimates from a pose bag or TUM pose file

## Command Line Interface

PyCuSFM provides the `cusfm_cli` command-line tool that runs the complete SfM pipeline:

![PyCuSFM Overview](images/overview.png)

### Basic Usage

```bash
cusfm_cli --input_dir <input_dir> --cusfm_base_dir <cusfm_base_dir>
```

**Required Parameters:**
- `--input_dir`: Path to your mapping data
- `--cusfm_base_dir`: Output directory for PyCuSFM results

<details>
<summary><strong>⚙️ General Options</strong></summary>

- `--binary_dir <binary_dir>`: Specify path to cuSFM binary files
- `--config_dir <config_dir>`: Specify path to configuration files
- `--enable_debug`: Enable debug mode (saves intermediate results like feature extraction and matching)
- `--av_data`: Use default parameters optimized for outdoor autonomous driving scenarios
- `--use_rsc`: Enable rolling shutter correction (use when data contains rolling shutter correction information)

</details>

<details>
<summary><strong>🔄 Pipeline Control</strong></summary>

Control which steps to run using `--steps_to_run` or skip specific steps with `--skip_*` options:

```bash
# Run specific steps
--steps_to_run feature_extractor vocab_generator pose_graph matcher mapper map_convertor

# Skip specific steps
--skip_cuvslam
--skip_feature_extractor
--skip_vocab_generator
--skip_pose_graph
--skip_matcher
--skip_mapper
--skip_map_convertor
```

</details>

<details>
<summary><strong>🛠️ Step-Specific Options</strong></summary>

#### 1. Feature Extraction (`<cusfm_base_dir>/keyframes`)
- `--mask_dir <mask_dir>`: Specify image mask regions (masked areas are excluded from feature extraction)
- `--feature_type=[aliked,sift,superpoint]` (default: aliked)
- `--multi_track_input`: Process multiple image sequences

**Geometry-based Keyframe Selection:**
- `--min_inter_frame_distance`: Minimum translational distance (in meters) between consecutive keyframes for geometry-based selection (default: 0.5)
- `--min_inter_frame_rotation_degrees`: Minimum rotational change (in degrees) between consecutive keyframes for geometry-based selection (default: 5)

These parameters filter keyframes based on geometric criteria - only frames that exceed the specified translation or rotation thresholds relative to the previous keyframe will be selected.

#### 2. Dictionary Construction (`<cusfm_base_dir>/cuvgl_map`)

- `--skip_vocab_generator --cuvgl_dir=<cuvgl_dir>`: Use prebuilt vocabulary instead of generating a new one

#### 3. Feature Matching (`<cusfm_base_dir>/matches`)

- `--debug_interval`: Interval for saving debug matching images (default: 500)

#### 4. Pose Graph Optimization (`<cusfm_base_dir>/pose_graph`)

#### 5. Mapping (`<cusfm_base_dir>/kpmap`)
- `--ba_frame_type=vehicle_rig`: Fix extrinsics but do not optimize them
- `--optimize_extrinsics --ba_frame_type=vehicle_rig`: Enable extrinsic parameter refinement during bundle adjustment
- `--output_rgb`: Extract RGB colors for 3D map points from raw images
  - When enabled, RGB values are sampled from the original images at each 2D keypoint location
  - The final color for each 3D point is the average RGB across all its observations
  - Requires `--input_dir` to be set (uses raw images from input directory)
  - Example: `cusfm_cli --input_dir data/input --cusfm_base_dir results/output --output_rgb`

#### 6. COLMAP Conversion (`<cusfm_base_dir>/sparse`)

Converts poses and sparse 3D maps to COLMAP format for visualization.

- `--export_binary_colmap_files`: Export COLMAP data in binary format (`.bin`) instead of text format (`.txt`)
  - Default: Text format (`.txt` files)
  - Binary format is more compact and faster to load in COLMAP GUI
  - Example: `cusfm_cli --input_dir data/input --cusfm_base_dir results/output --export_binary_colmap_files`

**Output Files:**
- Text format (default): `cameras.txt`, `images.txt`, `points3D.txt`
- Binary format (with `--export_binary_colmap_files`): `cameras.bin`, `images.bin`, `points3D.bin`

**Camera Model Support:**

The following table shows how pycusfm camera models are mapped to COLMAP camera models during export:

| pycusfm Model | COLMAP Model | Parameters | Description |
|---------------|--------------|------------|-------------|
| `PINHOLE` | `PINHOLE` | fx, fy, cx, cy | Standard pinhole camera model |
| `DISTORTED_PINHOLE` | `FULL_OPENCV` | fx, fy, cx, cy, k1-k6, p1-p2 | Pinhole model with radial and tangential distortion |
| `FTHETA_WINDSHIELD` | `PINHOLE` | fx, fy, cx, cy | Fisheye camera behind windshield (AV scenarios). No direct COLMAP equivalent, so exports as undistorted PINHOLE using projection matrix intrinsics |

</details>

### Pre-compiling Model .engine Files

CUSFM supports multiple models for feature extraction and matching. During runtime, the system requires `.engine` files. If these files don't exist, the system will automatically load `.onnx` files from the model folder, compile them, and save the resulting `.engine` files. This compilation process can be time-consuming, but typically only needs to be performed once per device platform. Subsequent runs will use the previously saved `.engine` files for faster initialization.

Alternatively, you can use the Engine Exporter tool to pre-compile and save `.engine` files to the model folder before runtime. For detailed usage instructions of the Engine Exporter tool, please refer to the documentation [here](README_auxiliary_tools.md#tool-3-lightglue-engine-exporter).


## Quick Start Guide

Follow these steps to run PyCuSFM with example data:

### 1. Download Example Data

```bash
wget2 --max-threads=5 -r --no-parent --reject "index.html" --cut-dirs=5 -nH -P data/NRE1boxr0_2024_05_01-17_02_15/mapping_data https://pdx.s8k.io/v1/AUTH_team-osmo-ops/workflows/cuvslam_and_cusfm_benchmark-120/rosbag_to_mapping_data_conversion_0/
```

**Data Details:**
- Source: [NRE1boxr0 rosbag](https://swiftstack-maglev.ngc.nvidia.com/v1/AUTH_team-osmo-svc/isaac-safety/rosbags/inca/NRE/NRE1boxr0_2024_05_01-17_02_15/)
- Content: 8 cameras, 450 images per camera
- Initial poses: From cuVSLAM

### 2. Run PyCuSFM

```bash
cusfm_cli --input_dir data/NRE1boxr0_2024_05_01-17_02_15/mapping_data --cusfm_base_dir data/NRE1boxr0_2024_05_01-17_02_15/cusfm
```

### 3. Expected Output Structure

```
├── cuvgl_map
│   ├── bow_index.pb
│   └── vocabulary
├── keyframes
│   ├── back_stereo_camera_left
│   │   └── 1714608155873290968.pb
│   ├── ...
│   └── frames_meta.json
├── kpmap
│   ├── keyframes
│   └── map_keypoints.pb
├── matches
├── pose_graph
├── output_poses
│   ├── 0
│   │   ├── camera_name-back_stereo_camera_left_pose_file.txt
│   │   └── ...
│   └── merged_pose_file.tum
└── sparse
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

### Output Files Explained

- **`kpmap/keyframes/frames_meta.json`**: Optimized keyframe poses (PyCuSFM format)
- **`output_poses/`**: TUM format pose files
  - `merged_pose_file.tum`: Combined poses from all cameras
  - `0/`: Individual camera pose files
- **`sparse/`**: COLMAP sparse format for 3D point cloud and camera parameters


## Multi-track Input or Localization Mode

CuSFM supports joint mapping from multiple track inputs. The key difference in localization mode is that some tracks serve as fixed reference maps, while other tracks adjust their poses relative to these reference tracks. Therefore, input tracks can be categorized into **fixed tracks** (which can be considered as the map) and **floating tracks**.

### Use Cases

Multi-track input is ideal for:
- **Localization**: Use existing map tracks as reference for new data
- **Map merging**: Combine multiple mapping sessions
- **Cross-validation**: Compare results across different data collection runs
- **Incremental mapping**: Add new areas to existing maps

### Isaac Data or Stereo Camera Data

For Isaac data or data with stereo cameras, CuSFM can independently compute relative poses between tracks without requiring initial poses to be in the same coordinate system. This is achieved through stereo camera geometry and robust feature matching across tracks.

**Basic command for Isaac data:**
```bash
cusfm_cli \
    --input_dir <multi_track_data_dir> \
    --cusfm_base_dir <output_dir> \
    --multi_track_input \
    --anchor_track=track_folder_name0,track_folder_name1 \
    --use_cuvslam_slam_pose=False \
    --skip_pose_graph
```

### AV Data or Non-Stereo Camera Data

For AV data or data without stereo cameras, CuSFM currently cannot independently compute relative poses between tracks. Therefore, initial poses must be provided in the same coordinate system. This limitation requires pre-aligned pose estimates.

**Basic command for AV data:**
```bash
cusfm_cli \
    --input_dir <multi_track_data_dir> \
    --cusfm_base_dir <output_dir> \
    --multi_track_input \
    --anchor_track=track_folder_name0,track_folder_name1 \
    --av_data
```

### Key Parameters

- **`--multi_track_input`**: Enables multi-track processing mode
- **`--anchor_track`**: Comma-separated list of track folder names to fix during mapping. These tracks serve as the reference coordinate system.

### Data Organization

Multi-track data should be organized as follows:

```
multi_track_data_dir/
├── track_00/
│   ├── camera_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── camera_2/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── frames_meta.json
├── track_01/
│   ├── camera_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── camera_2/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── frames_meta.json
└── ...
```

Each track folder contains its own camera subdirectories and a `frames_meta.json` file with pose information for that track.

**Data Requirements per Track:**
- At least one camera subdirectory with images
- `frames_meta.json` with timestamp, pose, and image path information
- Consistent naming convention across tracks (camera folder names should match)
- For stereo setups: left/right camera pairs should be clearly identified


## KITTI Dataset Example

This section demonstrates how to run stereo visual odometry on the KITTI dataset.

### Dataset Setup

1. **Download KITTI Data**
   - Visit [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
   - Register for an account
   - Download `data_odometry_gray.zip` (22 GB)

2. **Extract Data**
   ```bash
   unzip data_odometry_gray.zip -d data/kitti
   ```

3. **frames_meta.json File Setup**

    KITTI datasets don't include the `frames_meta.json` file required by PyCuSFM.
    The provided script can generate the `frames_meta.json` file from the KITTI dataset:

    ```bash
    python data/kitti/get_framemeta_file_for_KITTI.py <dataset_dir> [--output-name OUTPUT_NAME]
    ```

    For other custom datasets, you can refer to this script as a reference to generate the required `frames_meta.json` file.

4. **Expected Structure**
   ```
   data/kitti/
   ├── 00
   │   ├── image_0
   │   │   ├── 000000.png
   │   │   └── ...
   │   ├── image_1
   │   │   ├── 000000.png
   │   │   └── ...
   │   └── frames_meta.json
   ├── 02
   │   └── ...
   └── config/
   ```

### Running PyCuSFM on KITTI

Execute the following command for KITTI sequence 00:

```bash
cusfm_cli --input_dir data/kitti/00 --cusfm_base_dir data/kitti/00_result --config_dir data/kitti/config
```

**Note:** The experiments in the cuSFM paper were conducted using pose graph optimization without data association. You can use the `--skip_data_association` flag to skip the data association step.


## Bundle Adjustment Runner

The `bundle_adjustment_runner` is a standalone tool that performs bundle adjustment optimization on COLMAP sparse reconstruction data using the Isaac Visual Mapping bundle adjustment infrastructure. This tool is useful when you have existing COLMAP sparse model files and want to refine them using the robust Isaac bundle adjustment solver.

### Usage

```bash
cusfm_tool --binary_name bundle_adjustment_runner \
    --args "--colmap_sparse_dir /path/to/colmap/sparse --output_dir /path/to/output [options]"
```

### Key Features

- **COLMAP Format Support**: Reads standard COLMAP sparse model files (images.txt, cameras.txt, points3D.txt) in both text and binary formats
- **Isaac Bundle Adjustment**: Uses the same robust bundle adjustment infrastructure as the main cuSFM pipeline
- **Configurable Optimization**: Supports various loss functions, iteration limits, and solver options
- **Output Formats**: Saves optimized camera poses, 3D points, and camera parameters in readable formats

### Integration with cuSFM Pipeline

The bundle adjustment runner can be used as a standalone tool or integrated into the cuSFM pipeline:

1. **Standalone Usage**: Directly optimize existing COLMAP sparse models
2. **Post-Processing**: Refine results from the cuSFM pipeline's COLMAP output
3. **Comparison**: Compare Isaac bundle adjustment results with COLMAP's native bundle adjuster

For detailed usage instructions and configuration options, see the [auxiliary tools document](README_auxiliary_tools.md).
