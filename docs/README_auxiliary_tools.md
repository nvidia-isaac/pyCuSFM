# PyCuSFM Auxiliary Tools

This document describes auxiliary tools that support various operations in the PyCuSFM pipeline, including COLMAP format data processing and TensorRT engine generation for deep learning models.

## Overview

This document covers five main auxiliary tools:

1. **[Bundle Adjustment Runner](#tool-1-bundle-adjustment-runner)** - Optimizes COLMAP sparse reconstruction data using bundle adjustment
2. **[COLMAP to Keyframe Converter](#tool-2-colmap-to-keyframe-converter)** - Converts COLMAP sparse model data to keyframe metadata format
3. **[LightGlue Engine Exporter](#tool-3-lightglue-engine-exporter)** - Exports TensorRT engine for LightGlue feature matching models
4. **[Feature Extractor Engine Exporter](#tool-4-feature-extractor-engine-exporter)** - Exports TensorRT engine for feature extraction models
5. **[Keyframe Metadata Generator](#tool-5-keyframe-metadata-generator)** - Generates `frames_meta.json` from poses.csv or stereo image folders

All tools are accessed through the `cusfm_tool` command-line interface and integrate seamlessly with the Isaac Visual Mapping infrastructure.

---

# Tool 1: Bundle Adjustment Runner

The `bundle_adjustment_runner` binary reads COLMAP sparse model files (images.txt, cameras.txt, points3D.txt) and performs bundle adjustment optimization using the Isaac Visual Mapping bundle adjustment infrastructure. It optimizes camera poses and 3D point positions to minimize reprojection error.

## Usage

**Basic Usage:**
```bash
cusfm_tool --binary_name bundle_adjustment_runner \
  --args "--colmap_sparse_dir=<sparse_directory> \
          --output_dir=<output_directory> \
          [options]"
```

**Example:**
```bash
# Run bundle adjustment on COLMAP sparse reconstruction
cusfm_tool --binary_name bundle_adjustment_runner \
  --args "--colmap_sparse_dir=/path/to/colmap/sparse \
          --output_dir=/path/to/optimized_results \
          --max_iterations=200 \
          --verbose=true \
          --loss_function=CAUCHY \
          --fix_first_camera=true"
```

## Required Arguments

- `--colmap_sparse_dir`: Path to COLMAP sparse model directory containing:
  - `images.txt` or `images.bin` (camera poses and 2D keypoints)
  - `cameras.txt` or `cameras.bin` (camera intrinsics)
  - `points3D.txt` or `points3D.bin` (3D points and their observations)

- `--output_dir`: Directory to save optimized results

## Optional Arguments

- `--max_iterations=100`: Maximum number of bundle adjustment iterations
- `--verbose=false`: Enable verbose optimization output
- `--use_cudss_solver=false`: Use CUDA sparse solver if available
- `--reprojection_error_std_dev=4.0`: Standard deviation of reprojection error in pixels
- `--loss_function=CAUCHY`: Loss function type (TRIVIAL, SOFT_L1, CAUCHY, HUBER)
- `--loss_function_scale=1.0`: Scale parameter for loss function
- `--fix_first_camera=true`: Fix the first camera pose as reference frame
- `--refine_extrinsics=false`: Refine camera extrinsic parameters

## Output Files

The tool generates the following output files in the specified output directory:

1. `optimized_poses.txt`: Optimized camera poses in format:
   ```
   # Image ID, Camera ID, qw, qx, qy, qz, tx, ty, tz
   0 0 0.707 0 0.707 0 1.0 2.0 3.0
   ```

2. `optimized_points.txt`: Optimized 3D points in format:
   ```
   # Point ID, X, Y, Z, Num Observations
   0 1.5 2.3 4.1 5
   ```

3. `optimized_cameras.txt`: Camera intrinsic parameters in format:
   ```
   # Camera ID, fx, fy, cx, cy, width, height
   0 1000.0 1000.0 960.0 540.0 1920 1080
   ```

## Technical Details

- The tool supports both text and binary COLMAP file formats
- Only 3D points with at least 2 observations are included in optimization
- Camera poses are represented as quaternions and translations
- The first camera pose is typically fixed as a reference frame to remove gauge freedom
- The optimization uses Ceres Solver with configurable loss functions for robustness

## Integration with COLMAP Pipeline

This tool can be used as a replacement for COLMAP's bundle_adjuster in a reconstruction pipeline:

1. Run COLMAP feature extraction and matching
2. Run COLMAP sparse reconstruction (mapper)
3. Run this bundle adjuster for optimization
4. Use the optimized results for dense reconstruction or other applications

## Performance

- The tool is optimized for large-scale reconstructions
- Supports multi-threading for faster optimization
- Can use CUDA sparse solver for GPU acceleration when available
- Memory usage scales with the number of 3D points and observations

---

# Tool 2: COLMAP to Keyframe Converter

The `colmap_to_keyframe_convertor` binary converts COLMAP sparse reconstruction data into keyframe metadata format used by the cuSfM system. It can optionally perform scale fitting using ground truth poses.

## Usage

This tool has two basic usage example commands:

```bash
cusfm_tool --binary_name colmap_to_keyframe_convertor \
  --args "--colmap_sparse_dir=/path/to/colmap/sparse \
          --output_keyframe_file=/path/to/output/keyframes.json \
          --gt_pose_tum_file=/path/to/groundtruth.txt \
          [options]"
```
or

```bash
cusfm_tool --binary_name colmap_to_keyframe_convertor \
  --args "--colmap_sparse_dir=/path/to/colmap/sparse \
          --output_keyframe_file=/path/to/output/keyframes.json \
          --ref_keyframes_file=/path/to/reference_keyframes.json \
          [options]"
```

## Required Arguments

- `--colmap_sparse_dir`: Path to COLMAP sparse model directory containing:
  - `images.txt` or `images.bin` (camera poses and 2D keypoints)
  - `cameras.txt` or `cameras.bin` (camera intrinsics)
  - `points3D.txt` or `points3D.bin` (3D points and their observations)

- `--output_keyframe_file`: Output file path to save keyframe metadata in JSON format

## Optional Arguments

- `--ref_keyframes_file`: Reference keyframe metadata file to get timestamps for ground truth pose queries
- `--gt_pose_tum_file`: Ground truth poses in TUM format for scale fitting (requires `--ref_keyframes_file`)
- `--camera_params_scale_factor=1.0`: Scale factor for camera parameters
- `--session_name`: Session name for this COLMAP project

---

# Tool 3: LightGlue Engine Exporter

The `export_lightglue_engine` binary exports TensorRT engines for LightGlue feature matching models. This tool optimizes deep learning models for GPU inference by converting them to TensorRT format, which provides significant performance improvements during feature matching operations.

## Usage

**Basic Usage:**
```bash
cusfm_tool --binary_name export_lightglue_engine \
  --args "--configure_file=<path_to_configure_file> \
          --model_dir=<path_to_model_dir> \
          --engine_file_path=<path_to_engine_file>"
```

**Example:**
```bash
# Export LightGlue TensorRT engine for ALIKED features
cusfm_tool --binary_name export_lightglue_engine \
  --args "--configure_file ./pycusfm/configs/isaac/matching_task_worker_config.pb.txt \
          --model_dir ./pycusfm/models/aliked_lightglue \
          --engine_file_path ./pycusfm/models/aliked_lightglue/lightglue_aliked_fp16_10_7_0_23_sm_8_9.engine"
```

## Required Arguments

- `--configure_file`: Path to the matching task worker configuration file (protobuf text format)
- `--model_dir`: Directory containing the LightGlue model files (ONNX format)
- `--engine_file_path`: Output path for the generated TensorRT engine file

---

# Tool 4: Feature Extractor Engine Exporter

The `export_extractor_engine` binary exports TensorRT engines for feature extraction models including ALIKED and SuperPoint. This tool converts deep learning models to optimized TensorRT format for faster feature extraction during the SfM pipeline.

## Usage

**Basic Usage:**
```bash
cusfm_tool --binary_name export_extractor_engine \
  --args "--configure_file=<path_to_configure_file> \
          --model_dir=<path_to_model_dir> \
          --runner_engine_file_path=<path_to_engine_file> \
          --feature_type=<aliked,superpoint>"
```

**Example:**
```bash
# Export ALIKED TensorRT engine
cusfm_tool --binary_name export_extractor_engine \
  --args "--configure_file ./pycusfm/configs/isaac/keypoint_creation_config.pb.txt \
          --model_dir ./pycusfm/models/aliked_lightglue \
          --runner_engine_file_path ./pycusfm/models/aliked_lightglue/aliked_fp16_10_7_0_23_sm_8_9.engine \
          --feature_type aliked"
```

## Required Arguments

- `--configure_file`: Path to the keypoint creation configuration file (protobuf text format)
- `--model_dir`: Directory containing the feature extractor model files (ONNX format)
- `--runner_engine_file_path`: Output path for the generated TensorRT engine file
- `--feature_type`: Type of feature extractor (aliked, superpoint)

---

# Tool 5: Keyframe Metadata Generator

Generates `frames_meta.json` from `poses.csv` or image folders for cuSFM/cuVSLAM compatibility.

**Two Modes:**
1. **POSES MODE**: Generate from poses.csv with real camera poses (for any number of cameras)
2. **IMAGES MODE**: Generate from stereo image folders with identity poses (for initial mapping with cuSFM)
   - **Only supports single stereo pair** (left/right cameras)
   - **No initial pose required** (identity transforms used)

---

#### Mode 1: POSES MODE

**Usage:**
```bash
python pycusfm/generate_frame_meta.py \
    --poses /path/to/poses.csv \
    --config /path/to/config.yaml \
    --output /path/to/frames_meta.json
```

**Parameters:**
- `--poses` - Path to CSV pose file containing camera poses
- `--config` - Path to YAML config file with camera parameters
- `--output` - Output path for generated frames_meta.json

**Input File Formats:**

##### poses.csv

A CSV file with header row containing camera pose information. Required and optional columns:

| Column | Required | Description |
|--------|----------|-------------|
| `image_id` | Yes | Unique identifier for the image |
| `qw` | Yes | Quaternion W component (scalar-first) |
| `qx` | Yes | Quaternion X component |
| `qy` | Yes | Quaternion Y component |
| `qz` | Yes | Quaternion Z component |
| `tx` | Yes | Translation X (meters) |
| `ty` | Yes | Translation Y (meters) |
| `tz` | Yes | Translation Z (meters) |
| `camera_params_id` | Yes | Camera parameters ID (use `-` for cameras without calibration) |
| `image_name` | Yes | Relative path to image file |
| `synced_sample_id` | Optional | Synced sample ID for multi-camera synchronization |
| `timestamp_us` | Optional | Timestamp in microseconds |

**Optional Column Handling:**

**`synced_sample_id` (Optional):**
- **Purpose**: Groups frames captured simultaneously across multiple cameras (e.g., 8 cameras on a vehicle rig capturing at the same time instant)
- **Can be omitted**: If missing or empty, the tool will **automatically detect** synchronized frames based on timestamps
  - Uses 0.05ms threshold (configurable in code)
  - Groups frames from all cameras within the threshold
  - Assigns same `synced_sample_id` to synchronized frames

**`timestamp_us` (Optional):**
- **Purpose**: Timestamp in microseconds for each frame
- **Can be omitted**: If missing or empty, the tool will **automatically extract** from `image_name`
  - Expected image name format: `camera_name/timestamp_nanoseconds.extension`
  - Example: `front_stereo_camera_left/1707938736136246532.jpeg`
  - Tool will convert nanoseconds to microseconds
- **Error**: If both `timestamp_us` is missing AND image name doesn't contain a parseable timestamp, the tool will exit with an error message

**Example poses.csv (full format with all columns):**
```csv
image_id,qw,qx,qy,qz,tx,ty,tz,camera_params_id,synced_sample_id,timestamp_us,image_name
2356,0.01832,0.00854,-0.70845,0.70547,4.6072,-0.0672,0.2406,6,318,1707938736136246,right_stereo_camera_left/1707938736136246532.jpeg
2357,0.48765,-0.50243,0.51488,-0.49464,4.9816,-0.0054,0.2279,3,318,1707938736136247,front_stereo_camera_right/1707938736136247532.jpeg
2363,0.51494,-0.49372,-0.48867,0.50228,4.3324,0.0051,0.2507,-,318,1707938736136244,back_stereo_camera_left/1707938736136244532.jpeg
```

**Example poses.csv (minimal format with only required columns):**
```csv
image_id,qw,qx,qy,qz,tx,ty,tz,camera_params_id,image_name
2356,0.01832,0.00854,-0.70845,0.70547,4.6072,-0.0672,0.2406,6,right_stereo_camera_left/1707938736136246532.jpeg
2357,0.48765,-0.50243,0.51488,-0.49464,4.9816,-0.0054,0.2279,3,front_stereo_camera_right/1707938736136247532.jpeg
2363,0.51494,-0.49372,-0.48867,0.50228,4.3324,0.0051,0.2507,-,back_stereo_camera_left/1707938736136244532.jpeg
```

##### config.yaml

A YAML configuration file containing camera calibration parameters and metadata.

**Required/Optional Fields:**

```yaml
# Pose type for initialization (optional, default: EGO_MOTION)
initial_pose_type: EGO_MOTION  # Options: EGO_MOTION | IDENTITY | UNKNOWN

# Image dimensions - all cameras must use the same resolution
image_width: 1920
image_height: 1200
frequency: 30

# Session to camera params ID mapping (required)
# Maps session names to comma-separated camera IDs
session_camera_params_id_mapping:
  session_0: 0, 1, 2, 3, 4, 5, 6, 7
  session_1: 8, 9, 10, 11

# Camera parameters (required)
# Key is camera_params_id, value contains calibration data
camera_params:
  0:
    sensor_name: front_stereo_camera_left
    # Extrinsic: [QW, QX, QY, QZ, TX, TY, TZ] - camera to vehicle transform
    extrinsic: [0.5, -0.5, 0.5, -0.5, 0.0, 0.0, 0.5]
    # Intrinsic: 3x3 camera matrix (row-major, 9 values)
    # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    intrinsic: [1000.0, 0, 960, 0, 1000.0, 600, 0, 0, 1]
    # Distortion coefficients (optional)
    # Empty or all zeros -> PINHOLE model
    # Non-zero values -> DISTORTED_PINHOLE model
    distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
  1:
    sensor_name: front_stereo_camera_right
    extrinsic: [0.5, -0.5, 0.5, -0.5, 0.15, 0.0, 0.5]
    intrinsic: [1000.0, 0, 960, 0, 1000.0, 600, 0, 0, 1]
    distortion_coefficients: []

# Stereo pair configuration (optional)
# Each entry: [left_camera_id, right_camera_id, baseline_meters]
stereo_pair:
  - [0, 1, 0.15]
  - [2, 3, 0.15]
```

**Output:**

The script generates a `frames_meta.json` file with the following structure:

```json
{
  "keyframes_metadata": [
    {
      "id": "2356",
      "camera_params_id": "6",
      "timestamp_microseconds": "1707938736136246",
      "image_name": "right_stereo_camera_left/1707938736136246532.jpeg",
      "camera_to_world": {
        "axis_angle": {"x": 0.0, "y": 0.0, "z": 1.0, "angle_degrees": 90.0},
        "translation": {"x": 4.607, "y": -0.067, "z": 0.241}
      },
      "synced_sample_id": "318"
    }
  ],
  "initial_pose_type": "EGO_MOTION",
  "camera_params_id_to_session_name": {"0": "0", "1": "0"},
  "camera_params_id_to_camera_params": {...},
  "stereo_pair": [...]
}
```

---

#### Mode 2: IMAGES MODE (Stereo Image Folders)

**Purpose**: Generate `frames_meta.json` from raw stereo images **without** existing poses. Outputs identity transforms suitable for initial mapping with cuSFM.

**Limitations**:
- **Only supports single stereo pair** (one left camera, one right camera)
- **No initial pose values** - all camera_to_world transforms are identity (zeros)

**Usage:**
```bash
python pycusfm/generate_frame_meta.py \
    --images /path/to/image_folder \
    --config /path/to/config.yaml \
    --output /path/to/frames_meta.json \
    [--use-pseudo-timestamps]
```

**Parameters:**
- `--images` - Path to directory containing left/right camera subdirectories
- `--config` - Path to YAML config file with camera parameters
- `--output` - Output path for generated frames_meta.json
- `--use-pseudo-timestamps` - (Optional) Use sequential 30Hz timestamps instead of extracting from filenames

**Input Requirements:**

**Directory Structure:**
```
/path/to/image_folder/
├── <folder_with_left>/     # Must contain "left" in name (case-insensitive)
│   ├── 1762401096580801652.jpeg
│   ├── 1762401096614135319.jpeg
│   └── ...
└── <folder_with_right>/    # Must contain "right" in name
    ├── 1762401096580801652.jpeg
    ├── 1762401096614135319.jpeg
    └── ...
```

**Folder Naming**: Folders must contain keywords "left" or "right" (case-insensitive)
- ✅ Valid: `camera_left`, `Left`, `stereo_left`, `LEFT_CAM`
- ✅ Valid: `camera_right`, `Right`, `stereo_right`, `RIGHT_CAM`

**Image Naming**:
- **Default mode** (without `--use-pseudo-timestamps`):
  - Filenames must be timestamps in **nanoseconds**
  - Example: `1762401096580801652.jpeg`
  - Tool converts nanoseconds to microseconds automatically
- **Pseudo-timestamp mode** (with `--use-pseudo-timestamps`):
  - Filenames can be anything (e.g., `img_0001.jpg`)
  - Tool generates sequential timestamps at 30Hz

**Config File (Minimal - for Images Mode):**
```yaml
# Minimal config - only intrinsics required!
camera_params:
  0:  # Left camera
    intrinsic: [1916.79, 0, 1118.24, 0, 1916.79, 635.51, 0, 0, 1]
    # Optional: extrinsic defaults to [1, 0, 0, 0, 0, 0, 0] (identity)
    # Optional: distortion_coefficients defaults to [] (no distortion)

  1:  # Right camera
    intrinsic: [1916.79, 0, 1118.24, 0, 1916.79, 635.51, 0, 0, 1]
    extrinsic: [1.0, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0]  # Baseline in TX

# Auto-generated if missing:
# - initial_pose_type: EGO_MOTION
# - image_width/height: 1920x1200
# - frequency: 30
# - session_camera_params_id_mapping: {session_0: "0, 1"}
# - stereo_pair: [[0, 1, 0.12]] (baseline from camera 1 extrinsic)
```

**Notes:**
- Requires `PyYAML` and `SciPy` (`pip install pyyaml scipy`) for YAML parsing and quaternion conversion.

---

# TensorRT Engine Naming Convention

The TensorRT engine files follow a specific naming convention that encodes important version and hardware information:

## Naming Format

```
{model_name}_{tensorrt_major}_{tensorrt_minor}_{tensorrt_patch}_{tensorrt_build}_sm_{compute_major}_{compute_minor}.engine
```

## Components

- `{model_name}`: Name of the deep learning model (e.g., "aliked", "lightglue_aliked", "superpoint")
- `{tensorrt_major}`: TensorRT major version (e.g., "10")
- `{tensorrt_minor}`: TensorRT minor version (e.g., "7")  
- `{tensorrt_patch}`: TensorRT patch version (e.g., "0")
- `{tensorrt_build}`: TensorRT build version (e.g., "23")
- `{compute_major}`: GPU compute capability major version (e.g., "8")
- `{compute_minor}`: GPU compute capability minor version (e.g., "9")

## Examples

- `aliked_fp16_10_7_0_23_sm_8_9.engine`: ALIKED model with FP16 precision, TensorRT 10.7.0.23, Compute Capability 8.9
