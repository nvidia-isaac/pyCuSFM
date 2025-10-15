# PyCuSFM Auxiliary Tools

This document describes auxiliary tools that support various operations in the PyCuSFM pipeline, including COLMAP format data processing and TensorRT engine generation for deep learning models.

## Overview

This document covers four main auxiliary tools:

1. **[Bundle Adjustment Runner](#tool-1-bundle-adjustment-runner)** - Optimizes COLMAP sparse reconstruction data using bundle adjustment
2. **[COLMAP to Keyframe Converter](#tool-2-colmap-to-keyframe-converter)** - Converts COLMAP sparse model data to keyframe metadata format
3. **[LightGlue Engine Exporter](#tool-3-lightglue-engine-exporter)** - Exports TensorRT engine for LightGlue feature matching models
4. **[Feature Extractor Engine Exporter](#tool-4-feature-extractor-engine-exporter)** - Exports TensorRT engine for feature extraction models

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
