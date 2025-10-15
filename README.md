# PyCuSFM: Cuda Accelerated Structure from Motion

[![platform](https://img.shields.io/badge/Platform-ubuntu--24.04-FCC624.svg?logo=ubuntu)](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
[![arXiv](https://img.shields.io/badge/Arxiv-2510.15271-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.15271)

## Overview

This repository provides the official python implementation of [cuSFM](https://arxiv.org/abs/2510.15271), a novel CUDA-accelerated Structure-from-Motion framework for reconstructing 3D environmental models from images. Key features include:

- CUDA-accelerated feature extraction, matching, and graph optimization for superior speed and scalability
- Precise and robust camera pose estimation
- Accurate and consistent 3D environment reconstruction with COLMAP-compatible outputs
- Support for any number and type of camera inputs
- Reliable extrinsic calibration for multi-camera setups
- Localization mode for integrating new data into pre-built maps

![cusfm_png](docs/images/cusfm.png)

Refer to [paper](https://arxiv.org/abs/2510.15271) for technical details and benchmark results about cuSFM.

## COLMAP vs cuSFM

| Feature | COLMAP | cuSFM |
|---------|--------|---------|
| **Trajectory Initialization** | Not required | Required |
| **Dense Reconstruction** | Supported | Not supported |
| **Built-in Features** | SIFT | ALIKED, SIFT_CV_CUDA, SuperPoint |
| **Vocabulary Building** | Supported | Supported |
| **Bundle Adjustment** | Incremental | Global |
| **Pose Graph Optimization** | Not supported | Supported |
| **Camera-to-Camera Extrinsic Optimization** | Supported | Supported |
| **Localization** | Supported | Supported |


## Installation

### Prerequisites

Before installation, ensure your system meets the following requirements:
- **Operating System:** Ubuntu 24.04 LTS (recommended)
- **NVIDIA Driver:** Version 560 or higher
- **CUDA Toolkit:** 12.6 or compatible version
- **Python:** Version 3.8 or higher
- **Git LFS:** For downloading large model files

### Download Repository

Install Git LFS (if not already installed):
```bash
# Install Git LFS
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install
```

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/nvidia-isaac/PyCuSFM
cd pycusfm
```

Choose one of the following installation methods based on your preference:

<details>
<summary><strong>📦 Method 1: Direct Installation on Host Machine</strong></summary>

#### Step 1: Verify System Requirements

Ensure your system meets the prerequisites listed above. You can check your current setup with:

```bash
# Check NVIDIA driver version
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python version
python3 --version
```

#### Step 2: Run Installation Script

Use the provided installation script to automatically install system dependencies and PyCuSFM:

```bash
# Run the automated installation script
./install_in_host.sh
```

The script will:
- Update the package list
- Install all required system dependencies (OpenCV, Google Logging, Protocol Buffers, etc.)
- Install PyCuSFM in development mode
- Verify the installation by testing the `cusfm_cli` command

#### Step 3: Add Installation Path to PATH

After installation, you need to add the PyCuSFM installation path to your environment variables.

**For most users** (default installation), the binaries are installed in `$HOME/.local/bin`:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**For virtual environment users**, you need to add your custom installation path to the environment variables.


</details>

<details>
<summary><strong>🐳 Method 2: Docker Environment</strong></summary>


#### Step 1: Install Docker and NVIDIA Container Toolkit

- Install NVIDIA Container Toolkit by following the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

- Configure NVIDIA GPU Cloud (NGC) access by following [this guide](https://docs.nvidia.com/ngc/latest/ngc-user-guide.html) if you need to pull from NGC registry.

#### Step 2: Run Docker Script

Use the provided Docker runner script for automated container management:

```bash
# Build Docker image and run container with PyCuSFM installation
./run_in_docker.sh --build_docker --install
```

**Script Options:**
- `./run_in_docker.sh` - Just run Docker container (image must exist)
- `./run_in_docker.sh --build_docker` - Build image then run container
- `./run_in_docker.sh --install` - Run container and install PyCuSFM
- `./run_in_docker.sh --build_docker --install` - Build image, run container, and install PyCuSFM


**Compatibility Note:** The base image is `nvcr.io/nvidia/tensorrt:24.12-py3`. Check support matrix [here](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2024).

</details>


## Usage

### Quick Start

```bash
# Basic usage
cusfm_cli --input_dir <input_dir> --cusfm_base_dir <output_dir>

# Example with sample data
cusfm_cli --input_dir data/r2b_galileo --cusfm_base_dir results/cusfm_output
```

For detailed instructions, examples, and advanced configurations, see our [Complete Tutorial](docs/tutorial.md) covering:

- **Data Requirements**: Input format and structure
- **Command Line Options**: All available parameters and flags
- **Quick Start Guide**: Step-by-step example with sample data
- **KITTI Dataset**: Running on standard benchmark datasets
- **Advanced Features**: Multi-camera setups, AV data, rolling shutter correction, bundle adjustment runner for COLMAP format


## Acknowledgments

We would like to express our gratitude to the authors of the following projects, whose work has significantly contributed to the development of cuSFM:

- [PyCuVSLAM](https://github.com/NVlabs/PyCuVSLAM)
- [cuDSS](https://developer.nvidia.com/cudss)
- [CV-CUDA](https://github.com/CVCUDA/CV-CUDA)
- [ALIKED-TensorRT](https://github.com/ajuric/aliked-tensorrt)
- [LightGlue](https://github.com/cvg/LightGlue)
- [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
- [COLMAP](https://github.com/colmap/colmap/tree/main)
- [SuperPoint](https://github.com/rpautrat/SuperPoint)

## Third-Party Dependency

| Name | Version | License |
|------|---------|---------|
| googletest | 1.14.0 | BSD |
| glog | 0.6.0 | BSD |
| ceres-solver | 2.2.0 | Apache License 2.0 |
| protobuf | 3.21.12 | BSD |
| opencv | 4.6.0 | Apache License 2.0 |
| LightGlueONNX | 2.0 | Apache License 2.0 |
| aliked-tensorrt | main | BSD |
| SuperPoint | master | MIT |
| eigen | 3.4.0 | MPL 2.0 |


## Citation

If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.
```bibTeX
@article{2025CuSfM,
  title={CuSfM: CUDA-Accelerated Structure-from-Motion},
  author={Yu, Jingrui and Liu, Jun and Ren, Kefei and Biswas, Joydeep and Ye, Rurui and Wu, Keqiang and Majithia, Chirag and Zeng, Di},
  journal={arXiv preprint arXiv:2510.15271},
  year={2025},
  eprint={2510.15271},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
