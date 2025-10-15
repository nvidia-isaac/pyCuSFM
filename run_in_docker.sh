#!/bin/bash

# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-License

# PyCuSFM Docker Runner Script
# This script manages Docker container operations for PyCuSFM

set -e  # Exit on any error

# Default values
BUILD_DOCKER=false
INSTALL_PYCUSFM=false
DOCKER_IMAGE="pycusfm:latest"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build_docker    Build Docker image before running container"
    echo "  --install         Install PyCuSFM inside the container after starting"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Just run Docker container"
    echo "  $0 --build_docker            # Build image then run container"
    echo "  $0 --install                 # Run container and install PyCuSFM"
    echo "  $0 --build_docker --install  # Build image, run container, and install PyCuSFM"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build_docker)
            BUILD_DOCKER=true
            shift
            ;;
        --install)
            INSTALL_PYCUSFM=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "========================================"
echo "    PyCuSFM Docker Runner Script       "
echo "========================================"
echo

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "[ERROR] Docker daemon is not running. Please start Docker service."
    exit 1
fi

# Function to build Docker image
build_docker_image() {
    echo "[INFO] Building Docker image..."
    
    if [[ ! -f "docker/DockerFile.dev" ]]; then
        echo "[ERROR] docker/DockerFile.dev not found. Please run this script from the PyCuSFM root directory."
        exit 1
    fi
    
    if docker build -f docker/DockerFile.dev -t "$DOCKER_IMAGE" .; then
        echo "[SUCCESS] Docker image built successfully: $DOCKER_IMAGE"
    else
        echo "[ERROR] Failed to build Docker image"
        exit 1
    fi
    echo
}

# Function to run Docker container
run_docker_container() {
    echo "[INFO] Starting Docker container..."
    
    # Prepare docker run command as array
    DOCKER_CMD=(
        "docker" "run" "-it" "--rm"
        "--runtime" "nvidia" "--privileged" "--network" "host"
        "--user" "$(id -u):$(id -g)"
        "-e" "HOME=/pycusfm"
        "-v" "$PWD:/pycusfm"
        "--workdir" "/pycusfm"
        "$DOCKER_IMAGE"
    )

    if [[ "$INSTALL_PYCUSFM" == "true" ]]; then
        # Run container and install PyCuSFM, then start interactive bash
        echo "[INFO] Running container with PyCuSFM installation..."
        "${DOCKER_CMD[@]}" bash -c "
            # Add ~/.local/bin to PATH for pip user installations
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            
            echo '[INFO] Installing PyCuSFM in docker environment...'
            if pip3 install --editable .; then
                echo '[SUCCESS] PyCuSFM installed successfully'
                echo '[INFO] Starting interactive bash session...'
                exec /bin/bash
            else
                echo '[ERROR] Failed to install PyCuSFM'
                exit 1
            fi
        "
    else
        # Just run container with interactive bash
        echo "[INFO] Running container in interactive mode..."
        "${DOCKER_CMD[@]}" bash -c "
            # Add ~/.local/bin to PATH for pip user installations
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            exec /bin/bash
        "
    fi
}

# Main execution flow
echo "[INFO] Configuration:"
echo "  Build Docker image: $BUILD_DOCKER"
echo "  Install PyCuSFM: $INSTALL_PYCUSFM"
echo "  Docker image: $DOCKER_IMAGE"
echo

# Step 1: Build Docker image if requested
if [[ "$BUILD_DOCKER" == "true" ]]; then
    build_docker_image
fi

# Step 2: Check if Docker image exists
if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo "[ERROR] Docker image '$DOCKER_IMAGE' not found."
    echo "[INFO] Please build the image first using: $0 --build_docker"
    echo "[INFO] Or build manually with: docker build -f docker/DockerFile.dev -t $DOCKER_IMAGE ."
    exit 1
fi

# Step 3: Run Docker container
run_docker_container

echo
echo "[INFO] Docker container session ended."
