#!/bin/bash

# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-License

# PyCuSFM Host Installation Script
# This script installs system dependencies and PyCuSFM

set -e  # Exit on any error

# Logging functions
log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_warning() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1"
}

# Exit with error message
exit_with_error() {
    log_error "$1"
    exit 1
}

echo "========================================"
echo "    PyCuSFM Host Installation Script    "
echo "========================================"
echo

# ============================================================================
# MODULE 1: System Dependencies Installation
# ============================================================================

log_info "MODULE 1: Installing system dependencies..."
echo

# Update package list
log_info "Updating package list..."
if sudo apt-get update; then
    log_success "Package list updated successfully"
else
    exit_with_error "Failed to update package list"
fi

# Define dependencies with descriptions
declare -A DEPENDENCIES=(
    ["libopencv-dev"]="OpenCV development libraries for computer vision operations"
    ["libgoogle-glog-dev"]="Google Logging library for logging functionality"
    ["libgflags-dev"]="Google command-line flags library for argument parsing"
    ["libabsl-dev"]="Abseil C++ library for core utilities"
    ["libprotobuf-dev"]="Protocol Buffers development libraries for data serialization"
    ["protobuf-compiler"]="Protocol Buffers compiler for data serialization"
)

# Install each dependency
for package in "${!DEPENDENCIES[@]}"; do
    description="${DEPENDENCIES[$package]}"
    log_info "Installing $package ($description)..."
    
    if sudo apt-get install -y "$package"; then
        log_success "$package installed successfully"
    else
        exit_with_error "Failed to install $package. Please check your internet connection and package availability."
    fi
done

log_success "All system dependencies installed successfully!"
echo

# ============================================================================
# MODULE 2: PyCuSFM Installation and Verification
# ============================================================================

log_info "MODULE 2: Installing PyCuSFM..."
echo

# Check if we're in the correct directory
if [[ ! -f "pyproject.toml" ]]; then
    exit_with_error "pyproject.toml not found. Please run this script from the PyCuSFM root directory."
fi

# Install PyCuSFM in development mode
log_info "Installing PyCuSFM in development mode..."
if pip3 install -e .; then
    log_success "PyCuSFM installed successfully"
else
    exit_with_error "Failed to install PyCuSFM. Please check the error messages above."
fi

# Verify installation by testing cusfm_cli command
log_info "Verifying installation..."
echo

# Wait a moment for the installation to complete
sleep 2

# Test if cusfm_cli command exists and is accessible
if command -v cusfm_cli &> /dev/null; then
    log_success "cusfm_cli command found and accessible"
    
    # Test if cusfm_cli can run (show help)
    log_info "Testing cusfm_cli functionality..."
    if cusfm_cli --help &> /dev/null; then
        log_success "cusfm_cli is working correctly"
    else
        log_warning "cusfm_cli command exists but may have runtime issues. Try running 'cusfm_cli --help' manually."
    fi
else
    exit_with_error "cusfm_cli command not found. Installation may have failed or PATH is not configured correctly."
fi

echo
echo "========================================"
log_success "PyCuSFM installation completed successfully!"
echo "========================================"
echo
log_info "You can now use PyCuSFM with the following command:"
echo "  cusfm_cli --input_dir <input_dir> --cusfm_base_dir <output_dir>"
echo
log_info "For more information, run:"
echo "  cusfm_cli --help"
echo
log_info "See the complete tutorial at: docs/tutorial.md"
echo
