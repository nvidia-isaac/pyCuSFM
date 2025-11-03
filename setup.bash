#!/bin/bash

# Setup script to create symlinks for CUDA lib and bin folders
# Usage: ./setup.bash cuda12|cuda13

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYCUSFM_DIR="$SCRIPT_DIR/pycusfm"

# Function to display usage
usage() {
    echo "Usage: $0 <cuda_version>"
    echo "  cuda_version: cuda12 or cuda13"
    echo ""
    echo "Example:"
    echo "  $0 cuda12    # Create symlinks for CUDA 12"
    echo "  $0 cuda13    # Create symlinks for CUDA 13"
    exit 1
}

# Function to remove existing symlinks
cleanup_symlinks() {
    echo "Cleaning up existing symlinks..."
    if [ -L "$PYCUSFM_DIR/bin" ]; then
        rm "$PYCUSFM_DIR/bin"
        echo "  Removed existing bin symlink"
    fi
    if [ -L "$PYCUSFM_DIR/lib" ]; then
        rm "$PYCUSFM_DIR/lib"
        echo "  Removed existing lib symlink"
    fi
}

# Function to create symlinks
create_symlinks() {
    local cuda_version=$1
    local source_dir="$PYCUSFM_DIR/x86_$cuda_version"

    echo "Creating symlinks for $cuda_version..."

    # Check if source directories exist
    if [ ! -d "$source_dir/bin" ]; then
        echo "Error: $source_dir/bin does not exist"
        exit 1
    fi

    if [ ! -d "$source_dir/lib" ]; then
        echo "Error: $source_dir/lib does not exist"
        exit 1
    fi

    # Create symlinks
    ln -sf "x86_$cuda_version/bin" "$PYCUSFM_DIR/bin"
    ln -sf "x86_$cuda_version/lib" "$PYCUSFM_DIR/lib"

    echo "  Created bin symlink: $PYCUSFM_DIR/bin -> x86_$cuda_version/bin"
    echo "  Created lib symlink: $PYCUSFM_DIR/lib -> x86_$cuda_version/lib"
}

# Main script
main() {
    # Check if argument is provided
    if [ $# -ne 1 ]; then
        echo "Error: Missing CUDA version argument"
        usage
    fi

    local cuda_version=$1

    # Validate CUDA version
    case $cuda_version in
        cuda12|cuda13)
            ;;
        *)
            echo "Error: Invalid CUDA version '$cuda_version'"
            echo "Supported versions: cuda12, cuda13"
            usage
            ;;
    esac

    # Check if pycusfm directory exists
    if [ ! -d "$PYCUSFM_DIR" ]; then
        echo "Error: pycusfm directory not found at $PYCUSFM_DIR"
        exit 1
    fi

    echo "Setting up CUDA $cuda_version environment..."
    echo "Working directory: $PYCUSFM_DIR"

    # Clean up existing symlinks
    cleanup_symlinks

    # Create new symlinks
    create_symlinks "$cuda_version"

    echo ""
    echo "Setup completed successfully!"
    echo "CUDA $cuda_version lib and bin directories are now linked in pycusfm/"

    # Display current symlinks
    echo ""
    echo "Current symlinks:"
    ls -la "$PYCUSFM_DIR" | grep -E "(bin|lib) ->"
}

# Run main function with all arguments
main "$@"
