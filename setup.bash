#!/bin/bash

# Setup script to create symlinks for platform-specific lib and bin folders
# Usage: ./setup.bash <platform>
#
# Supported platforms:
#   x86_64:  cuda12, cuda13
#   Jetson:  jp6 (Orin, Jetpack 6, CUDA 12)
#            jp7 (Thor, Jetpack 7, CUDA 13)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYCUSFM_DIR="$SCRIPT_DIR/pycusfm"

# Function to display usage
usage() {
    echo "Usage: $0 <platform>"
    echo ""
    echo "For x86_64 systems:"
    echo "  $0 cuda12              # x86_64 with CUDA 12"
    echo "  $0 cuda13              # x86_64 with CUDA 13"
    echo ""
    echo "For Jetson systems:"
    echo "  $0 jp6                 # Orin (Jetpack 6, CUDA 12)"
    echo "  $0 jp7                 # Thor (Jetpack 7, CUDA 13)"
    echo ""
    echo "Auto-detect:"
    echo "  $0 auto                # Auto-detect platform"
    exit 1
}

# Function to auto-detect platform
auto_detect_platform() {
    local arch=$(uname -m)
    
    if [ "$arch" = "aarch64" ]; then
        # On ARM, check Jetpack version from /etc/nv_tegra_release
        if [ -f "/etc/nv_tegra_release" ]; then
            local r_version=$(grep -oP '# R\K\d+' /etc/nv_tegra_release 2>/dev/null || echo "0")
            if [ "$r_version" -ge 38 ]; then
                echo "jp7"  # JP7 (Thor)
                return 0
            elif [ "$r_version" -ge 36 ]; then
                echo "jp6"  # JP6 (Orin)
                return 0
            fi
        fi
        echo ""  # Unknown Jetson
    else
        # On x86_64, check CUDA version
        local cuda_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K\d+' || echo "")
        if [ "$cuda_version" = "13" ]; then
            echo "cuda13"
        elif [ "$cuda_version" = "12" ]; then
            echo "cuda12"
        else
            echo ""  # Unknown
        fi
    fi
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

# Function to get source directory from config
get_source_dir() {
    local config=$1
    case $config in
        cuda12)
            echo "x86_cuda12"
            ;;
        cuda13)
            echo "x86_cuda13"
            ;;
        jp6|jp7)
            echo "$config"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to create symlinks
create_symlinks() {
    local config=$1
    local source_subdir=$(get_source_dir "$config")
    local source_dir="$PYCUSFM_DIR/$source_subdir"

    echo "Creating symlinks for $config..."
    echo "  Source directory: $source_subdir"

    # Check if source directories exist
    if [ ! -d "$source_dir/bin" ]; then
        echo "Error: $source_dir/bin does not exist"
        echo ""
        echo "Available platform directories:"
        ls -d "$PYCUSFM_DIR"/x86_* "$PYCUSFM_DIR"/jp* 2>/dev/null | xargs -n1 basename || echo "  None found"
        exit 1
    fi

    if [ ! -d "$source_dir/lib" ]; then
        echo "Error: $source_dir/lib does not exist"
        exit 1
    fi

    # Create symlinks
    ln -sf "$source_subdir/bin" "$PYCUSFM_DIR/bin"
    ln -sf "$source_subdir/lib" "$PYCUSFM_DIR/lib"

    echo "  Created bin symlink: $PYCUSFM_DIR/bin -> $source_subdir/bin"
    echo "  Created lib symlink: $PYCUSFM_DIR/lib -> $source_subdir/lib"
}

# Main script
main() {
    # Check if argument is provided
    if [ $# -ne 1 ]; then
        echo "Error: Missing platform argument"
        usage
    fi

    local config=$1

    # Handle auto-detection
    if [ "$config" = "auto" ]; then
        echo "Auto-detecting platform..."
        config=$(auto_detect_platform)
        if [ -z "$config" ]; then
            echo "Error: Could not auto-detect platform"
            echo "Please specify manually."
            usage
        fi
        echo "Detected: $config"
    fi

    # Validate configuration
    case $config in
        cuda12|cuda13|jp6|jp7)
            ;;
        *)
            echo "Error: Invalid platform '$config'"
            usage
            ;;
    esac

    # Check if pycusfm directory exists
    if [ ! -d "$PYCUSFM_DIR" ]; then
        echo "Error: pycusfm directory not found at $PYCUSFM_DIR"
        exit 1
    fi

    echo "Setting up $config environment..."
    echo "Working directory: $PYCUSFM_DIR"

    # Clean up existing symlinks
    cleanup_symlinks

    # Create new symlinks
    create_symlinks "$config"

    echo ""
    echo "Setup completed successfully!"

    # Display current symlinks
    echo ""
    echo "Current symlinks:"
    ls -la "$PYCUSFM_DIR" | grep -E "(bin|lib) ->"
    
    # Print platform info
    echo ""
    case $config in
        cuda12)
            echo "Platform: x86_64 CUDA 12"
            ;;
        cuda13)
            echo "Platform: x86_64 CUDA 13"
            ;;
        jp6)
            echo "Platform: Orin (Jetpack 6, CUDA 12)"
            ;;
        jp7)
            echo "Platform: Thor (Jetpack 7, CUDA 13)"
            ;;
    esac
}

# Run main function with all arguments
main "$@"
