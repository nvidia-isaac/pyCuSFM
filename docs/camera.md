# Camera Models

This document describes the camera projection models supported by cuSFM, including their mathematical formulations and parameter configurations in `frames_meta.json`.

## Table of Contents

- [Overview](#overview)
- [Supported Camera Models](#supported-camera-models)
  - [PINHOLE](#pinhole)
  - [DISTORTED_PINHOLE](#distorted_pinhole)
  - [OPENCV_FISHEYE](#opencv_fisheye)
  - [FTHETA_WINDSHIELD](#ftheta_windshield)
- [Rolling Shutter Correction](#rolling-shutter-correction)
- [Coordinate Systems](#coordinate-systems)
  - [Stereo Baseline Handling](#stereo-baseline-handling)

## Overview

cuSFM supports four camera projection models to accommodate different lens types and calibration requirements:

| Model Type | Description | COLMAP Equivalent |
|------------|-------------|-------------------|
| `PINHOLE` | Ideal pinhole camera (no distortion) | PINHOLE |
| `DISTORTED_PINHOLE` | Pinhole with lens distortion | FULL_OPENCV |
| `OPENCV_FISHEYE` | OpenCV fisheye lens model | OPENCV_FISHEYE |
| `FTHETA_WINDSHIELD` | Fisheye F-Theta with windshield refraction | Not supported |

---

## Supported Camera Models

### PINHOLE

The simplest camera model representing an ideal pinhole camera without lens distortion.

#### Mathematical Model

The projection from 3D camera coordinates $(X, Y, Z)$ to 2D pixel coordinates $(u, v)$ is:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =
\begin{bmatrix} f_x & 0 & c_x & 0 \\ 0 & f_y & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

Where:

- $(f_x, f_y)$: Focal lengths in pixel units
- $(c_x, c_y)$: Principal point (optical center) in pixel coordinates

After homogeneous division:

$$
u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y
$$

#### Parameters

| Parameter | Description |
|-----------|-------------|
| `fx` | Focal length in x direction (pixels) |
| `fy` | Focal length in y direction (pixels) |
| `cx` | Principal point x coordinate (pixels) |
| `cy` | Principal point y coordinate (pixels) |

#### Configuration in frames_meta.json

```json
{
  "camera_params_id_to_camera_params": {
    "0": {
      "sensor_meta_data": {
        "sensor_id": 0,
        "sensor_type": "CAMERA",
        "sensor_name": "front_stereo_camera_left",
        "frequency": 30,
        "sensor_to_vehicle_transform": {
          "axis_angle": {
            "x": 0.0,
            "y": 0.707107,
            "z": 0.0,
            "angle_degrees": 90.0
          },
          "translation": {
            "x": 1.5,
            "y": 0.0,
            "z": 1.2
          }
        }
      },
      "calibration_parameters": {
        "image_width": 1920,
        "image_height": 1200,
        "projection_matrix": {
          "data": [500.0, 0, 960.0, 0, 0, 500.0, 600.0, 0, 0, 0, 1, 0],
          "row_count": 3,
          "column_count": 4
        }
      },
      "camera_projection_model_type": "PINHOLE"
    }
  }
}
```

---

### DISTORTED_PINHOLE

An extended pinhole model that accounts for lens distortion using the OpenCV rational distortion model with 12 parameters.

#### Mathematical Model

**Step 1: Normalize to camera coordinates**

$$
x = \frac{X}{Z}, \quad y = \frac{Y}{Z}
$$

**Step 2: Apply distortion**

Compute radial distance:

$$
r^2 = x^2 + y^2, \quad r^4 = (r^2)^2, \quad r^6 = (r^2)^3
$$

Radial distortion:

$$
\text{radial} = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6
$$

Rational distortion:

$$
\text{rational} = 1 + k_4 r^2 + k_5 r^4 + k_6 r^6
$$

Combined radial-rational factor:

$$
\text{radial\\_rational} = \frac{\text{radial}}{\text{rational}}
$$

Tangential distortion:

$$
\Delta x_t = 2 p_1 x y + p_2 (r^2 + 2x^2)
$$

$$
\Delta y_t = p_1 (r^2 + 2y^2) + 2 p_2 x y
$$

**Step 3: Compute distorted coordinates**

$$
x' = x \cdot \text{radial\\_rational} + \Delta x_t
$$

$$
y' = y \cdot \text{radial\\_rational} + \Delta y_t
$$

**Step 4: Project to pixel coordinates**

$$
u = f_x \cdot x' + c_x, \quad v = f_y \cdot y' + c_y
$$

#### Parameters

cuSFM uses the **RATIONAL** model with 12 parameters (4 intrinsics + 8 distortion coefficients), which maps directly to COLMAP's FULL_OPENCV model:

| Parameter | Index | Description |
|-----------|-------|-------------|
| `fx` | 0 | Focal length x (pixels) |
| `fy` | 1 | Focal length y (pixels) |
| `cx` | 2 | Principal point x (pixels) |
| `cy` | 3 | Principal point y (pixels) |
| `k1` | 4 | Radial distortion coefficient 1 |
| `k2` | 5 | Radial distortion coefficient 2 |
| `p1` | 6 | Tangential distortion coefficient 1 |
| `p2` | 7 | Tangential distortion coefficient 2 |
| `k3` | 8 | Radial distortion coefficient 3 |
| `k4` | 9 | Radial distortion coefficient 4 |
| `k5` | 10 | Radial distortion coefficient 5 |
| `k6` | 11 | Radial distortion coefficient 6 |

#### Configuration in frames_meta.json

```json
{
  "camera_params_id_to_camera_params": {
    "0": {
      "sensor_meta_data": {
        "sensor_id": 0,
        "sensor_type": "CAMERA",
        "sensor_name": "front_camera",
        "sensor_to_vehicle_transform": {
          "axis_angle": {
            "x": 0.0,
            "y": 0.707107,
            "z": 0.0,
            "angle_degrees": 90.0
          },
          "translation": {
            "x": 1.5,
            "y": 0.0,
            "z": 1.2
          }
        }
      },
      "calibration_parameters": {
        "image_width": 1920,
        "image_height": 1080,
        "camera_matrix": {
          "data": [500.0, 0, 960.0, 0, 500.0, 540.0, 0, 0, 1],
          "row_count": 3,
          "column_count": 3
        },
        "distortion_coefficients": {
          "data": [0.1, -0.2, 0.001, 0.002, 0.05, 0.01, -0.01, 0.005],
          "row_count": 1,
          "column_count": 8
        }
      },
      "camera_projection_model_type": "DISTORTED_PINHOLE"
    }
  }
}
```

---

### OPENCV_FISHEYE

An OpenCV-compatible fisheye camera model using the equidistant projection with polynomial distortion. This model is suitable for wide-angle fisheye lenses and follows the same conventions as OpenCV's `cv::fisheye` module.

#### Mathematical Model

**Step 1: Normalize to camera coordinates**

$$
x = \frac{X}{Z}, \quad y = \frac{Y}{Z}
$$

**Step 2: Compute incidence angle**

Compute the radial distance and incidence angle:

$$
r = \sqrt{x^2 + y^2}
$$

$$
\theta = \arctan(r)
$$

**Step 3: Apply fisheye distortion**

The distorted angle is computed using a polynomial:

$$
\theta_d = \theta \cdot (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8)
$$

**Step 4: Compute distorted normalized coordinates**

Scale the normalized coordinates by the distortion factor:

$$
\text{scale} = \frac{\theta_d}{r}
$$

$$
x_d = \text{scale} \cdot x, \quad y_d = \text{scale} \cdot y
$$

Note: When $r < 10^{-8}$ (point at the optical center), $x_d = x$ and $y_d = y$.

**Step 5: Project to pixel coordinates**

$$
u = f_x \cdot x_d + c_x, \quad v = f_y \cdot y_d + c_y
$$

#### Back-Projection (Undistortion)

The back-projection from pixel coordinates to 3D ray uses Newton-Raphson iteration to invert the distortion polynomial:

1. Convert pixel to normalized distorted coordinates:

$$
x_d = \frac{u - c_x}{f_x}, \quad y_d = \frac{v - c_y}{f_y}
$$

2. Compute distorted radius:

$$
\theta_d = \sqrt{x_d^2 + y_d^2}
$$

3. Solve for $\theta$ using Newton-Raphson iteration on:

$$
f(\theta) = \theta \cdot (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8) - \theta_d = 0
$$

4. Compute undistorted coordinates:

$$
r = \tan(\theta), \quad \text{scale} = \frac{r}{\theta_d}
$$

$$
x = \text{scale} \cdot x_d, \quad y = \text{scale} \cdot y_d
$$

5. Return ray direction $(x, y, 1)$.

#### Parameters

The OpenCV fisheye model uses 8 parameters (4 intrinsics + 4 distortion coefficients):

| Parameter | Index | Description |
|-----------|-------|-------------|
| `fx` | 0 | Focal length x (pixels) |
| `fy` | 1 | Focal length y (pixels) |
| `cx` | 2 | Principal point x (pixels) |
| `cy` | 3 | Principal point y (pixels) |
| `k1` | 4 | Fisheye distortion coefficient 1 |
| `k2` | 5 | Fisheye distortion coefficient 2 |
| `k3` | 6 | Fisheye distortion coefficient 3 |
| `k4` | 7 | Fisheye distortion coefficient 4 |

#### COLMAP Compatibility

This model maps directly to COLMAP's `OPENCV_FISHEYE` model (model_id=9) with 8 parameters in the same order: `fx, fy, cx, cy, k1, k2, k3, k4`.

#### Configuration in frames_meta.json

```json
{
  "camera_params_id_to_camera_params": {
    "0": {
      "sensor_meta_data": {
        "sensor_id": 0,
        "sensor_type": "CAMERA",
        "sensor_name": "front_fisheye_camera",
        "sensor_to_vehicle_transform": {
          "axis_angle": {
            "x": 0.0,
            "y": 0.707107,
            "z": 0.0,
            "angle_degrees": 90.0
          },
          "translation": {
            "x": 1.5,
            "y": 0.0,
            "z": 1.2
          }
        }
      },
      "calibration_parameters": {
        "image_width": 1920,
        "image_height": 1200,
        "camera_matrix": {
          "data": [500.0, 0, 960.0, 0, 500.0, 600.0, 0, 0, 1],
          "row_count": 3,
          "column_count": 3
        },
        "distortion_coefficients": {
          "data": [-0.02, 0.01, -0.005, 0.001],
          "row_count": 1,
          "column_count": 4
        }
      },
      "camera_projection_model_type": "OPENCV_FISHEYE"
    }
  }
}
```

---

### FTHETA_WINDSHIELD

A specialized fisheye camera model using F-Theta projection with optional windshield refraction correction. This model is designed for NVIDIA Hyperion cameras.

**Note**: This model is not supported by COLMAP export. When exporting to COLMAP format, it falls back to PINHOLE using the projection matrix if available.

#### F-Theta Camera Model

**Mathematical Model (Ray2Pixel):**

For a 3D ray $(X, Y, Z)$ in camera coordinates:

1. Compute angle from optical axis:

$$
\theta = \arctan\left(\frac{\sqrt{X^2 + Y^2}}{Z}\right)
$$

2. Compute azimuth angle:

$$
\psi = \arctan2(Y, X)
$$

3. Apply forward polynomial to get distorted radius:

$$
r_d = f(\theta) = k_0 + k_1 \theta + k_2 \theta^2 + k_3 \theta^3 + k_4 \theta^4 + k_5 \theta^5
$$

   Note: When using backward polynomial as reference, Newton's method is applied to compute $r_d$ from the backward polynomial.

4. Convert to distorted normalized coordinates:

$$
d_x = r_d \cos(\psi), \quad d_y = r_d \sin(\psi)
$$

5. Apply linear transform:

$$
\begin{bmatrix} d'_x \\ d'_y \end{bmatrix} =
\begin{bmatrix} c & d \\ e & 1 \end{bmatrix}
\begin{bmatrix} d_x \\ d_y \end{bmatrix}
$$

6. Add principal point offset:

$$
u = d'_x + \text{ppx}, \quad v = d'_y + \text{ppy}
$$

**Backward Polynomial:**

The backward polynomial maps from distorted image radius $r$ to angle $\theta$:

$$
b(r) = j_0 + j_1 r + j_2 r^2 + j_3 r^3 + j_4 r^4 + j_5 r^5
$$

Where:
- $j_0$ must be zero
- $j_1$ represents the inverse of focal length

**F-Theta Parameters:**

The F-Theta model uses 17 parameters:

| Parameter | Index | Description |
|-----------|-------|-------------|
| `ppx` | 0 | Principal point x |
| `ppy` | 1 | Principal point y |
| `c` | 2 | Linear transform element (0,0) |
| `d` | 3 | Linear transform element (0,1) |
| `e` | 4 | Linear transform element (1,0) |
| `bw_0` | 5 | Backward polynomial coefficient 0 (must be 0) |
| `bw_1` | 6 | Backward polynomial coefficient 1 (1/focal_length) |
| `bw_2` | 7 | Backward polynomial coefficient 2 |
| `bw_3` | 8 | Backward polynomial coefficient 3 |
| `bw_4` | 9 | Backward polynomial coefficient 4 |
| `bw_5` | 10 | Backward polynomial coefficient 5 |
| `fw_0` | 11 | Forward polynomial coefficient 0 (must be 0) |
| `fw_1` | 12 | Forward polynomial coefficient 1 (focal_length) |
| `fw_2` | 13 | Forward polynomial coefficient 2 |
| `fw_3` | 14 | Forward polynomial coefficient 3 |
| `fw_4` | 15 | Forward polynomial coefficient 4 |
| `fw_5` | 16 | Forward polynomial coefficient 5 |

#### Windshield Model

The windshield model compensates for optical distortion caused by the vehicle's windshield glass. It is used in conjunction with the F-Theta camera model for cameras mounted behind windshields.

**Mathematical Model:**

The windshield is modeled as a view-dependent ray mapping that transforms the original ray direction to a refracted ray direction.

**Projection (Out2In Ray Mapping):**

For a 3D ray $(X, Y, Z)$ in camera coordinates:

1. Normalize the ray:

$$
\hat{X} = \frac{X}{\|r\|}, \quad \hat{Y} = \frac{Y}{\|r\|}, \quad \hat{Z} = \frac{Z}{\|r\|}
$$

2. Compute horizontal and vertical angles:

$$
\phi = \arcsin(\hat{X})
$$

$$
\theta = \arcsin(\hat{Y})
$$

3. Apply bivariate polynomials to compute refracted angles:

$$
\phi' = P_\phi(\phi, \theta) = \sum_{i=0}^{n} \sum_{j=0}^{n-i} a_{ij} \phi^i \theta^j
$$

$$
\theta' = P_\theta(\phi, \theta) = \sum_{i=0}^{m} \sum_{j=0}^{m-i} b_{ij} \phi^i \theta^j
$$

4. Convert back to ray direction:

$$
X' = \sin(\phi'), \quad Y' = \sin(\theta'), \quad Z' = \sqrt{1 - X'^2 - Y'^2}
$$

**Back-Projection (In2Out Ray Mapping):**

Uses Newton's method to iteratively solve for the original ray given the refracted ray.

**Windshield Parameters:**

The windshield model uses bivariate polynomials with configurable degrees:

| Parameter | Description |
|-----------|-------------|
| `phi_poly_degree` | Degree of the horizontal (phi) polynomial (default: 2) |
| `theta_poly_degree` | Degree of the vertical (theta) polynomial (default: 4) |
| `phi_poly_coefficients` | Coefficients for phi polynomial (6 coefficients for degree 2) |
| `theta_poly_coefficients` | Coefficients for theta polynomial (15 coefficients for degree 4) |

The number of coefficients for a 2D polynomial of degree $n$ is:

$$
\frac{(n+1)(n+2)}{2}
$$

#### Configuration in frames_meta.json

```json
{
  "camera_params_id_to_camera_params": {
    "0": {
      "sensor_meta_data": {
        "sensor_id": 0,
        "sensor_type": "CAMERA",
        "sensor_name": "front_fisheye_camera",
        "sensor_to_vehicle_transform": {
          "axis_angle": {
            "x": 0.0,
            "y": 0.707107,
            "z": 0.0,
            "angle_degrees": 90.0
          },
          "translation": {
            "x": 1.5,
            "y": 0.0,
            "z": 1.2
          }
        }
      },
      "calibration_parameters": {
        "image_width": 1920,
        "image_height": 1200,
        "ftheta_parameters": {
          "principal_point_x": 960.0,
          "principal_point_y": 600.0,
          "linear_transform_c": 1.0,
          "linear_transform_d": 0.0,
          "linear_transform_e": 0.0,
          "poly_type": "BACKWARD_POLY_TYPE",
          "backward_poly_coefficients": [0, 0.002, 0.0001, -0.00001, 0.0, 0.0],
          "forward_poly_coefficients": [0, 500.0, 10.0, 1.0, 0.0, 0.0]
        },
        "windshield_parameters": {
          "phi_poly_degree": 2,
          "theta_poly_degree": 4,
          "phi_poly_coefficients": [0.001, 0.99, 0.0001, 0.0, 0.0, 0.0],
          "theta_poly_coefficients": [0.002, 0.001, 0.98, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
      },
      "camera_projection_model_type": "FTHETA_WINDSHIELD"
    }
  }
}
```

---

## Rolling Shutter Correction

### Overview

Rolling shutter cameras expose different rows of the sensor at different times. For a moving camera, this causes geometric distortions where different parts of the image correspond to different camera poses.

### Configuration

Rolling shutter correction is enabled via:
- Command line: `--use_rsc` flag in `cusfm_cli`
- Configuration: `do_rolling_shutter_correction` in bundle adjustment config

### Input Requirements in frames_meta.json

To enable rolling shutter correction, the following fields must be configured in `frames_meta.json`:

```json
{
  "camera_params_id_to_camera_params": {
    "0": {
      "calibration_parameters": {
        "rolling_shutter_delay_microseconds": 33000
      }
    }
  },
  "vehicle_trajectory_file": "trajectory.json"
}
```

| Field | Description |
|-------|-------------|
| `rolling_shutter_delay_microseconds` | Total readout time from first row to last row, in microseconds. Set to 0 for global shutter cameras. |
| `vehicle_trajectory_file` | Path to trajectory data file containing vehicle poses for pose interpolation. |

### Vehicle Trajectory Data

The vehicle trajectory provides the pose history needed to interpolate camera poses for each row's capture time.

Each trajectory file contains a `VehicleTrajectory` message with the following structure:

| Field | Type | Description |
|-------|------|-------------|
| `track_name` | string | Name of the track this trajectory belongs to |
| `session_name` | string | Session identifier (a session may contain multiple tracks) |
| `trajectory_type` | enum | Type of trajectory data (see below) |
| `timestamp_microseconds` | repeated uint64 | Timestamps for each pose (must match pose count) |
| `poses` | repeated RigidTransform3d | Vehicle poses at each timestamp |

Example `trajectory.json`:

```json
{
  "track_name": "track_001",
  "session_name": "session_2024_01_15",
  "trajectory_type": "EGO_MOTION",
  "timestamp_microseconds": [1000000000000, 1000000010000, 1000000020000],
  "poses": [
    {
      "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
      "translation": {"x": 0.0, "y": 0.0, "z": 0.0}
    },
    {
      "rotation": {"w": 0.9999, "x": 0.01, "y": 0.0, "z": 0.0},
      "translation": {"x": 0.5, "y": 0.0, "z": 0.0}
    },
    {
      "rotation": {"w": 0.9998, "x": 0.02, "y": 0.0, "z": 0.0},
      "translation": {"x": 1.0, "y": 0.0, "z": 0.0}
    }
  ]
}
```

#### Trajectory Types

| Type | Description | Use Case |
|------|-------------|----------|
| `EGO_MOTION` | Poses relative to the first frame of the track | IMU-derived motion, visual odometry |
| `ABSOLUTE_POSE` | Full 6-DoF poses in world/ENU coordinates | GPS+IMU fusion with orientation |
| `ABSOLUTE_POSITION` | Position-only observations (no orientation) | GPS measurements only |

For rolling shutter correction, `EGO_MOTION` or `ABSOLUTE_POSE` trajectories are typically used, as they provide the high-frequency pose updates needed for accurate per-row interpolation.

### Mathematical Model

For a frame with `num_rows` rows and total readout time `T`:
- The last row is captured at the reference timestamp `t_ref`
- Row `r` is captured at time:

$$
t(r) = t_{ref} - T \cdot \frac{(\text{num\\_rows} - 1 - r)}{\text{num\\_rows} - 1}
$$

The correction computes a per-row rigid transformation $T_r$ that maps points from the reference camera frame to the camera frame at row $r$'s capture time.

#### Pose Interpolation

During rolling shutter correction, the system:
1. Looks up the trajectory data for the current track
2. For each image row, computes the capture time using a fixed frame capture time (currently 32.5ms)
3. Interpolates the vehicle pose at that exact timestamp from the trajectory
4. Transforms the interpolated pose to the camera frame using `camera_to_vehicle` transform
5. Applies the per-row correction to feature observations

**Note**: The `timestamp_microseconds` array in the trajectory should have sufficient temporal resolution (typically from IMU data at 100-400 Hz) to enable accurate interpolation between samples.

### Output: start_camera_to_world

When rolling shutter correction is enabled, the output `frames_meta.json` will include an additional pose field in `KeyframeMetaData`:

| Field | Description |
|-------|-------------|
| `start_camera_to_world` | Camera pose at the **first row** (start of exposure) |

This field, combined with the standard `camera_to_world` (pose at the last row), allows reconstruction of camera motion during the frame's exposure time.

---

## Coordinate Systems

### Camera Coordinate System

cuSFM uses a right-handed camera coordinate system following the [OpenCV convention](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html):
- **X-axis**: Points right (positive X = right in the image)
- **Y-axis**: Points down (positive Y = down in the image)
- **Z-axis**: Points forward (optical axis, positive Z = in front of camera)

This is also known as the **right-down-forward (RDF)** convention.

```
        Z (forward/optical axis)
        ^
       /
      /
     +-------> X (right)
     |
     |
     v
     Y (down)
```

### Image Coordinate System

- **Origin**: Top-left corner of the image
- **u-axis**: Points right (column index, corresponds to camera X)
- **v-axis**: Points down (row index, corresponds to camera Y)

The relationship between camera coordinates and image coordinates:
- Camera X positive → image u increases (move right in image)
- Camera Y positive → image v increases (move down in image)
- Camera Z positive → object is in front of camera (visible)

### Vehicle Coordinate System

The vehicle coordinate system uses a right-handed **forward-left-up (FLU)** convention:
- **X-axis**: Points forward (vehicle front)
- **Y-axis**: Points left
- **Z-axis**: Points up

### World Coordinate System

The world coordinate system is usually defined as the first frame of the vehicle coordinate (FLU convention).

### Stereo Baseline Handling

In cuSFM, stereo baseline is computed at runtime from the `sensor_to_vehicle_transform` of each camera.

**Baseline Calculation:**

```
baseline = ComputeBaseline(left_camera, right_camera):
    # Get sensor-to-vehicle transforms
    T_left  = left_camera.sensor_to_vehicle_transform   # 4x4 SE3 matrix
    T_right = right_camera.sensor_to_vehicle_transform  # 4x4 SE3 matrix

    # Compute relative transform: left_camera → right_camera
    T_left_to_right = inverse(T_right) * T_left

    # Baseline is the X-component of translation (in camera frame)
    return -T_left_to_right.translation.x
```
