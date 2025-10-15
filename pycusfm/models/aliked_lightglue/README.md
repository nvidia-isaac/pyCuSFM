# ALIKED and LightGlue Models README

This README provides instructions for setting up and using the ALIKED and LightGlue models

## TensorRT Parameters Setup

To use the ALIKED and LightGlue models effectively, you need to configure the following:

1. **ALIKED**: Refer to the configuration file located at `keypoint_creation_config.pb.txt`, section aliked_detector
2. **LightGlue**: Refer to the configuration file located at `matching_task_worker_config.pb.txt`, section  lightglue_params/aliked_tensorrt_config

## ONNX Files

The `.onnx` files are converted from computation graphs using TensorRT tools:

- **aliked.onnx**: Created from [ALIKED](https://github.com/ajuric/aliked-tensorrt) under BSD-3 Clause license.
- **lightglue_aliked.onnx**: Derived from the [LightGlue-ONNX repository](https://github.com/fabio-sim/LightGlue-ONNX), but we modify the repo and add support for ALIKED feature descriptor.

Ensure you have the necessary environment set up to utilize these models effectively. For any issues or further inquiries, please refer to the respective repositories or contact the maintainers.