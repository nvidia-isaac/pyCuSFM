# SuperPoint and LightGlue Models README

This README provides instructions for setting up and using the SuperPoint and LightGlue models, including necessary configurations and file details.

## TensorRT Parameters Setup

To use the SuperPoint and LightGlue models effectively, you need to configure `TensorRTParams`:

1. **SuperPoint**: Refer to the configuration file located at `keypoint_creation_config.pb.txt`.
2. **LightGlue**: Refer to the configuration file located at `matching_config.pb.txt`.

## ONNX Files

The `.onnx` files are converted from original computation graphs using TensorRT tools:

- **superpoint_net_free.onnx**: Derived from [SuperPoint](https://github.com/rpautrat/SuperPoint/blob/master/weights/superpoint_v6_from_tf.pth) under MIT license.
- **superpoint_post.onnx**: Custom implementation by jingruiy@nvidia.com for SuperPoint post-processing, without using network model parameters.
- **lightglue.onnx**: Created from the [LightGlue-ONNX repository](https://github.com/fabio-sim/LightGlue-ONNX).

Note that the provided SuperPoint model is a free commercial version that was not jointly trained with the LightGlue model, which may impact matching performance.

