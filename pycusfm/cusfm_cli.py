# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from .cusfm_runner import create_cusfm_runner


def parse_bool(value):
    """Convert various string inputs to boolean or None.

    Args:
        value: Input value that could be string, int, bool or None

    Returns:
        bool or None: Converted boolean value or None if input is None

    Handles inputs:
        - None -> None
        - bool -> bool
        - int -> bool (0 -> False, non-zero -> True)
        - str -> bool ('true', 't', 'yes', 'y', '1' -> True)
                     ('false', 'f', 'no', 'n', '0' -> False)
                     (case insensitive)
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', 't', 'yes', 'y', '1'):
            return True
        if value in ('false', 'f', 'no', 'n', '0'):
            return False
    raise ValueError(f'Cannot convert {value} to boolean')


def print_params(params, indent=4):
    """Print parameters in a pretty format.

    Args:
        params: Dictionary of parameters
        indent: Number of spaces for indentation
    """
    print("Parameters:")
    for key, value in sorted(params.items()):
        print(f"{' ' * indent}{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='CUSFM Feature Extraction and Mapping')

    # Create a function to handle both flag-style and value-style boolean args
    def add_bool_argument(parser, name, default=None, help=None):
        parser.add_argument(
            f'--{name}',
            nargs='?',  # Makes the value optional
            const=True,  # Used when flag is present without value
            type=parse_bool,  # Used when value is provided
            default=default,
            help=help)

    parser.add_argument(
        '--input_dir',
        default=None,
        help='Input directory, required when feature_extractor is not skipped')
    parser.add_argument(
        '--cusfm_base_dir', required=True, help='CUSFM base directory')
    parser.add_argument('--binary_dir', default=None, help='Binary directory')
    parser.add_argument('--config_dir', default=None, help="Config directory")
    parser.add_argument('--feature_type', default=None, help='Feature type')
    parser.add_argument('--mask_dir', default=None, help='Mask directory')
    parser.add_argument('--model_dir', default=None, help='Model directory for feature extraction')
    parser.add_argument(
        '--steps_to_run',
        nargs='+',
        choices=[
            'feature_extractor', 'vocab_generator', 'pose_graph', 'matcher',
            'mapper', 'map_convertor'
        ],
        help='Specify which steps to run.')

    # Replace boolean arguments with the new style
    add_bool_argument(
        parser,
        'skip_cuvslam',
        help='Skip cuvslam when steps_to_run is not specified')
    add_bool_argument(
        parser,
        'skip_feature_extractor',
        help='Skip feature extractor when steps_to_run is not specified')
    add_bool_argument(
        parser,
        'skip_vocab_generator',
        help='Skip vocabulary generator when steps_to_run is not specified.')
    add_bool_argument(
        parser,
        'skip_pose_graph',
        help='Skip pose graph when steps_to_run is not specified')
    add_bool_argument(
        parser,
        'skip_matcher',
        help='Skip matcher when steps_to_run is not specified')
    add_bool_argument(
        parser,
        'skip_mapper',
        help='Skip mapper when steps_to_run is not specified')
    add_bool_argument(
        parser,
        'skip_map_convertor',
        help='Skip map convertor when steps_to_run is not specified')
    add_bool_argument(parser, 'enable_debug', help='Enable debug')
    add_bool_argument(parser, 'dry_run', help='Dry run mode')
    add_bool_argument(
        parser, 'optimize_extrinsics', help='Actually optimize extrinsics')
    add_bool_argument(
        parser, 'optimize_intrinsics', help='Actually optimize intrinsics')
    add_bool_argument(
        parser, 'multi_track_input', help='If input is multi-track')
    add_bool_argument(
        parser,
        'downsampling_matches',
        help='If downsample image pairs when building match tasks')
    add_bool_argument(
        parser, 'use_rsc', help='use rolling shutter correction when mapping')
    add_bool_argument(
        parser,
        'av_data',
        help='Use AV data to skip frame selection or sampling')
    add_bool_argument(
        parser,
        'use_vehicle_trajectory',
        help='Use vehicle trajectory for constraints')
    add_bool_argument(
        parser,
        'add_tensorrt_path',
        help='Whether to add TensorRT path to LD_LIBRARY_PATH')
    add_bool_argument(
        parser,
        'use_cuvslam_slam_pose',
        help='Whether to use cuvslam slam pose or odom pose')
    add_bool_argument(
        parser,
        'skip_track_global_transform',
        help='Whether to skip track global transformation or not')
    add_bool_argument(
        parser,
        'skip_data_association',
        help='Whether to use data association or not')
    parser.add_argument(
        "--ba_frame_type",
        type=str,
        default='config',
        choices=["config", "camera_frame", "vehicle_frame", "vehicle_rig"],
        help="The frame mode to use, valid options are: "
        "config, camera_frame, vehicle_frame, vehicle_rig. "
        "If set to config, we will use the ba_frame_type provided in the config file."
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        help='Number of available threads')
    parser.add_argument(
        '--override_frames_meta_file',
        default=None,
        help='Override frames meta file')
    parser.add_argument(
        '--cuvgl_dir',
        default=None,
        help='override default path for bow vocab')
    parser.add_argument(
        '--min_inter_frame_distance',
        type=float,
        default=None,
        help='Translation distance.')
    parser.add_argument(
        '--min_inter_frame_rotation_degrees',
        type=float,
        default=None,
        help='Rotation distance.')
    parser.add_argument(
        '--sample_sync_threshold_microseconds',
        type=int,
        help=(
            'Optional. Specifies the allowable timestamp difference (in microseconds)'
            'for the same sample. Suggested value: 100.'))
    parser.add_argument(
        '--stereo_pair_non_baseline_max_distance',
        type=float,
        help=(
            'Optional. Specifies the maximum translation distance between'
            'one stereo pair cameras non-baseline for the stereo camera.'
            ' If specified >0, we will compute stereo pairs based on this distance'
            ' if one pair of cameras has y, z translation distance less than this value.'
        ))
    parser.add_argument(
        '--debug_interval',
        type=int,
        default=None,
        help='Optional, The interval of debug data')
    parser.add_argument(
        '--feature_matching_batch_size',
        type=int,
        help=
        'Batch size for GPU-accelerated feature matching. If > 0, uses GPU batch matching'
    )
    parser.add_argument(
        '--max_keypoints_num',
        type=int,
        help='Optional, the max number of keypoints for feature extractor, if '
        'set to -1, it will use the max number of keypoints in the '
        'keypoint creation config')
    parser.add_argument(
        '--feature_extractor_batch_size',
        type=int,
        help='Batch size for feature extraction')
    parser.add_argument(
        '--anchor_track',
        type=str,
        help='Specify the name of the folder for the track(s) to be fixed in multi-track scenarios.')
    parser.add_argument(
        '--global_localize_succ_sample',
        type=int,
        help=(
            'Optional. Specifies sample number for successful global localization to transform track coordinates in multi_track_input'
        ))
    parser.add_argument(
        "--previous_cusfm_ws",
        type=str,
        help="If built map folder is provided, using patch mode.")
    parser.add_argument(
        "--previous_raw_image_dir",
        type=str,
        help="If built map raw image folder is provided, using debug mode.")

    args = parser.parse_args()

    if args.steps_to_run:
        step_to_skip_map = {
            'cuvslam': 'skip_cuvslam',
            'feature_extractor': 'skip_feature_extractor',
            'vocab_generator': 'skip_vocab_generator',
            'pose_graph': 'skip_pose_graph',
            'matcher': 'skip_matcher',
            'mapper': 'skip_mapper',
            'map_convertor': 'skip_map_convertor'
        }

        for step, skip_arg in step_to_skip_map.items():
            if step not in args.steps_to_run:
                setattr(args, skip_arg, True)

    # if optimize extrinsics is set, and user didn't specify frame mode,
    # set it to vehicle_rig
    if args.optimize_extrinsics and (args.ba_frame_type != "vehicle_rig"):
        raise ValueError(
            "--optimize_extrinsics is only supported for --ba_frame_type=vehicle_rig"
        )

    # Create runner using the new helper function
    cusfm_runner = create_cusfm_runner(
        av_data=args.av_data,
        input_dir=args.input_dir,
        cusfm_base_dir=args.cusfm_base_dir,
        binary_dir=args.binary_dir,
        config_dir=args.config_dir,
        mask_dir=args.mask_dir,
        feature_type=args.feature_type,
        model_dir=args.model_dir,
        num_threads=args.num_threads,
        override_frames_meta_file=args.override_frames_meta_file,
        cuvgl_dir=args.cuvgl_dir,
        min_inter_frame_distance=args.min_inter_frame_distance,
        min_inter_frame_rotation_degrees=args.min_inter_frame_rotation_degrees,
        sample_sync_threshold_microseconds=args.
        sample_sync_threshold_microseconds,
        stereo_pair_non_baseline_max_distance=args.
        stereo_pair_non_baseline_max_distance,
        skip_cuvslam=args.skip_cuvslam,
        skip_feature_extractor=args.skip_feature_extractor,
        skip_vocab_generator=args.skip_vocab_generator,
        skip_pose_graph=args.skip_pose_graph,
        skip_matcher=args.skip_matcher,
        skip_mapper=args.skip_mapper,
        skip_map_convertor=args.skip_map_convertor,
        enable_debug=args.enable_debug,
        dry_run=args.dry_run,
        ba_frame_type=args.ba_frame_type,
        optimize_extrinsics=args.optimize_extrinsics,
        optimize_intrinsics=args.optimize_intrinsics,
        multi_track_input=args.multi_track_input,
        use_roll_shutter_correction=args.use_rsc,
        downsampling_matches=args.downsampling_matches,
        use_vehicle_trajectory=args.use_vehicle_trajectory,
        debug_interval=args.debug_interval,
        max_keypoints_num=args.max_keypoints_num,
        feature_extractor_batch_size=args.feature_extractor_batch_size,
        feature_matching_batch_size=args.feature_matching_batch_size,
        previous_cusfm_ws=args.previous_cusfm_ws,
        previous_raw_image_dir=args.previous_raw_image_dir,
        add_tensorrt_path=args.add_tensorrt_path,
        anchor_track=args.anchor_track,
        global_localize_succ_sample=args.global_localize_succ_sample,
        use_cuvslam_slam_pose=args.use_cuvslam_slam_pose,
        skip_track_global_transform=args.skip_track_global_transform,
        skip_data_association=args.skip_data_association)

    # Validate parameters
    if cusfm_runner.do_rolling_shutter_correction and (
            not cusfm_runner.skip_mapper):
        assert cusfm_runner.input_dir != '', (
            'Input directory is required when mapper ' +
            'is not skipped and use_roll_shutter_correction is True')

    assert cusfm_runner.skip_feature_extractor or cusfm_runner.input_dir != '', \
        'Input directory is required when feature_extractor is not skipped'

    print_params(cusfm_runner.__dict__)
    cusfm_runner.run_all()


if __name__ == "__main__":
    main()
