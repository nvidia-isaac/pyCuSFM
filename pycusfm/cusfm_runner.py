# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
import pkg_resources
import os
import shutil
import logging
from pathlib import Path
from .command_runner import CommandRunner  # Ensure this import path is correct
from .constants import (
    kKEYFRAME_DIR, kPOSE_GRAPH_DIR, kMATCHES_DIR, kMATCHES_TASK_DIR, kMAP_DIR,
    kOPEN_MAP_DIR, kCUVGL_MAP_DIR, kVOC_DIR, kASSOCIATIONS_DIR,
    kCUVSLAM_OUTPUT_DIR, kOUTPUT_POSES_DIR, kFRAME_META_FILE,
    kFRAME_META_FILE_CUVSLAM, kSLAM_POSES_FILE, kODOM_POSES_FILE)


def setup_logger():
    """Setup logger if it hasn't been configured yet."""
    logger = logging.getLogger(__name__)

    # Only configure if no handlers exist (avoid duplicate configuration)
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# Get logger instance
logger = setup_logger()


def list_subdirectories(directory):
    subdirectories = []

    root_dir = os.path.abspath(directory)

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            subdirectories.append((item, item_path))

    return subdirectories


class CuSFMRunner:

    def __init__(
            self,
            input_dir,
            cusfm_base_dir,
            binary_dir,
            config_dir,
            model_dir,
            mask_dir=None,
            feature_type="sift",
            skip_feature_extractor=False,
            skip_vocab_generator=False,
            skip_pose_graph=False,
            skip_matcher=False,
            skip_mapper=False,
            skip_map_convertor=False,
            skip_cuvslam=False,
            enable_debug=False,
            num_threads=1,
            override_frames_meta_file="",
            cuvgl_dir="",
            optimize_extrinsics=False,
            ba_frame_type=None,
            min_inter_frame_distance=0.5,
            min_inter_frame_rotation_degrees=5,
            dry_run=False,
            multi_track_input=False,
            downsampling_matches=True,
            use_roll_shutter_correction=False,
            sample_sync_threshold_microseconds=0,
            stereo_pair_non_baseline_max_distance=0,
            use_vehicle_trajectory=False,
            debug_interval=500,
            max_keypoints_num=-1,
            feature_extractor_batch_size=0,
            feature_matching_batch_size=0,
            previous_cusfm_ws="",
            previous_raw_image_dir="",
            export_pose_in_vehicle_frame=True,
            add_tensorrt_path=True,
            anchor_track="",
            global_localize_succ_sample=10,
            use_cuvslam_slam_pose=True,
            skip_track_global_transform=False,
            skip_data_association=False):

        self.input_dir = input_dir
        self.cusfm_base_dir = cusfm_base_dir
        # False will use odom pose
        self.use_cuvslam_slam_pose = use_cuvslam_slam_pose
        self.skip_track_global_transform = skip_track_global_transform
        self.mask_dir = mask_dir
        self.feature_type = feature_type
        self.skip_feature_extractor = skip_feature_extractor
        self.skip_vocab_generator = skip_vocab_generator
        self.skip_pose_graph = skip_pose_graph
        self.skip_matcher = skip_matcher
        self.skip_mapper = skip_mapper
        self.skip_map_convertor = skip_map_convertor
        self.skip_cuvslam = skip_cuvslam
        self.enable_debug = enable_debug
        self.num_threads = num_threads
        self.override_frames_meta_file = override_frames_meta_file
        self.dry_run = dry_run
        self.optimize_extrinsics = optimize_extrinsics
        self.ba_frame_type = ba_frame_type
        self.multi_track_input = multi_track_input
        self.downsampling_matches = downsampling_matches
        self.do_rolling_shutter_correction = use_roll_shutter_correction
        self.min_inter_frame_distance = min_inter_frame_distance
        self.min_inter_frame_rotation_degrees = min_inter_frame_rotation_degrees
        self.sample_sync_threshold_microseconds = sample_sync_threshold_microseconds
        self.stereo_pair_non_baseline_max_distance = stereo_pair_non_baseline_max_distance
        self.use_vehicle_trajectory = use_vehicle_trajectory
        self.previous_cusfm_ws = previous_cusfm_ws
        self.previous_raw_image_dir = previous_raw_image_dir
        self.export_pose_in_vehicle_frame = export_pose_in_vehicle_frame
        self.add_tensorrt_path = add_tensorrt_path
        self.max_keypoints_num = max_keypoints_num

        self.keyframe_dir = os.path.join(self.cusfm_base_dir, kKEYFRAME_DIR)
        self.feature_extractor_batch_size = feature_extractor_batch_size
        self.feature_matching_batch_size = feature_matching_batch_size
        self.pose_graph_dir = os.path.join(
            self.cusfm_base_dir, kPOSE_GRAPH_DIR)
        self.matches_dir = os.path.join(self.cusfm_base_dir, kMATCHES_DIR)
        self.tasks_dir = os.path.join(self.matches_dir, kMATCHES_TASK_DIR)
        self.map_dir = os.path.join(self.cusfm_base_dir, kMAP_DIR)
        self.open_map_dir = os.path.join(self.cusfm_base_dir, kOPEN_MAP_DIR)
        self.cuvslam_output_dir = os.path.join(
            self.cusfm_base_dir, kCUVSLAM_OUTPUT_DIR)
        self.cuvslam_output_keyframe_metadata_file = os.path.join(
            self.cuvslam_output_dir, kFRAME_META_FILE_CUVSLAM)
        if cuvgl_dir == "":
            self.cuvgl_dir = os.path.join(self.cusfm_base_dir, kCUVGL_MAP_DIR)
        else:
            self.cuvgl_dir = cuvgl_dir
        self.voc_dir = os.path.join(self.cuvgl_dir, kVOC_DIR)
        self.association_dir = os.path.join(
            self.cusfm_base_dir, kASSOCIATIONS_DIR)
        self.debug_interval = debug_interval
        if feature_type == "superpoint":
            self.model_dir = os.path.join(model_dir, "superpoint_lightglue")
        elif feature_type == "aliked":
            self.model_dir = os.path.join(model_dir, "aliked_lightglue")
        else:
            self.model_dir = "."

        # single track data association mode is replaced by  incremental pose_graph because it is better
        self.anchor_track = anchor_track
        self.skip_data_association = skip_data_association
        # Enforce data-association only when multi-track is enabled
        if self.multi_track_input:
            self.global_localize_succ_sample = global_localize_succ_sample

        self.runner = CommandRunner(
            binary_dir=binary_dir,
            config_dir=config_dir,
            dry_run=self.dry_run,
            add_tensorrt_path=self.add_tensorrt_path,
            runtime_log_dir=self.cusfm_base_dir)
        self.logger = logging.getLogger('cusfm_runner')

    def run_cuvslam(
            self,
            input_dir,
            cuvslam_output_dir,
            override_frames_meta_file=""):
        """Run CUVSLAM processing to generate pose estimates.

        Args:
            input_dir (str): Directory containing input images and metadata.
            cuvslam_output_dir (str): Output directory for CUVSLAM results.
            override_frames_meta_file (str, optional): Override frames metadata file path.

        Returns:
            str: Path to keyframe metadata file with pose information.
                - If skip_cuvslam=True: Returns input metadata file path (no processing).
                - If skip_cuvslam=False: Returns updated metadata file path in cuvslam_output_dir.

        Raises:
            RuntimeError: If pose file not found after CUVSLAM processing.
        """
        if override_frames_meta_file:
            keyframe_file_to_use = override_frames_meta_file
        else:
            keyframe_file_to_use = os.path.join(input_dir, kFRAME_META_FILE)

        if self.skip_cuvslam:
            return keyframe_file_to_use

        log_dir = os.path.join(cuvslam_output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # first convert keyframe metadata to edex
        self.runner.run_binary(
            'keyframe_metadata_to_edex_main', [
                "--keyframe_metadata_file", keyframe_file_to_use,
                "--output_edex_dir", input_dir
            ],
            logs_file_path=os.path.join(
                log_dir, 'keyframe_metadata_to_edex_main.txt'))

        # then run cuvslam_api_launcher
        self.logger.info("Running cuvslam_api_launcher")
        self.runner.run_binary(
            'cuvslam_api_launcher', [
                "--dataset",
                input_dir,
                "--output_map",
                os.path.join(cuvslam_output_dir, "cuvslam_map"),
                "--print_format",
                "tum",
                "--ros_frame_conversion=true",
                "--cfg_enable_slam",
                "--print_odom_poses",
                os.path.join(cuvslam_output_dir, kODOM_POSES_FILE),
                "--print_slam_poses",
                os.path.join(cuvslam_output_dir, kSLAM_POSES_FILE),
            ],
            logs_file_path=os.path.join(log_dir, 'cuvslam_api_launcher.txt'))

        # Update poses in frames_meta.json
        if self.use_cuvslam_slam_pose:
            use_pose_file = kSLAM_POSES_FILE
        else:
            use_pose_file = kODOM_POSES_FILE

        pose_file_path = os.path.join(cuvslam_output_dir, use_pose_file)
        if not os.path.isfile(pose_file_path):
            raise RuntimeError(
                f"Expected pose file not found: {pose_file_path}")

        output_keyframe_metadata_file = os.path.join(
            cuvslam_output_dir, kFRAME_META_FILE_CUVSLAM)

        self.runner.run_binary(
            'update_keyframe_pose_main', [
                "--input_file",
                keyframe_file_to_use,
                "--output_file",
                output_keyframe_metadata_file,
                "--tum_pose_file",
                pose_file_path,
            ],
            logs_file_path=os.path.join(
                log_dir, 'update_keyframe_pose_main.txt'))

        return output_keyframe_metadata_file

    def extract_features(
            self,
            input_dir,
            keyframe_dir,
            override_frames_meta_file=""):
        if not self.skip_feature_extractor:
            self.logger.info("Extracting features ...")
            self.runner.remove_directory(keyframe_dir)
            self.runner.ensure_directory_exists(keyframe_dir)
            logs_file = os.path.join(keyframe_dir, 'logs.txt')

            if override_frames_meta_file:
                keyframe_file_to_use = override_frames_meta_file
            else:
                keyframe_file_to_use = os.path.join(
                    input_dir, kFRAME_META_FILE)

            cmd = [
                'feature_extractor_main',
                "--input_image_directory",
                input_dir,
                "--frames_meta_file",
                keyframe_file_to_use,
                "--output_dir",
                keyframe_dir,
                "--keypoint_creation_config",
                self.runner.get_config_path('keypoint_creation_config.pb.txt'),
                "--num_thread",
                str(self.num_threads),
                "--feature_type",
                self.feature_type,
                "--min_inter_frame_distance",
                str(self.min_inter_frame_distance),
                "--min_inter_frame_rotation_degrees",
                str(self.min_inter_frame_rotation_degrees),
                "--sample_sync_threshold_microseconds",
                str(self.sample_sync_threshold_microseconds),
                "--stereo_pair_non_baseline_max_distance",
                str(self.stereo_pair_non_baseline_max_distance),
                "--model_dir",
                str(self.model_dir),
                "--debug_image_interval",
                str(self.debug_interval),
                "--batch_size",
                str(self.feature_extractor_batch_size),
            ]

            if self.enable_debug:
                debug_images_dir = os.path.join(keyframe_dir, 'debug_images')
                cmd.extend(
                    [
                        "--debug_dir", debug_images_dir,
                        "--debug_image_interval", "50"
                    ])

            if self.mask_dir:
                cmd.extend(["--input_image_mask_directory", self.mask_dir])

            if self.max_keypoints_num > 0:
                cmd.extend(
                    ["--max_keypoints_num",
                     str(self.max_keypoints_num)])

            self.runner.run_binary(cmd[0], cmd[1:], logs_file_path=logs_file)
            self.logger.info(
                "\033[0;32m Feature Extraction Finished ...\033[0m")

    def build_anchor_cuvgl_map(self, cuvslam_dir, cuvgl_dir):
        """Build anchor CUVGL map
        Args:
            cuvslam_dir: CUVSLAM output directory
            cuvgl_dir: CUVGL directory
            return cuvgl_map directory
        """
        self.logger.info("Generate anchor CUVGL map...")
        track_name = self.anchor_track

        anchor_raw_dir = os.path.join(self.input_dir, track_name)

        if not os.path.isdir(anchor_raw_dir):
            raise FileNotFoundError(
                f"Anchor track folder '{anchor_raw_dir}' does not exist")

        track_cuvslam_output_dir = os.path.join(cuvslam_dir, track_name)
        self.runner.remove_directory(track_cuvslam_output_dir)
        self.runner.ensure_directory_exists(track_cuvslam_output_dir)

        cuvslam_output_keyframe_metadata_file = self.run_cuvslam(
            anchor_raw_dir, track_cuvslam_output_dir)

        track_keyframe_dir = os.path.join(self.keyframe_dir, track_name)
        self.extract_features(
            anchor_raw_dir, track_keyframe_dir,
            cuvslam_output_keyframe_metadata_file)

        if not self.skip_track_global_transform:
            cuvgl_map_dir = os.path.join(cuvgl_dir, "anchor_cuvgl_map")
            self.runner.remove_directory(cuvgl_map_dir)
            self.runner.ensure_directory_exists(cuvgl_map_dir)

            voc_dir = os.path.join(cuvgl_map_dir, kVOC_DIR)
            self.generate_bow(track_keyframe_dir, cuvgl_map_dir, voc_dir)

            # Create symbolic link
            anchor_keyframe_link = os.path.join(cuvgl_map_dir, "keyframes")
            if os.path.islink(anchor_keyframe_link) or os.path.isfile(
                    anchor_keyframe_link):
                os.unlink(anchor_keyframe_link)
            elif os.path.isdir(anchor_keyframe_link):
                shutil.rmtree(anchor_keyframe_link)

            os.symlink(track_keyframe_dir, anchor_keyframe_link)
            return cuvgl_map_dir
        else:
            return None

    def align_track_to_cuvglmap(
            self, track_dir, cuvgl_map_dir, result_dir, cuvslam_output_dir):
        """Align track to CUVGL map
        Args:
            track_dir: Track data directory
            cuvgl_map_dir: CUVGL map directory
            result_dir: Result output directory
        """
        self.logger.info("Aligning track to CUVGL map...")

        # Ensure result directory exists
        self.runner.ensure_directory_exists(result_dir)

        # Run global localizer evaluation
        localizer_logs_file = os.path.join(
            result_dir, 'run_global_localization_main.txt')
        self.runner.run_binary(
            'run_global_localization_main', [
                "--do_evaluate=true", "--input_image_directory", track_dir,
                "--localizer_config_folder",
                os.path.join(self.runner.config_dir), "--map_dir",
                cuvgl_map_dir, "--result_dir", result_dir, "--model_dir",
                self.model_dir, "--success_sample_num",
                str(self.global_localize_succ_sample)
            ],
            logs_file_path=localizer_logs_file)

        # Convert ego to global coordinates
        if self.use_cuvslam_slam_pose:
            use_pose_fle = kSLAM_POSES_FILE
        else:
            use_pose_fle = kODOM_POSES_FILE

        cuvslam_poses_file = os.path.join(cuvslam_output_dir, use_pose_fle)
        cuvslam_poses_global_file = os.path.join(
            cuvslam_output_dir, "global_poses.tum")

        # that is possible there are no successful localization, no overlap between the two tracks !!!
        loc_results_file = os.path.join(result_dir, "loc_results.json")
        if not os.path.isfile(loc_results_file):
            raise RuntimeError(
                f"Global-localiser did not create '{loc_results_file}'. "
                "Alignment cannot continue.")

        convert_logs_file = os.path.join(
            result_dir, 'estimate_transform_between_poses_main.txt')
        self.runner.run_binary(
            'estimate_transform_between_poses_main', [
                "--loc_results_file", loc_results_file, "--input_session_tum",
                cuvslam_poses_file, "--output_tum", cuvslam_poses_global_file
            ],
            logs_file_path=convert_logs_file)

        if not os.path.exists(cuvslam_poses_global_file):
            raise RuntimeError(
                f"Fail to generate global pose file '{cuvslam_poses_global_file}'. "
                "Alignment cannot continue.")

        global_pose_meta_data_file = os.path.join(
            cuvslam_output_dir, "frames_meta_cuvslam_global.json")

        self.runner.run_binary(
            'update_keyframe_pose_main', [
                "--input_file",
                os.path.join(track_dir, kFRAME_META_FILE),
                "--output_file",
                global_pose_meta_data_file,
                "--tum_pose_file",
                cuvslam_poses_global_file,
            ],
            logs_file_path=os.path.join(
                cuvslam_output_dir, 'update_keyframe_pose_main.txt'))

        self.logger.info("Track alignment completed!")
        return global_pose_meta_data_file

    def multi_track_keyframe_for_av(self):
        track_dirs = list_subdirectories(self.input_dir)

        for track_name, track_dir in track_dirs:
            track_keyframe_dir = os.path.join(
                self.keyframe_dir, track_name)
            self.logger.info(
                f"Processing track: {track_dir}, keyframe_dir: {track_keyframe_dir}"
            )

            self.extract_features(
                track_dir, track_keyframe_dir)

        self.keyframe_aggregation(self.keyframe_dir)

    def multi_track_keyframe_for_isaac(self):
        self.logger.info("Aggregating frames ...")
        track_dirs = list_subdirectories(self.input_dir)

        if not track_dirs:
            raise RuntimeError(
                "Muist use ln-sf <real ros_bag_Data> <link> "
                "to include all tracks data")

        #  if anchor_track is empty, the first folder will be anchor
        if not self.anchor_track:
            self.anchor_track = track_dirs[0][0]
            self.logger.info(
                f"anchor track is not set, will automatically set to {self.anchor_track}"
            )

        cuvslam_dir = os.path.join(
            self.cusfm_base_dir, 'cuvslam_multi_track')
        self.runner.remove_directory(cuvslam_dir)
        self.runner.ensure_directory_exists(cuvslam_dir)

        anchor_cuvgl_dir = os.path.join(
            self.cusfm_base_dir, 'anchor_cuvgl')
        self.runner.remove_directory(anchor_cuvgl_dir)
        self.runner.ensure_directory_exists(anchor_cuvgl_dir)

        # for anchor track, build cuvgl_map use cuvslam
        anchor_cuvgl_map_dir = self.build_anchor_cuvgl_map(
            cuvslam_dir, anchor_cuvgl_dir)

        # for un-anchor track
        for track_name, track_dir in track_dirs:
            if track_name == self.anchor_track:
                continue
            track_keyframe_dir = os.path.join(
                self.keyframe_dir, track_name)
            self.logger.info(
                f"Processing track: {track_dir}, keyframe_dir: {track_keyframe_dir}"
            )

            track_cuvslam_output_dir = os.path.join(
                cuvslam_dir, track_name)
            self.runner.remove_directory(track_cuvslam_output_dir)
            self.runner.ensure_directory_exists(track_cuvslam_output_dir)

            cuvslam_output_keyframe_metadata_file = self.run_cuvslam(
                track_dir, track_cuvslam_output_dir)

            if not self.skip_track_global_transform:
                result_dir = os.path.join(anchor_cuvgl_dir, track_name)
                global_pose_meta_data_file = self.align_track_to_cuvglmap(
                    track_dir, anchor_cuvgl_map_dir, result_dir,
                    track_cuvslam_output_dir)
            else:
                global_pose_meta_data_file = cuvslam_output_keyframe_metadata_file

            self.extract_features(
                track_dir, track_keyframe_dir, global_pose_meta_data_file)

        self.keyframe_aggregation(self.keyframe_dir)

        self.logger.info(
            "\033[0;32m Frames Aggregating Finished ...\033[0m")

    def keyframe_aggregation(self, keyframe_dir):
        keyframe_aggregation_logs_file = os.path.join(
            keyframe_dir, 'keyframe_aggregation_main.txt')
        self.runner.run_binary(
            'keyframe_aggregation_main', [
                "--keyframe_directory", keyframe_dir
            ],
            logs_file_path=keyframe_aggregation_logs_file)

    def generate_bow(self, keyframe_dir, cuvgl_dir, voc_dir):
        if not self.skip_vocab_generator:
            self.logger.info("Building BoW ...")
            self.runner.remove_directory(voc_dir)
            self.runner.ensure_directory_exists(voc_dir)
            bow_vocabulary_logs_file = os.path.join(
                voc_dir, 'generate_bow_vocabulary_main.txt')
            bow_index_logs_file = os.path.join(
                cuvgl_dir, 'generate_bow_index_main.txt')

            self.runner.run_binary(
                'generate_bow_vocabulary_main', [
                    "--keyframe_directory", keyframe_dir,
                    "--image_retrieval_config",
                    self.runner.get_config_path(
                        'image_retrieval_config.pb.txt'), "--output_voc_dir",
                    voc_dir
                ],
                logs_file_path=bow_vocabulary_logs_file)

            self.runner.run_binary(
                'generate_bow_index_main', [
                    "--keyframe_directory", keyframe_dir, "--output_map_dir",
                    cuvgl_dir, "--image_retrieval_config",
                    self.runner.get_config_path(
                        'image_retrieval_config.pb.txt')
                ],
                logs_file_path=bow_index_logs_file)
            self.logger.info("\033[0;32m BoW Build Finished ...\033[0m")

    def generate_associations(self):
        self.logger.info("Generating associations ...")
        self.runner.remove_directory(self.association_dir)
        self.runner.ensure_directory_exists(self.association_dir)
        associations_logs_file = os.path.join(
            self.association_dir, 'associations_main.txt')

        cmd = [
            'generate_association_main', "--keyframe_dir", self.keyframe_dir,
            "--gl_map_dir", self.cuvgl_dir, "--output_dir",
            self.association_dir, "--association_worker_config",
            self.runner.get_config_path('association_config.pb.txt'),
            "--matching_worker_config",
            self.runner.get_config_path('matching_task_worker_config.pb.txt'),
            "--model_dir", self.model_dir,
        ]

        if self.enable_debug:
            cmd.extend(["--image_dir", self.input_dir])

        if self.skip_pose_graph:
            cmd.extend(["--skip_pose_graph=true"])

        self.runner.run_binary(
            cmd[0], cmd[1:], logs_file_path=associations_logs_file)

        self.logger.info(
            "\033[0;32m Associations Generation Finished ...\033[0m")

    def pose_graph(self):
        if not self.skip_pose_graph:
            self.logger.info("Pose graph ...")
            self.runner.remove_directory(self.pose_graph_dir)
            self.runner.ensure_directory_exists(self.pose_graph_dir)
            pose_graph_logs_file = os.path.join(
                self.pose_graph_dir, 'pose_graph_main.txt')

            cmd = [
                'pose_graph_main', "--keyframe_directory", self.keyframe_dir,
                "--match_directory", self.matches_dir, "--voc_directory",
                self.cuvgl_dir, "--config_directory",
                self.runner.get_config_folder(), "--pose_graph_directory",
                self.pose_graph_dir, '--model_dir', self.model_dir
            ]

            if self.enable_debug:
                debug_images_dir = os.path.join(self.pose_graph_dir, 'debug')
                cmd.extend(
                    [
                        "--debug_dir", debug_images_dir, "--raw_dir",
                        self.input_dir
                    ])

            self.runner.run_binary(
                cmd[0], cmd[1:], logs_file_path=pose_graph_logs_file)

            self.logger.info("\033[0;32m Pose Graph Finished ...\033[0m")

    def association_graph(self):
        self.logger.info("Association graph ...")
        self.runner.remove_directory(self.pose_graph_dir)
        self.runner.ensure_directory_exists(self.pose_graph_dir)
        pose_graph_logs_file = os.path.join(
            self.pose_graph_dir, 'association_graph_main.txt')

        cmd = [
            'pose_graph_association_main', "--association_directory",
            self.association_dir, "--pose_graph_directory",
            self.pose_graph_dir, "--pose_graph_config",
            self.runner.get_config_path('pose_graph_config.pb.txt')
        ]

        if self.enable_debug:
            debug_images_dir = os.path.join(self.pose_graph_dir, 'debug')
            cmd.extend(["--debug_directory", debug_images_dir])

        self.runner.run_binary(
            cmd[0], cmd[1:], logs_file_path=pose_graph_logs_file)

        self.logger.info("\033[0;32m Association Graph Finished ...\033[0m")

    def match_features(self):
        if not self.skip_matcher:
            self.logger.info("Feature matching ...")
            self.runner.remove_directory(self.matches_dir)
            self.runner.ensure_directory_exists(self.tasks_dir)

            input_keyframe_dir = self.keyframe_dir

            new_frame_meta_file = ''
            if not self.skip_pose_graph:
                new_frame_meta_file = os.path.join(
                    self.pose_graph_dir, kFRAME_META_FILE)

            task_logs_file = os.path.join(
                self.matches_dir, 'feature_matcher_task_builder_main.txt')

            task_builder_args = [
                "--current_keyframe_directory",
                input_keyframe_dir,
                "--output_dir",
                self.tasks_dir,
                "--match_pair_select_config",
                self.runner.get_config_path('match_pair_select_config.pb.txt'),
                f'--downsampling_matches={self.downsampling_matches}',
            ]

            if not self.skip_pose_graph:
                task_builder_args.extend(
                    ["--pose_graph_association_dir", self.pose_graph_dir])
            else:
                task_builder_args.extend(
                    ["--pose_graph_association_dir", self.association_dir])

            if self.previous_cusfm_ws != "":
                task_builder_args.extend(
                    ["--previous_cusfm_ws", self.previous_cusfm_ws])

            if os.path.exists(self.cuvgl_dir):
                task_builder_args.extend(
                    ["--current_retrieval_db_dir", self.cuvgl_dir])

            if new_frame_meta_file:
                task_builder_args.extend(
                    ["--keyframe_metadata_file", new_frame_meta_file])

            self.runner.run_binary(
                'feature_matcher_task_builder_main',
                task_builder_args,
                logs_file_path=task_logs_file)

            if self.previous_cusfm_ws != "":
                print("Using patch mode, using new frame meta file")
                new_frame_meta_file = os.path.join(
                    self.tasks_dir, kFRAME_META_FILE)

            commands = []
            for task_file_name in os.listdir(self.tasks_dir):
                if not task_file_name.endswith('.pb'):
                    continue
                matcher_logs_file = os.path.join(
                    self.matches_dir,
                    f'feature_matcher_main_{task_file_name}.txt')
                matcher_args = [
                    "--current_keyframe_directory", input_keyframe_dir,
                    "--output_dir", self.matches_dir, "--task_file",
                    os.path.join(self.tasks_dir,
                                 task_file_name), "--matching_worker_config",
                    self.runner.get_config_path(
                        'matching_task_worker_config.pb.txt'),
                    "--match_pair_select_config",
                    self.runner.get_config_path(
                        'match_pair_select_config.pb.txt'), "--num_thread",
                    "1", "--model_dir", self.model_dir, "--debug_interval",
                    str(self.debug_interval), "--batch_size",
                    str(self.feature_matching_batch_size),
                    f"--use_rolling_shutter_correction={self.do_rolling_shutter_correction}"
                ]
                if self.do_rolling_shutter_correction or self.enable_debug:
                    matcher_args.extend(
                        ["--current_raw_image_dir", self.input_dir])

                if self.previous_cusfm_ws != "":
                    matcher_args.extend(
                        [
                            "--previous_cusfm_ws", self.previous_cusfm_ws,
                            "--keyframe_metadata_file", new_frame_meta_file
                        ])
                elif new_frame_meta_file:
                    matcher_args.extend(
                        ["--keyframe_metadata_file", new_frame_meta_file])

                if self.enable_debug:
                    debug_images_dir = os.path.join(self.matches_dir, 'debug')
                    matcher_args.extend([
                        "--debug_dir",
                        debug_images_dir,
                    ])

                    if self.previous_cusfm_ws:
                        matcher_args.extend(
                            [
                                "--previous_raw_image_dir",
                                self.previous_raw_image_dir
                            ])

                commands.append(
                    {
                        'binary_name': 'feature_matcher_main',
                        'args': matcher_args,
                        'log_file': matcher_logs_file
                    })

            print(f"Running {len(commands)} feature matching tasks")
            self.runner.run_binaries_parallel(commands)
            self.logger.info("\033[0;32m Feature Matching Finished ...\033[0m")

    def map_keypoints(self):
        if not self.skip_mapper:
            self.logger.info("Mapping ...")
            self.runner.ensure_directory_exists(self.map_dir)

            input_keyframe_dir = self.keyframe_dir

            cmd = [
                'keypoints_mapper_main',
                "--mapping_config_file",
                self.runner.get_config_path('vision_mapping_config.pb.txt'),
                "--keyframe_dir",
                input_keyframe_dir,
                "--matches_dir",
                self.matches_dir,
                "--output_map_dir",
                self.map_dir,
                "--raw_data_dir",
                self.input_dir,
                f'--use_vehicle_trajectory={self.use_vehicle_trajectory}',
            ]

            if not self.skip_pose_graph:
                new_frame_meta_file = os.path.join(
                    self.pose_graph_dir, kFRAME_META_FILE)
                cmd.extend(["--keyframe_metadata_file", new_frame_meta_file])
            elif self.previous_cusfm_ws != "":
                new_frame_meta_file = os.path.join(
                    self.tasks_dir, kFRAME_META_FILE)
                cmd.extend(
                    [
                        "--keyframe_metadata_file", new_frame_meta_file,
                        "--previous_cusfm_ws", self.previous_cusfm_ws
                    ])

            if self.do_rolling_shutter_correction:
                cmd.extend(
                    [
                        f'--do_rolling_shutter_correction={self.do_rolling_shutter_correction}'
                    ])

            if self.ba_frame_type:
                cmd.extend(["--ba_frame_type", self.ba_frame_type])

            if self.optimize_extrinsics:
                cmd.extend(
                    [f'--optimize_extrinsics={self.optimize_extrinsics}'])

            if self.anchor_track:
                cmd.extend(["--fixed_track_name", self.anchor_track])

            if self.enable_debug:
                debug_map_dir = os.path.join(self.map_dir, 'debug')
                cmd.extend(["--debug_dir", debug_map_dir])

            logs_file = os.path.join(self.map_dir, 'keypoints_mapper_main.txt')
            self.runner.run_binary(cmd[0], cmd[1:], logs_file_path=logs_file)
            self.logger.info("\033[0;32m Mapping Finished ...\033[0m")

    def refine_extrinsics(self):
        self.logger.info('Refine extrinsic ...')
        self.runner.ensure_directory_exists(self.map_dir)
        debug_map_dir = ''

        if self.enable_debug:
            debug_map_dir = os.path.join(self.map_dir, 'debug')

        new_frame_meta_file = os.path.join(
            self.pose_graph_dir, kFRAME_META_FILE)
        cmd = [
            'keypoints_mapper_main',
            '--mapping_config_file',
            self.runner.get_config_path('vision_mapping_config.pb.txt'),
            '--keyframe_dir',
            self.keyframe_dir,
            '--matches_dir',
            self.matches_dir,
            '--output_map_dir',
            self.map_dir,
            f'--optimize_extrinsics={self.optimize_extrinsics}',
            f'--ba_frame_type={self.ba_frame_type}',
            f'--raw_data_dir={self.input_dir}',
            f'--use_vehicle_trajectory={self.use_vehicle_trajectory}',
        ]

        if self.do_rolling_shutter_correction:
            cmd.extend(
                [
                    f'--do_rolling_shutter_correction={self.do_rolling_shutter_correction}'
                ])

        if os.path.exists(new_frame_meta_file):
            cmd.extend(['--keyframe_metadata_file', new_frame_meta_file])

        if self.enable_debug:
            cmd.extend(['--debug_dir', debug_map_dir])

        logs_file = os.path.join(self.map_dir, 'extrinsic_refiner_main.txt')
        self.runner.run_binary(cmd[0], cmd[1:], logs_file_path=logs_file)
        self.logger.info('\033[0;32m Refine extrinsic Finished ...\033[0m')

    def convert_map(self):
        if not self.skip_map_convertor:
            self.logger.info("Map conversion ...")
            self.runner.remove_directory(self.open_map_dir)
            self.runner.ensure_directory_exists(self.open_map_dir)

            logs_file = os.path.join(self.open_map_dir, 'kpmap_to_colmap.txt')
            self.runner.run_binary(
                'kpmap_to_colmap',
                ["--map_dir", self.map_dir, "--output_dir", self.open_map_dir],
                logs_file_path=logs_file)
            self.logger.info("\033[0;32m Map Conversion Finished ...\033[0m")

    def convert_poses(
            self,
            keyframe_file=None,
            output_pose_dir=None,
            pose_file_type="tum",
            debug=False,
            export_pose_in_vehicle_frame=True):
        """Convert poses from map to specified format.

        Args:
            keyframe_file: Path to keyframe metadata file. If None, uses the one from map_dir.
            output_pose_dir: Directory name for the output poses.
            pose_file_type: Pose file type, default is 'tum'.
            pose_dir: Directory to store output poses. If None, uses cusfm_base_dir/output_poses.
            debug: Whether to enable debug mode.
        """
        if self.skip_map_convertor:
            return

        if keyframe_file is None:
            keyframe_file = os.path.join(
                self.map_dir, kKEYFRAME_DIR, kFRAME_META_FILE)

        if output_pose_dir is None:
            output_pose_dir = os.path.join(
                self.cusfm_base_dir, kOUTPUT_POSES_DIR)

        print(f"\033[0;32m Pose Conversion - {output_pose_dir} \033[0m")
        self.runner.ensure_directory_exists(output_pose_dir)
        logs_file = os.path.join(
            output_pose_dir, 'extract_pose_from_map_main_log.txt')

        extract_pose_binary = "extract_pose_from_map_main"

        pose_conversion_cmd = [
            extract_pose_binary,
            "--input_keyframe_metadata_file",
            keyframe_file,
            "--output_pose_dir",
            output_pose_dir,
            "--output_pose_file_type",
            pose_file_type,
            "--export_pose_in_vehicle_frame=" +
            str(export_pose_in_vehicle_frame),
            "--also_merge_poses",
        ]

        if debug:
            print(f"Running command: {' '.join(pose_conversion_cmd)}")

        self.runner.run_binary(
            pose_conversion_cmd[0],
            pose_conversion_cmd[1:],
            logs_file_path=logs_file)

    def run_all(self):

        if self.multi_track_input:
            # For data with a stereo configuration in Isaac, we need to use cuvslam and stereo cameras to solve for relative poses.
            # However, this strategy cannot be applied to AV data.
            if not self.skip_cuvslam:
                self.multi_track_keyframe_for_isaac()
            else:
                self.multi_track_keyframe_for_av()
        else:
            output_frames_meta_file = self.run_cuvslam(
                self.input_dir, self.cuvslam_output_dir,
                self.override_frames_meta_file)

            self.extract_features(
                self.input_dir, self.keyframe_dir, output_frames_meta_file)

        self.generate_bow(self.keyframe_dir, self.cuvgl_dir, self.voc_dir)

        # generate associations for both feature_matcher and pose_graph
        # because  feature_matcher choose to use pose_graph for association
        # av_data may not use this
        if not self.skip_data_association:
            self.generate_associations()

        if not self.skip_pose_graph:
            if self.multi_track_input:
                self.association_graph()
            else:
                self.pose_graph()

        self.match_features()
        self.map_keypoints()
        self.convert_map()
        self.convert_poses(
            export_pose_in_vehicle_frame=self.export_pose_in_vehicle_frame)
        self.logger.info("Finish all tasks ...")


def get_default_cusfm_params(package_path: Path):
    """Get default parameters for CUSFM runner.

    Args:
        package_path (Path, optional): Repository root path. If None, uses REPO_PATH constant.

    Returns:
        dict: Dictionary containing default CUSFM parameters
    """

    return {
        'input_dir': '',  # Default empty string
        'cusfm_base_dir': '',  # Must be provided
        'cuvgl_dir': '',  # Default empty string
        'binary_dir': str(package_path / 'bin'),  # Default path
        'config_dir': str(package_path / 'configs/isaac'),  # Default path
        'model_dir': str(package_path / 'models'),  # Default path
        'feature_type': 'aliked',  # Default value
        'skip_feature_extractor': False,
        'skip_vocab_generator': False,
        'skip_pose_graph': False,
        'skip_matcher': False,
        'skip_mapper': False,
        'skip_map_convertor': False,
        'enable_debug': False,
        'num_threads': 1,  # Default value
        'override_frames_meta_file': '',  # Default empty string
        'dry_run': False,
        'optimize_extrinsics': False,
        'multi_track_input': False,
        'use_roll_shutter_correction': False,
        'sample_sync_threshold_microseconds': 100,  # Default value
        'stereo_pair_non_baseline_max_distance': 0.001,  # Default value
        'downsampling_matches': True,
        'min_inter_frame_distance': 0.5,  # Default value
        'min_inter_frame_rotation_degrees': 5.0,  # Default value
        'use_vehicle_trajectory': False,
        'feature_extractor_batch_size': 0,
        'feature_matching_batch_size': 0,
        'export_pose_in_vehicle_frame': True,
        'global_localize_succ_sample': 10,
        'use_cuvslam_slam_pose': True,
        'skip_track_global_transform': False,
        'skip_data_association': False
    }


def get_av_data_cusfm_params(package_path: Path):
    """Get CUSFM parameters configured for AV data.

    Args:
        package_path (Path, optional): Repository root path. If None, uses REPO_PATH constant.

    Returns:
        dict: Dictionary containing CUSFM parameters optimized for AV data
    """
    params = get_default_cusfm_params(package_path)

    # Override defaults with AV-specific settings
    params.update(
        {
            'config_dir': str(package_path / 'configs/av'),
            'downsampling_matches': False,
            'min_inter_frame_distance': 0.0,
            'min_inter_frame_rotation_degrees': 0.0,
            'skip_pose_graph': True,
            'use_roll_shutter_correction': True,
            'skip_data_association': True,
            'skip_cuvslam': True
        })

    return params


def create_cusfm_runner(package_path=None, av_data=False, **kwargs):
    """Create a default CuSFMRunner instance with optional overrides.

    Args:
        package_path (Path, optional): Repository root path. If None, uses REPO_PATH constant.
        av_data (bool): Whether to use AV data specific configurations
        **kwargs: Additional parameters to override defaults

    Returns:
        CuSFMRunner: Configured CuSFMRunner instance
    """

    if package_path is None:
        if os.environ.get('PYCUSFM_DIR'):
            package_path = Path(os.environ.get('PYCUSFM_DIR'))
        else:
            try:
                package_path = Path(
                    pkg_resources.resource_filename('pycusfm', ''))
            except (TypeError, AttributeError, ImportError, Exception):
                # Fallback: get the project root directory (parent of the pycusfm package directory)
                package_path = Path(__file__).parent
                print(
                    f"pkg_resources failed, using fallback path: {package_path}"
                )

    # Get base parameters based on av_data flag
    if av_data:
        params = get_av_data_cusfm_params(package_path)
    else:
        params = get_default_cusfm_params(package_path)

    # Override with any provided kwargs
    params.update({k: v for k, v in kwargs.items() if v is not None})

    # Create and return runner instance
    return CuSFMRunner(**params)
