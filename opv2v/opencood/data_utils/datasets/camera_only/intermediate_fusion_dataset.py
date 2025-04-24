"""
Fusion for intermediate level (camera)
"""
from collections import OrderedDict

import numpy as np
import torch

import opencood
from opencood.data_utils.datasets.camera_only import base_camera_dataset
from opencood.utils import common_utils


class CamIntermediateFusionDataset(base_camera_dataset.BaseCameraDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(CamIntermediateFusionDataset, self).__init__(params,
                                                           visualize,
                                                           train,
                                                           validate)
        self.visible = params['train_params']['visible']

    def __getitem__(self, idx):
        data_sample = self.get_sample_random(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = OrderedDict()

        ego_id = -999
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in data_sample.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(data_sample.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.get_pairwise_transformation(data_sample,
                                             self.params['train_params']['max_cav'])

        # Final shape: (L, M, H, W, 3)
        camera_data = []
        # (L, M, 3, 3)
        camera_intrinsic = []
        # (L, M, 4, 4)
        camera2ego = []

        # (max_cav, 4, 4)
        transformation_matrix = []
        # (1, H, W)
        gt_static = []
        # (1, h, w)
        gt_dynamic = []
        #shilpa lidar
        # lidar_data = OrderedDict()
        # lidar shape is dynamic
        lidar_data = {}

        # loop over all CAVs to process information
        #shilpa ego index
        ego_index_in_processed_matrix =-1
        for cav_id, selected_cav_base in data_sample.items():
            
            distance = common_utils.cav_distance_cal(selected_cav_base,
                                                     ego_lidar_pose)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = \
                self.get_single_cav(selected_cav_base)

            camera_data.append(selected_cav_processed['camera']['data'])
            camera_intrinsic.append(
                selected_cav_processed['camera']['intrinsic'])
            camera2ego.append(
                selected_cav_processed['camera']['extrinsic'])
            transformation_matrix.append(
                selected_cav_processed['transformation_matrix'])
            #shilpa lidar
            lidar_data.update({cav_id: selected_cav_processed['lidar']['data']})
            #shilpa ego index
            ego_index_in_processed_matrix +=1

            if cav_id == ego_id:
                #shilpa ego index
                ego_mat_index = ego_index_in_processed_matrix
                
                gt_dynamic.append(
                    selected_cav_processed['gt']['dynamic_bev'])
                gt_static.append(
                    selected_cav_processed['gt']['static_bev'])

        # stack all agents together
        camera_data = np.stack(camera_data)
        camera_intrinsic = np.stack(camera_intrinsic)
        camera2ego = np.stack(camera2ego)

        gt_dynamic = np.stack(gt_dynamic)
        gt_static = np.stack(gt_static)

        # padding
        transformation_matrix = np.stack(transformation_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
                                               transformation_matrix), 1, 1))
        transformation_matrix = np.concatenate(
            [transformation_matrix, padding_eye], axis=0)

        processed_data_dict['ego'].update({
            #shilpa include ego
            'ego_id': ego_id,
            'ego_mat_index': ego_mat_index,
            'transformation_matrix': transformation_matrix,
            'pairwise_t_matrix': pairwise_t_matrix,
            #shilpa lidar
            'lidar_data': lidar_data,
            'camera_data': camera_data,
            'camera_intrinsic': camera_intrinsic,
            'camera_extrinsic': camera2ego,
            'gt_dynamic': gt_dynamic,
            'gt_static': gt_static})

        return processed_data_dict

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        # return pairwise_t_matrix

        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix


    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for intermediate fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict()

        # update the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']
        selected_cav_processed.update({
            'transformation_matrix': transformation_matrix
        })

        # for intermediate fusion, we only need ego's gt
        if selected_cav_base['ego']:
            # process the groundtruth
            if self.visible:
                dynamic_bev = \
                    self.post_processor.generate_label(
                        selected_cav_base['bev_visibility_corp.png'])
            else:
                dynamic_bev = \
                    self.post_processor.generate_label(
                        selected_cav_base['bev_dynamic.png'])
            road_bev = \
                self.post_processor.generate_label(
                    selected_cav_base['bev_static.png'])
            lane_bev = \
                self.post_processor.generate_label(
                    selected_cav_base['bev_lane.png'])
            static_bev = self.post_processor.merge_label(road_bev, lane_bev)

            gt_dict = {'static_bev': static_bev,
                       'dynamic_bev': dynamic_bev}

            selected_cav_processed.update({'gt': gt_dict})

        all_camera_data = []
        all_camera_origin = []
        all_camera_intrinsic = []
        all_camera_extrinsic = []

        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            all_camera_origin.append(camera_data)
            camera_data = self.pre_processor.preprocess(camera_data)
            camera_intrinsic = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_intrinsic']
            cam2ego = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_extrinsic_to_ego']

            all_camera_data.append(camera_data)
            all_camera_intrinsic.append(camera_intrinsic)
            all_camera_extrinsic.append(cam2ego)

        camera_dict = {
            'origin_data': np.stack(all_camera_origin),
            'data': np.stack(all_camera_data),
            'intrinsic': np.stack(all_camera_intrinsic),
            'extrinsic': np.stack(all_camera_extrinsic)
        }

        selected_cav_processed.update({'camera': camera_dict})

        #shilpa lidar
        lidar_dict = {
            'data': np.stack(selected_cav_base['lidar_np']),
        }
        selected_cav_processed.update({'lidar': lidar_dict})

        return selected_cav_processed

    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        if not self.train:
            assert len(batch) == 1

        output_dict = {'ego': {}}

        cam_rgb_all_batch = []
        cam_to_ego_all_batch = []
        cam_intrinsic_all_batch = []

        gt_static_all_batch = []
        gt_dynamic_all_batch = []

        transformation_matrix_all_batch = []
        pairwise_t_matrix_all_batch = []
        # used to save each scenario's agent number
        record_len = []

        #shilpa lidar
        # lidar shape is dynamic
        lidar_data = []
        # extrinsic lidar to camera
        lidar_to_camera = []

        #shilpa ego index
        ego_mat_index = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']

            camera_data = ego_dict['camera_data']
            camera_intrinsic = ego_dict['camera_intrinsic']
            camera_extrinsic = ego_dict['camera_extrinsic']

            assert camera_data.shape[0] == \
                   camera_intrinsic.shape[0] == \
                   camera_extrinsic.shape[0]

            record_len.append(camera_data.shape[0])
            #shilpa ego index
            ego_mat_index.append(ego_dict['ego_mat_index'])

            cam_rgb_all_batch.append(camera_data)
            cam_intrinsic_all_batch.append(camera_intrinsic)
            cam_to_ego_all_batch.append(camera_extrinsic)

            # ground truth
            gt_dynamic_all_batch.append(ego_dict['gt_dynamic'])
            gt_static_all_batch.append(ego_dict['gt_static'])

            # transformation matrix
            transformation_matrix_all_batch.append(
                ego_dict['transformation_matrix'])
            # pairwise matrix
            pairwise_t_matrix_all_batch.append(ego_dict['pairwise_t_matrix'])

            #shilpa lidar
            lidar_data.append(ego_dict['lidar_data'])

            # compute the lidar to camera matrix
            # for frame in camera_extrinsic:
            #     for agent in frame:
            #         for camera in agent:
            #             lidar2cam_r = np.linalg.inv(camera[:3, :3])
            #             lidar2cam_t = camera[:3, 3] @ lidar2cam_r.T
            #             lidar2cam_rt = np.eye(4)
            #             lidar2cam_rt[:3, :3] = lidar2cam_r.T
            #             lidar2cam_rt[3, :3] = -lidar2cam_t
            #
            # for frame in camera_intrinsic:
            #     for agent in frame:
            #         for camera in agent:
            #             intrinsic = camera
            #             viewpad = np.eye(4)
            #             viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            #             lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            #             lidar_to_camera.append(lidar2img_rt)
            lidar_to_camera.append(self.get_lidar2cam(camera_extrinsic, camera_intrinsic))  # stacks it by batches

        # (B*L, 1, M, H, W, C)
        cam_rgb_all_batch = torch.from_numpy(
            np.concatenate(cam_rgb_all_batch, axis=0)).unsqueeze(1).float()
        cam_intrinsic_all_batch = torch.from_numpy(
            np.concatenate(cam_intrinsic_all_batch, axis=0)).unsqueeze(1).float()
        cam_to_ego_all_batch = torch.from_numpy(
            np.concatenate(cam_to_ego_all_batch, axis=0)).unsqueeze(1).float()
        # (B,)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        #shilpa ego index
        ego_mat_index = torch.from_numpy(np.array(ego_mat_index, dtype=int))

        # (B, 1, H, W)
        gt_static_all_batch = \
            torch.from_numpy(np.stack(gt_static_all_batch)).long()
        gt_dynamic_all_batch = \
            torch.from_numpy(np.stack(gt_dynamic_all_batch)).long()

        # (B,max_cav,4,4)
        transformation_matrix_all_batch = \
            torch.from_numpy(np.stack(transformation_matrix_all_batch)).float()
        pairwise_t_matrix_all_batch = \
            torch.from_numpy(np.stack(pairwise_t_matrix_all_batch)).float()
        
        #shilpa lidar
        lidar_to_camera = torch.from_numpy(np.concatenate(lidar_to_camera, axis=1)).float()

        # convert numpy arrays to torch tensor
        output_dict['ego'].update({
            #shilpa ego index
            'ego_mat_index': ego_mat_index,
            #shilpa lidar
            'lidar_data': lidar_data,
            'lidar_to_camera': lidar_to_camera,
            'inputs': cam_rgb_all_batch,
            'extrinsic': cam_to_ego_all_batch,
            'intrinsic': cam_intrinsic_all_batch,
            'gt_static': gt_static_all_batch,
            'gt_dynamic': gt_dynamic_all_batch,
            'transformation_matrix': transformation_matrix_all_batch,
            'pairwise_t_matrix': pairwise_t_matrix_all_batch,
            'record_len': record_len
        })

        return output_dict

    def post_process(self, batch_dict, output_dict):
        output_dict = self.post_processor.post_process(batch_dict,
                                                       output_dict)

        return output_dict

    #shilpa lidar
    def get_lidar2cam(self, camera_extrinsic, camera_intrinsic):
        lidar_to_camera = []

        # Iterate over frames
        for frame_idx, (frame_extrinsics, frame_intrinsics) in enumerate(zip(camera_extrinsic, camera_intrinsic)):
            frame_lidar_to_camera = []

            # Iterate over agents
            for agent_idx, (agent_extrinsics, agent_intrinsics) in enumerate(zip(frame_extrinsics, frame_intrinsics)):
                agent_lidar_to_camera = []

                # Iterate over cameras
                for camera_idx, (camera_extrinsic, camera_intrinsic) in enumerate(
                        zip(agent_extrinsics, agent_intrinsics)):
                    # Extract rotation (R) and translation (t)
                    R = camera_extrinsic[:3, :3]  # 3x3 rotation matrix
                    t = camera_extrinsic[:3, 3:]  # 3x1 translation vector

                    # Compute the inverse extrinsic matrix (LiDAR to camera)
                    R_inv = R.T  # Transpose of the rotation matrix
                    t_inv = -np.dot(R_inv, t)  # Inverse translation

                    # Construct the LiDAR-to-camera extrinsic matrix
                    extrinsic_lidar_to_cam = np.eye(4)
                    extrinsic_lidar_to_cam[:3, :3] = R_inv
                    extrinsic_lidar_to_cam[:3, 3] = t_inv[:, 0]

                    # Append the transformation matrix for this camera
                    agent_lidar_to_camera.append(extrinsic_lidar_to_cam)

                # Append the matrices for this agent
                frame_lidar_to_camera.append(agent_lidar_to_camera)

            # Append the matrices for this frame
            lidar_to_camera.append(frame_lidar_to_camera)

        # Convert list to a NumPy array or PyTorch tensor with the correct shape
        lidar_to_camera = np.array(lidar_to_camera)

        return lidar_to_camera