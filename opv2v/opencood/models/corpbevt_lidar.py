"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import math
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder, LidarEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from opencood.models.pointpillars.pillars_feature_net import *
from opencood.models.fusion_modality.modality_fuser import *

from mmdetection3d.projects.BEVFusion.demo.init_encoder import init_encoder
from mmdetection3d.projects.BEVFusion.demo.build_encoder import build_encoder
import open3d as o3d
import matplotlib.pyplot as plt
from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes, Det3DDataSample,
                                LiDARInstance3DBoxes, LiDARInstance3DBoxesVelocity)
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData


class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b l c h w  -> b l c w h')
        x = torch.flip(x, dims=(4,))
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x


class CameraBEVModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        self.resize = config['encoder']['resize']
        if config['encoder']['resize'] > 0:
            self.resize_conv = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0,
                                                  output_padding=0)
            # self.resize_conv = transforms.Resize(size=(self.resize , self.resize ))

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        # transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        camera_features = self.encoder(x)
        batch_dict.update({'features': camera_features})
        camera_features = self.fax(batch_dict)

        # B*L, C, H, W
        camera_features = camera_features.squeeze(1)

        if self.resize > 0:
            b, c, h, w = camera_features.shape
            # camera_features = self.resize_conv(camera_features.reshape(b * c, h, w)).reshape((b, c, self.resize, self.resize))
            camera_features = self.resize_conv(camera_features)

        # compressor
        if self.compression:
            camera_features = self.naive_compressor(camera_features)
        # Reformat to (B, max_cav, C, H, W)
        camera_features, mask = regroup(camera_features, record_len, self.max_cav)
        return camera_features, mask


class LidarBEVModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_cav = config['max_cav']

        self.lidar_encoder = build_encoder(config['lidar_encoder'])

    def forward(self, batch_dict):
        '''
        lidar data:
            data --> pillar_layer and pillar_encoder --> (b x l, 64, 496, 432)
                --> LidarEncoder (BackBone) --> (b x l, 512, 32, 32)- -> (b , l, 512, 32, 32)
        Returns (b , l, 512, 32, 32)
        -------
        '''
        record_len = batch_dict['record_len']
        lidar_data = batch_dict['lidar_data']

        lidar_features = []
        for agent, data in lidar_data.items():
            lidar_features_single = self.lidar_encoder(batch_inputs_dict={'points': [data]},
                                                       batch_data_samples=[])  # {'box_type_3d': LiDARInstance3DBoxes}
            # lidar_features_single = self.lidar_encoder(batch_inputs_dict={'points': [data]}, batch_data_samples=[])
            lidar_features.append(lidar_features_single[0])
        lidar_features = torch.cat(lidar_features, dim=0)
        # Reformat lidar data to (B, max_cav, C, H, W)
        lidar_features, mask_lidar = regroup(lidar_features, record_len, self.max_cav)

        return lidar_features, mask_lidar


class CorpBEVFusionLidarDetection(nn.Module):
    def __init__(self, config):
        super(CorpBEVFusionLidarDetection, self).__init__()

        self.max_cav = config['max_cav']

        self.camera_bev_model = CameraBEVModel(config)
        self.lidar_bev_model = LidarBEVModel(config)

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_image_net = SwapFusionEncoder(config['fax_image_fusion'])
        self.fusion_lidar_net = SwapFusionEncoder(config['fax_lidar_fusion'])

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

        self.fuse_method = config['fuse']['fuse_method']
        self.fuse_type = config['fuse']['fuse_type']

        self.trans_conv = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)

        if self.fuse_method == 'conv':
            self.fuse_modality = ConvFuser(
                in_channels=[config['fax_image_fusion']['input_dim'], config['fax_lidar_fusion']['input_dim']],
                out_channels=config['fuse']['fuse_out_channels'])
        if self.fuse_method == 'add':
            self.fuse_modality = AddFuser(
                in_channels=[config['fax_image_fusion']['input_dim'], config['fax_lidar_fusion']['input_dim']],
                out_channels=config['fuse']['fuse_out_channels'], dropout=0)

        self.head_type = config['head']['head_type']
        self.cam_conv = nn.Conv2d(in_channels=128, out_channels=384, kernel_size=1)
        self.lidar_conv = nn.ConvTranspose2d(in_channels=384, out_channels=512, kernel_size=4, stride=2, padding=1)
        if self.head_type == 'det':
            self.bevfusion_det_head = init_encoder(config['head']['bevfusion_det_head'])
        # decoder params
        decoder_params = config['decoder'][self.fuse_type]
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.cls_head = nn.Conv2d(128 * 3, config['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * config['anchor_number'],
                                  kernel_size=1)

        self.cam_cls_head = nn.Conv2d(128, config['anchor_number'],
                                  kernel_size=1)
        self.cam_reg_head = nn.Conv2d(128, 7 * config['anchor_number'],
                                  kernel_size=1)




    def forward(self, batch_dict):
        transformation_matrix = batch_dict['transformation_matrix']

        # visualize
        # self.visualize_pcd_img(batch_dict)

        if self.fuse_type == 'only_camera':
            camera_features, mask = self.camera_bev_model(batch_dict)
            camera_features = self.sttf(camera_features, transformation_matrix)
            camera_features = self.fuse_agent(camera_features, mask, transformation_matrix, self.fusion_image_net)

            features = camera_features
            features = self.decoder(features)
            features = features.squeeze(1)
            psm = self.cam_cls_head(features)
            rm = self.cam_reg_head(features)
            output_dict = {'psm': psm,
                           'rm': rm}
            return output_dict

        elif self.fuse_type == 'only_lidar':
            lidar_features, mask_lidar = self.lidar_bev_model(batch_dict)
            lidar_features = self.sttf(lidar_features, transformation_matrix)
            lidar_features = self.fuse_agent(lidar_features, mask_lidar, transformation_matrix, self.fusion_lidar_net)

            lidar_features = self.decoder(lidar_features)

            lidar_features = lidar_features.squeeze(1)
            psm = self.cls_head(lidar_features)
            rm = self.reg_head(lidar_features)

            output_dict = {'psm': psm,
                           'rm': rm}

            return output_dict

            # # bevfusion decoder
            # # features = lidar_features
            # batch_data_samples = self.get_batch_data_samples(batch_dict)
            # lidar_features = rearrange(lidar_features, 'b l c h w -> (b l) c h w')
            # # lidar_features = self.lidar_conv(lidar_features)
            # batch_data_samples, bbox_head = self.bevfusion_det_head(lidar_features, batch_data_samples)
            #
            # #loss
            # loss_dict = bbox_head.loss(lidar_features, batch_data_samples)
            # output_dict = {'loss_dict': loss_dict,
            #                'pred_instances_3d': batch_data_samples[0].pred_instances_3d,
            #                'gt_instances_3d': batch_data_samples[0].gt_instances_3d}
            #
            # # det_loss = loss_dict['layer_-1_loss_bbox']
            # # det_loss.backward()

            # features = lidar_features
            batch_data_samples = self.get_batch_data_samples(batch_dict)
            lidar_features = rearrange(lidar_features, 'b l c h w -> (b l) c h w')
            lidar_features = self.lidar_conv(lidar_features)
            batch_data_samples, bbox_head = self.bevfusion_det_head(lidar_features, batch_data_samples)

            #loss
            loss_dict = bbox_head.loss(lidar_features, batch_data_samples)
            output_dict = {'loss_dict': loss_dict,
                           'pred_instances_3d': batch_data_samples[0].pred_instances_3d,
                           'gt_instances_3d': batch_data_samples[0].gt_instances_3d}

            # det_loss = loss_dict['layer_-1_loss_bbox']
            # det_loss.backward()



            return output_dict




        else:
            camera_features, mask = self.camera_bev_model(batch_dict)
            camera_features = self.sttf(camera_features, transformation_matrix)
            lidar_features, mask_lidar = self.lidar_bev_model(batch_dict)
            lidar_features = self.sttf(lidar_features, transformation_matrix)
            if self.fuse_type == 'modality_after_agent':
                camera_features = self.fuse_agent(camera_features, mask, transformation_matrix, self.fusion_image_net)
                camera_features = torch.squeeze(camera_features, dim=1)
                camera_features = self.trans_conv(camera_features)
                lidar_features = self.fuse_agent(lidar_features, mask_lidar, transformation_matrix,
                                                 self.fusion_lidar_net)
                lidar_features = torch.squeeze(lidar_features, dim=1)

                if self.fuse_method == 'conv':
                    features = self.fuse_modality([camera_features, lidar_features])
                elif self.fuse_method == 'add':
                    features = self.fuse_modality([camera_features, lidar_features])
                else:
                    raise NotImplementedError

            elif self.fuse_type == 'modality_before_agent':
                # (B, max_cav, C, H, W)
                camera_features_list = []
                for i in range(self.max_cav):
                    camera_features_, lidar_features_ = camera_features[:, i], lidar_features[:, i]
                    camera_features_ = rearrange(camera_features_, 'b h w c -> b c h w')
                    lidar_features_ = rearrange(lidar_features_, 'b h w c -> b c h w')
                    if self.fuse_method == 'conv':
                        camera_features_ = self.fuse_modality([camera_features_, lidar_features_])
                    if self.fuse_method == 'add':
                        camera_features_ = self.fuse_modality([camera_features_, lidar_features_])

                    camera_features_list.append(camera_features_)
                camera_features = torch.stack(camera_features_list, dim=1)
                camera_features = rearrange(camera_features, 'b max_cav c h w -> b max_cav h w c')
                camera_features = self.fuse_agent(camera_features, mask, transformation_matrix, self.fusion_lidar_net)
                features = torch.squeeze(camera_features, dim=1)
            else:
                raise NotImplementedError

            features = torch.unsqueeze(features, dim=1)
            # lidar_features = torch.unsqueeze(lidar_features, dim=1)

        if self.head_type == 'seg':
            # dynamic segmentation head
            features = self.decoder(features)
            features = rearrange(features, 'b l c h w -> (b l) c h w')
            b = features.shape[0]
            output_dict = self.seg_head(features, b, 1)
            resize = transforms.Resize(size=(256, 256))
            for key in output_dict.keys():
                b, l, c, h, w = output_dict[key].shape
                output_dict[key] = resize(output_dict[key].reshape(b * l * c, h, w)).reshape((b, l, c, 256, 256))
            return output_dict

        elif self.head_type == 'det':
            # detection head returns 3d bbox
            features = rearrange(features, 'b l c h w -> (b l) c h w')
            features = self.cam_conv(features)
            detection_outputs = self.bevfusion_det_head(features,
                                                        batch_data_samples=[])  # {'box_type_3d': LiDARInstance3DBoxes}
            output_dict = {}
            output_dict['labels_3d'] = detection_outputs[0].labels_3d
            output_dict['bboxes_3d'] = detection_outputs[0].bboxes_3d.tensor
            output_dict['scores_3d'] = detection_outputs[0].scores_3d
            return output_dict

    def fuse_agent(self, feature, mask, transformation_matrix, fusion_net):
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(feature.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        feature = rearrange(feature, 'b l h w c -> b l c h w')
        feature = fusion_net(feature, com_mask)
        feature = feature.unsqueeze(1)
        return feature

    def get_batch_data_samples(self, batch_dict):

        batch_data_samples = []
        batch_size = batch_dict['gt_box3d'].shape[0]
        device = batch_dict['gt_box3d'].device

        for i in range(batch_size):
            # initiialize data_sample
            data_sample = Det3DDataSample()

            # add meta_info to data_sample
            meta_info = {}
            meta_info['box_type_3d'] = LiDARInstance3DBoxesVelocity
            data_sample.set_metainfo(meta_info)

            # add gt_instances_3d to data_sample
            gt_instance = InstanceData()
            gt_boxes = batch_dict['gt_box3d'][i][batch_dict['gt_box3d_mask'][i] == 1].to(device)
            gt_instance.bboxes_3d = LiDARInstance3DBoxesVelocity(gt_boxes)
            gt_instance.labels_3d = torch.zeros(gt_instance.bboxes_3d.shape[0], dtype=torch.long).to(device)
            data_sample.gt_instances_3d = gt_instance

            batch_data_samples.append(data_sample)

        return batch_data_samples