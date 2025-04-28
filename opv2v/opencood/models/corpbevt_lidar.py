"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import *
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.heads.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from opencood.models.pointpillars.pillars_feature_net import *
from opencood.models.fusion_modality.modality_fuser import *

#shilpa apptainer
import sys
sys.path.append('/home/shilpa/autoRL/CoBEVT_AutoNet')
from mmdetection3d.projects.BEVFusion.demo.init_encoder import build_encoder #init_encoder



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


class CorpBEVTLidar(nn.Module):
    def __init__(self, config):
        super(CorpBEVTLidar, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # self.lidar_encoder = LidarResnetEncoder(config['lidar_encoder'])

        # # default parameters make sure to include into CorpBEVT config later
        # point_cloud_range =[0, -39.68, -3, 69.12, 39.68, 1]
        voxel_size=[0.16, 0.16, 4]
        #
        point_cloud_range = [0, -40.96/2, -3, 40.96, 40.96/2, 1]
        # # voxel_size = [0.32*2, 0.32*2, 4]
        #
        max_num_points=32
        max_voxels=(16000, 40000)
        in_channel = 9
        out_channel = 64
        #
        # # Point Pillar layer and encoder
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, point_cloud_range=point_cloud_range,max_num_points=max_num_points, max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, point_cloud_range=point_cloud_range, in_channel=in_channel, out_channel=out_channel)
        # self.lidar_encoder = LidarEncoder()

        self.lidar_encoder = build_encoder() #init_encoder()
        print(self.lidar_encoder)

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = SwapFusionEncoder(config['fax_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

    def forward(self, batch_dict):
        lidar_data = batch_dict['lidar_data']
        # b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        # x = self.encoder(x)
        # batch_dict.update({'features': x})
        # x = self.fax(batch_dict)

        # # B*L, C, H, W
        # x = x.squeeze(1)

        # # compressor
        # if self.compression:
        #     x = self.naive_compressor(x)

        # # Reformat to (B, max_cav, C, H, W)
        # x, mask = regroup(x, record_len, self.max_cav)
        # # perform feature spatial transformation,  B, max_cav, H, W, C
        # x = self.sttf(x, transformation_matrix)

        lidar_features = []
        #shilpa lidar
        # for agent, data in lidar_data.items():
        for data in lidar_data:
            pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts=[data])
            lidar_features_single = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
            lidar_features.append(lidar_features_single)
        lidar_features = torch.cat(lidar_features, dim=0)
        # Reformat lidar data to (B, max_cav, C, H, W)
        lidar_features, mask_lidar = regroup(lidar_features, record_len, self.max_cav)
        # Lidar encoder
        lidar_features = self.lidar_encoder(lidar_features)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        lidar_features = self.sttf(lidar_features, transformation_matrix)


        com_mask = mask_lidar.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(lidar_features.shape,
                                      mask_lidar,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        lidar_features = rearrange(lidar_features, 'b l h w c -> b l c h w')
        lidar_features = self.fusion_net(lidar_features, com_mask)
        lidar_features = lidar_features.unsqueeze(1)

        # dynamic head
        lidar_features = self.decoder(lidar_features)
        lidar_features = rearrange(lidar_features, 'b l c h w -> (b l) c h w')
        b = lidar_features.shape[0]
        output_dict = self.seg_head(lidar_features, b, 1)

        return output_dict