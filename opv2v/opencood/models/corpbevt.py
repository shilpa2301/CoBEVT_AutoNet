"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix

#shilpa bev dim match
# import torch.nn.functional as F

#shilpa entropy
from scipy.stats import entropy


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
     


class CorpBEVT(nn.Module):
    def __init__(self, config):
        super(CorpBEVT, self).__init__()
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

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])
        #shilpa
        # self.find_transformed_indices = STTF(config['sttf']).find_transformed_indices

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
        
        #shilpa entropy
        # self.prev_avg_entropy = None

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        #shilpa - SA and CA performed both
        # #shilpa - bev is calculated inside fax, so need to get that in output, and send transformer matrix for sttf inside
        # x = self.fax(batch_dict)

        # # B*L, C, H, W
        # x = x.squeeze(1)

        #shilpa new fax - ca + egosend + cav reconstruction phase 1
        #shilpa channel entropy
        # if self.prev_avg_entropy is not None:
        #     orig_bev_data_from_all_cav, selected_indices = self.fax(batch_dict, self.prev_avg_entropy)
        # else:
        #     orig_bev_data_from_all_cav, selected_indices = self.fax(batch_dict)
        
        #shilpa channel entropy soft
        # orig_bev_data_from_all_cav, selected_indices, normalized_uncertainty = self.fax(batch_dict)

        orig_bev_data_from_all_cav, selected_indices = self.fax(batch_dict)

        x = orig_bev_data_from_all_cav

        x, _ = regroup(x, record_len, self.max_cav)
        x = self.sttf(x, transformation_matrix)
        x = rearrange(x, 'b l h w c -> b l c h w')

        n, c, h, w = orig_bev_data_from_all_cav.shape
        max_cav = x.shape[1]  # max_cav = 5 (from x.shape)
        batch_size = x.shape[0]

        #shilpa channel entropy
        # x = x.reshape(batch_size, max_cav, c, -1)
        # selected_output_values = torch.zeros(batch_size, max_cav, c, selected_indices.shape[0], device=x.device) 
        # for idx, value in enumerate(selected_indices):
        #         # Use advanced indexing to copy values
        #         selected_output_values[:, :, :, idx] = x[:, :, :, value].clone()


        selected_output_values = torch.zeros(batch_size, max_cav, selected_indices.shape[0], h, w, device=x.device) 
        for idx, value in enumerate(selected_indices):
                # Use advanced indexing to copy values
                selected_output_values[:, :, idx, :,:] = x[:, :, value, :,:].clone()

        #response : selected_output_values
      

        #shilpa stack cav data at ego
        # Step 1: Extract cav_id=0 data
        # # print("device of orig_bev_data_from_all_cav=", orig_bev_data_from_all_cav.device)
        cav_id_0_data = orig_bev_data_from_all_cav[batch_dict['ego_mat_index'][0]]  # Shape: [128, 32, 32]

        #enable for fuse auto

        # # # Step 2: Replicate cav_id=0 data across all CAVs
        replicated_data = cav_id_0_data.unsqueeze(0).expand(n, -1, -1, -1)  # Shape: [5, 128, 32, 32]
        replicated_data = replicated_data.unsqueeze(0).expand(1, -1, -1, -1, -1)  # Shape: [1, 5, 128, 32, 32]

        #enable without cav0 in base

        # replicated_data = torch.zeros((n, c, h, w), device=x.device)  # Shape: [5, 128, 32, 32]
        # # Step 2: Replace the 0th index of the first dimension with cav_id_0_data
        # replicated_data[0] = cav_id_0_data
        # # Step 3: Expand to shape [1, 5, 128, 32, 32]
        # replicated_data = replicated_data.unsqueeze(0)
        
        # Step 3: Replace values at locations indicated by select_indices
        # Step 1: Reshape replicated_data to [1, k, 128, 1024]

        #shilpa channel entropy comment
        # replicated_data_flat = replicated_data.reshape(batch_size, n, c, h * w)

        # Step 2: Slice selected_output_values to only consider the first k slices
        selected_output_values_k = selected_output_values[:, :n, :, :]  # Shape: [1, k, 128, 307]

        # Step 3: Expand selected_indices to match the shape of selected_output_values_k
        #shilpa channel entropy
        # selected_indices_expanded = selected_indices.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # Shape: [1, 1, 1, 307]
        # selected_indices_expanded = selected_indices_expanded.expand(batch_size, n, c, selected_output_values.shape[3])  # Shape: [1, k, 128, 307]
        # selected_indices_expanded = selected_indices.unsqueeze(0).unsqueeze(0)
        # selected_indices_expanded = selected_indices_expanded.unsqueeze(-1).unsqueeze(-1)
        # selected_indices_expanded = selected_indices_expanded.expand(1, n, 128, h, w)

        
        #shilpa channel entropy
        # Step 4: Replace values in replicated_data_flat using scatter_
        # replicated_data_flat_clone = replicated_data_flat.clone()  # Create a copy of the tensor
        # replicated_data_flat_clone.scatter_(3, selected_indices_expanded, selected_output_values_k)
        # # Step 5: Reshape replicated_data_flat back to [k, 128, 32, 32]
        # replicated_data = replicated_data_flat_clone.view(n, c, h, w)
        
        replicated_data = replicated_data.clone()
        replicated_data[:, :, selected_indices, :, :] = selected_output_values_k[:, :, :len(selected_indices), :, :]
        replicated_data=replicated_data.squeeze(0)



        
        # print("device of replocated_data=", replicated_data.device)
        #shilpa transform sa fix
        x = replicated_data
        
        # compressor
        #shilpa - to check during ablation study
        if self.compression:
            x = self.naive_compressor(x)

        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        
        # perform feature spatial transformation,  B, max_cav, H, W, C
        #shilpa
        # # x = self.sttf(x, transformation_matrix)
        

        x = rearrange(x, 'b l c h w -> b l h w c')
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)
        
        
        # transformation_matrix_identity = torch.eye(4, device=transformation_matrix.device).repeat(1, transformation_matrix.shape[1], 1, 1)
        # com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
        #     3) if not self.use_roi_mask \
        #     else get_roi_and_cav_mask(x.shape,
        #                               mask,
        #                               transformation_matrix_identity,
        #                               self.discrete_ratio,
        #                               self.downsample_rate)

        # # fuse all agents together to get a single bev map, b h w c
        x = rearrange(x, 'b l h w c -> b l c h w')
        
    
        x = self.fusion_net(x, com_mask)
        x = x.unsqueeze(1)

        #shilpa channel entropy soft
        # b = cav_id_0_data.unsqueeze(0).unsqueeze(0)
        # x = (1-normalized_uncertainty)*b + normalized_uncertainty*x

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        #shilpa channel entropy
        # #shilpa entropy
        # # Normalize the tensor to get probabilities
        # if self.target == 'static':
        #     target_tensor = output_dict['static_seg'].detach().cpu().numpy()
        # else:
        #     target_tensor = output_dict['dynamic_seg'].detach().cpu().numpy()
            
        # # probabilities = target_tensor / target_tensor.sum(axis=2, keepdims=True)
        # # # Add a small epsilon to avoid log(0)
        # # epsilon = 1e-10
        # # probabilities += epsilon

        # # Convert logits to probabilities using softmax
        # exp_logits = np.exp(target_tensor - np.max(target_tensor, axis=2, keepdims=True))  # Stabilize for numerical safety
        # probabilities = exp_logits / exp_logits.sum(axis=2, keepdims=True)

        # # Add a small epsilon to avoid log(0)
        # epsilon = 1e-10
        # probabilities = np.clip(probabilities, a_min=epsilon, a_max=None)  # Ensure probabilities are valid

        # # Calculate entropy for each pixel using vectorized operations
        # entropy_map = entropy(probabilities, axis=2, base=2)
        # # Reshape and calculate average entropy over 32x32 buckets
        # entropy_map = entropy_map.squeeze()
        # bucket_size = 8
        # entropy_map_reshaped = entropy_map.reshape(256 // bucket_size, bucket_size, 256 // bucket_size, bucket_size)
        # # entropy_map_reshaped = entropy_map.reshape(bucket_size, 256//bucket_size, bucket_size, 256//bucket_size)
        # avg_entropy = entropy_map_reshaped.mean(axis=(1, 3))
        # #shilpa soft entropy
        # # Normalize avg_entropy between 0 and 1
        # min_entropy = avg_entropy.min()
        # max_entropy = avg_entropy.max()
        # normalized_avg_entropy = (avg_entropy - min_entropy) / (max_entropy - min_entropy)

        # # Reshape to desired output shape
        # # self.prev_avg_entropy = torch.tensor(avg_entropy.reshape(1, 1, 1, 32, 32))
        # self.prev_avg_entropy = torch.tensor(normalized_avg_entropy.reshape(1, 1, 1, 32, 32))



        return output_dict