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


class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    # def forward(self, x, spatial_correction_matrix):
    #     """
    #     Transform the bev features to ego space.

    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         B L C H W
    #     spatial_correction_matrix : torch.Tensor
    #         Transformation matrix to ego

    #     Returns
    #     -------
    #     The bev feature same shape as x but with transformation
    #     """
    #     dist_correction_matrix = get_discretized_transformation_matrix(
    #         spatial_correction_matrix, self.discrete_ratio,
    #         self.downsample_rate)

    #     # transpose and flip to make the transformation correct
    #     x = rearrange(x, 'b l c h w  -> b l c w h')
    #     x = torch.flip(x, dims=(4,))
    #     # Only compensate non-ego vehicles
    #     B, L, C, H, W = x.shape

    #     T = get_transformation_matrix(
    #         dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
    #     cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
    #                                (H, W))
    #     cav_features = cav_features.reshape(B, -1, C, H, W)

    #     # flip and transpose back
    #     x = cav_features
    #     x = torch.flip(x, dims=(4,))
    #     x = rearrange(x, 'b l c w h -> b l h w c')

    #     return x
    


    #shilpa sttf find transformed location
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

        # Generate the spatial grid (H x W)
        h_coords, w_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        spatial_grid = torch.stack([h_coords, w_coords, torch.ones_like(h_coords)], dim=-1)  # Shape: (H, W, 3)

        # Flatten the grid
        spatial_grid_flat = spatial_grid.view(-1, 3)  # Shape: (H*W, 3)

        # Repeat the grid for batch and layer dimensions
        spatial_grid_flat = spatial_grid_flat.unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1).to(T.device)  # Shape: [B, L, H*W, 3]

        # Reshape T to match batch and layer dimensions
        T = T.view(B, L, 2, 3)  # Shape: [B, L, 2, 3]

        # Perform batched matrix multiplication for each batch and layer
        transformed_grid_flat = torch.einsum('blmj,blkj->blmk', spatial_grid_flat.float(), T.float())  # Shape: [B, L, H*W, 2]

        # Reshape back to grid form
        transformed_grid = transformed_grid_flat.view(B, L, H, W, 2)  # Shape: [B, L, H, W, 2]

        # Clamp the transformed coordinates to valid ranges
        transformed_grid[..., 0] = torch.clamp(transformed_grid[..., 0], 0, H - 1)
        transformed_grid[..., 1] = torch.clamp(transformed_grid[..., 1], 0, W - 1)

        # Map to flattened indices if needed
        mapped_indices = (transformed_grid[..., 0] * W + transformed_grid[..., 1]).long()

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x, transformed_grid  # Return the transformed grid
    
    


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

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        #shilpa - SA and CA performed both
        #shilpa - bev is calculated inside fax, so need to get that in output, and send transformer matrix for sttf inside
        # x = self.fax(batch_dict)

        # # B*L, C, H, W
        # x = x.squeeze(1)

        #shilpa new fax - ca + egosend + cav reconstruction phase 1
        x, orig_bev_data_from_all_cav, selected_indices = self.fax(batch_dict, transformation_matrix)

        #shilpa new fax - individual sttf at cav
        # Extract the number of valid CAV entries
        valid_cav_count = x.shape[0]
        # Select only the valid transformation matrices (first valid_cav_count entries)
        valid_transformation_matrices = transformation_matrix[0, :valid_cav_count, :, :]  # Shape: [valid_cav_count, 4, 4]
        # Compute the inverse for each valid transformation matrix
        valid_ego_to_cav_matrices = torch.linalg.inv(valid_transformation_matrices)  # Shape: [valid_cav_count, 4, 4]
        # Create an output tensor with the same shape as the original transformation matrix
        ego_to_cav_transformation_matrices = torch.eye(4, device=transformation_matrix.device).repeat(1, transformation_matrix.shape[1], 1, 1)  # Shape: [1, max_cav, 4, 4]
        # Fill in the valid inverses
        # ego_to_cav_transformation_matrices[0, :valid_cav_count, :, :] = valid_ego_to_cav_matrices
        ego_to_cav_transformation_matrices = ego_to_cav_transformation_matrices.clone()
        ego_to_cav_transformation_matrices[0, :valid_cav_count, :, :] = valid_ego_to_cav_matrices
        
        # B*L, C, H, W
        x = x.squeeze(1)
        # Reformat to (B, max_cav, C, H, W)
        x, _ = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        x, transformed_grid = self.sttf(x, ego_to_cav_transformation_matrices)
        x = rearrange(x, 'b l h w c -> b l c h w')

        #shilpa reconstruct bev at indiv cav with received + own data
        # Create a mask for valid points (where the transformed coordinates are within bounds)
        H = x.shape[3]
        W = x.shape[4]
        
        # Replace non-transformed points with zeros (or another value)
        output = torch.zeros_like(x)  # Create a tensor of the same shape as x, filled with zeros
        # valid_mask_expanded = valid_mask.unsqueeze(2).expand(-1, -1, x.shape[2], -1, -1)  # Shape: [1, 5, 128, 32, 32]
                
        # Step 5: Replace Original BEV Data for All Points
        # n is the number of entries, c=128, h=32, w=32
        n, c, h, w = orig_bev_data_from_all_cav.shape
        max_cav = x.shape[1]  # max_cav = 5 (from x.shape)

        # Create an output tensor with the same shape as x, filled with zeros initially
        output = torch.zeros_like(x)

        # Directly copy the original BEV data into the first n entries of the output tensor
        # output[:, :n, :, :, :] = orig_bev_data_from_all_cav.unsqueeze(0).expand(output.shape[0], -1, -1, -1, -1)
        output = output.clone()  # Clone the tensor to avoid modifying the original
        output[:, :n, :, :, :] = orig_bev_data_from_all_cav.unsqueeze(0).expand(output.shape[0], -1, -1, -1, -1)
        
        
        # Fill the remaining entries (if n < max_cav) with the identity tensor or zeros
        if n < max_cav:
            identity_tensor = torch.eye(h, w, device=x.device).unsqueeze(0).repeat(c, 1, 1)  # Shape: [128, 32, 32]
            # output[:, n:, :, :, :] = identity_tensor.unsqueeze(0).unsqueeze(0).expand(output.shape[0], max_cav - n, c, h, w)
            output = output.clone()
            output[:, n:, :, :, :] = identity_tensor.unsqueeze(0).unsqueeze(0).expand(output.shape[0], max_cav - n, c, h, w)


        # Step 6: Flatten transformed_grid and output for easier indexing
        batch_size, max_cav, channels, height, width = output.shape
        flattened_transformed_grid = transformed_grid.view(batch_size, max_cav, -1, 2)  # Shape: [B, L, H*W, 2]
        flattened_output = output.reshape(batch_size, max_cav, channels, -1) # Shape: [B, L, C, H*W]

        # Step 8: Extract values from output using selected indices
        #shilpa Transmission 2 - this data is transmitted from CAV to ego for response
        selected_transformed_grid = flattened_transformed_grid[..., selected_indices, :]  # Shape: [B, L, 307, 2]
        # Initialize selected_output_values tensor
        batch_size, max_cav, num_selected, _ = selected_transformed_grid.shape  # [1, 5, 307, 2]
        channels = output.shape[2]  # Number of feature channels (128)

        # Create a tensor to hold the selected output values
        #shilpa Transmission 2 - this data is transmitted from CAV to ego for response
        selected_output_values = torch.zeros(batch_size, max_cav, channels, num_selected, device=output.device)  # [1, 5, 128, 307]

        # Iterate over batch and cav dimensions
        for b in range(batch_size):
            for l in range(max_cav):
                for i in range(num_selected):
                    x, y = selected_transformed_grid[b, l, i]  # Extract (x, y) coordinates
                    x = int(x)  # Convert to integer for indexing
                    y = int(y)  # Convert to integer for indexing
                    
                    # Check bounds to avoid indexing errors
                    if 0 <= x < output.shape[3] and 0 <= y < output.shape[4]:
                        # Fetch values from output at the transformed locations
                        # selected_output_values[b, l, :, i] = output[b, l, :, x, y]
                        selected_output_values = selected_output_values.clone()
                        selected_output_values[b, l, :, i] = output[b, l, :, x, y]

        #shilpa stack cav data at ego
        # Step 1: Extract cav_id=0 data
        cav_id_0_data = orig_bev_data_from_all_cav[batch_dict['ego_mat_index'][0]]  # Shape: [128, 32, 32]

        # Step 2: Replicate cav_id=0 data across all CAVs
        replicated_data = cav_id_0_data.unsqueeze(0).expand(orig_bev_data_from_all_cav.shape[0], -1, -1, -1)  # Shape: [5, 128, 32, 32]
        replicated_data = replicated_data.unsqueeze(0).expand(1, -1, -1, -1, -1)  # Shape: [1, 5, 128, 32, 32]
        
        # Step 3: Replace values at locations indicated by selected_transformed_grid
        batch_size, max_cav, num_selected, _ = selected_transformed_grid.shape  # [1, 5, 307, 2]
        channels = selected_output_values.shape[2]  # Number of feature channels (128)

        # Iterate over batch and cav dimensions to update replicated_data
        for b in range(batch_size):
            for l in range(orig_bev_data_from_all_cav.shape[0]):
                for i in range(num_selected):
                    x, y = selected_transformed_grid[b, l, i]  # Extract (x, y) coordinates
                    x = int(x)  # Convert to integer for indexing
                    y = int(y)  # Convert to integer for indexing
                    
                    # Check bounds to avoid indexing errors
                    if 0 <= x < replicated_data.shape[3] and 0 <= y < replicated_data.shape[4]:
                        # Replace the value in replicated_data with the corresponding value from selected_output_values
                        # replicated_data[b, l, :, x, y] = selected_output_values[b, l, :, i]
                        replicated_data = replicated_data.clone()
                        replicated_data[b, l, :, x, y] = selected_output_values[b, l, :, i]

        replicated_data = replicated_data.squeeze(0)
        x = self.fax.selfatt_module(replicated_data, b=batch_size, l=orig_bev_data_from_all_cav.shape[0])
                        
        #shilpa self attention with cav oriented all data

        # B*L, C, H, W
        #shilpa
        # x = x.squeeze(1)
        x = x.squeeze(0)

        # compressor
        #shilpa - to check during ablation study
        if self.compression:
            x = self.naive_compressor(x)

        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        #shilpa
        # x = self.sttf(x, transformation_matrix)
        x, _ = self.sttf(x, transformation_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        x = rearrange(x, 'b l h w c -> b l c h w')
        x = self.fusion_net(x, com_mask)
        x = x.unsqueeze(1)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        return output_dict