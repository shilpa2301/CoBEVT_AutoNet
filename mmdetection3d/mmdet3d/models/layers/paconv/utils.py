# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor


def calc_euclidian_dist(xyz1: Tensor, xyz2: Tensor) -> Tensor:
    """Calculate the Euclidean distance between two sets of points.

    Args:
        xyz1 (Tensor): (N, 3) The first set of points.
        xyz2 (Tensor): (N, 3) The second set of points.

    Returns:
        Tensor: (N, ) The Euclidean distance between each point pair.
    """
    assert xyz1.shape[0] == xyz2.shape[0], 'number of points are not the same'
    assert xyz1.shape[1] == xyz2.shape[1] == 3, \
        'points coordinates dimension is not 3'
    return torch.norm(xyz1 - xyz2, dim=-1)


def assign_score(scores: Tensor, point_features: Tensor) -> Tensor:
    """Perform weighted sum to aggregate output features according to scores.
    This function is used in non-CUDA version of PAConv.

    Compared to the cuda op assigh_score_withk, this pytorch implementation
    pre-computes output features for the neighbors of all centers, and then
    performs aggregation. It consumes more GPU memories.

    Args:
        scores (Tensor): (B, npoint, K, M) Predicted scores to
            aggregate weight matrices in the weight bank.
            `npoint` is the number of sampled centers.
            `K` is the number of queried neighbors.
            `M` is the number of weight matrices in the weight bank.
        point_features (Tensor): (B, npoint, K, M, out_dim)
            Pre-computed point features to be aggregated.

    Returns:
        Tensor: (B, npoint, K, out_dim) The aggregated features.
    """
    B, npoint, K, M = scores.size()
    scores = scores.view(B, npoint, K, 1, M)
    output = torch.matmul(scores, point_features).view(B, npoint, K, -1)
    return output


def assign_kernel_withoutk(features: Tensor, kernels: Tensor,
                           M: int) -> Tuple[Tensor]:
    """Pre-compute features with weight matrices in weight bank. This function
    is used before cuda op assign_score_withk in CUDA version PAConv.

    Args:
        features (Tensor): (B, in_dim, N) Input features of all points.
            `N` is the number of points in current point cloud.
        kernels (Tensor): (2 * in_dim, M * out_dim) Weight matrices in
            the weight bank, transformed from (M, 2 * in_dim, out_dim).
            `2 * in_dim` is because the input features are concatenation of
            (point_features - center_features, point_features).
        M (int): Number of weight matrices in the weight bank.

    Returns:
        Tuple[Tensor]: Both of shape (B, N, M, out_dim).

            - point_features: Pre-computed features for points.
            - center_features: Pre-computed features for centers.
    """
    B, in_dim, N = features.size()
    feat_trans = features.permute(0, 2, 1)  # [B, N, in_dim]
    out_feat_half1 = torch.matmul(feat_trans, kernels[:in_dim]).view(
        B, N, M, -1)  # [B, N, M, out_dim]
    out_feat_half2 = torch.matmul(feat_trans, kernels[in_dim:]).view(
        B, N, M, -1)  # [B, N, M, out_dim]

    # TODO: why this hard-coded if condition?
    # when the network input is only xyz without additional features
    # xyz will be used as features, so that features.size(1) == 3 % 2 != 0
    # we need to compensate center_features because otherwise
    # `point_features - center_features` will result in all zeros?
    if features.size(1) % 2 != 0:
        out_feat_half_coord = torch.matmul(
            feat_trans[:, :, :3],  # [B, N, 3]
            kernels[in_dim:in_dim + 3]).view(B, N, M, -1)  # [B, N, M, out_dim]
    else:
        out_feat_half_coord = torch.zeros_like(out_feat_half2)

    point_features = out_feat_half1 + out_feat_half2
    center_features = out_feat_half1 + out_feat_half_coord
    return point_features, center_features
