import random
import numpy as np

import torch
import torch.nn.functional as F


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_knn_3d(xyz, kernel_size=5, knn=20):
    """ Use 3D Conv to compute neighbour distance and find k nearest neighbour
          xyz: (B, 3, D, H, W)

      Returns:
        idx: (B, D*H*W, k)
    """
    batch_size, _, depth, height, width = list(xyz.size())
    assert (kernel_size % 2 == 1)
    hk = (kernel_size // 2)
    k2 = kernel_size ** 2
    k3 = kernel_size ** 3

    t = np.zeros((kernel_size, kernel_size, kernel_size, 1, kernel_size ** 3))
    ind = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                t[i, j, k, 0, ind] -= 1.0
                t[hk, hk, hk, 0, ind] += 1.0
                ind += 1
    weight = np.zeros((kernel_size, kernel_size, kernel_size, 3, 3 * k3))
    weight[:, :, :, 0:1, :k3] = t
    weight[:, :, :, 1:2, k3:2 * k3] = t
    weight[:, :, :, 2:3, 2 * k3:3 * k3] = t
    weight = torch.tensor(weight).float()

    weights_torch = torch.Tensor(weight.permute((4, 3, 0, 1, 2))).to(xyz.device)
    dist = F.conv3d(xyz, weights_torch, padding=hk)

    dist_flat = dist.contiguous().view(batch_size, 3, k3, -1)
    dist2 = torch.sum(dist_flat ** 2, dim=1)

    _, nn_idx = torch.topk(-dist2, k=knn, dim=1)
    nn_idx = nn_idx.permute(0, 2, 1)
    d_offset = nn_idx // k2 - hk
    h_offset = (nn_idx % k2) // kernel_size - hk
    w_offset = nn_idx % kernel_size - hk

    idx = torch.arange(depth * height * width).to(xyz.device)
    idx = idx.view(1, -1, 1).expand(batch_size, -1, knn)
    idx = idx + (d_offset * height * width) + (h_offset * width) + w_offset

    idx = torch.clamp(idx, 0, depth * height * width - 1)

    return idx


def get_flat_nn(xyz, knn=16):
    batch_size, _, depth, height, width = xyz.shape
    assert depth == 1
    assert knn == 16
    h_offset = torch.tensor([0, 0, -1, 1, -1, -1, 1, 1, 0, 0, -2, 2, -1, -1, 1, 1]).long().to(xyz.device)
    w_offset = torch.tensor([-1, 1, 0, 0, -1, 1, -1, 1, -2, 2, 0, 0, -2, 2, -2, 2]).long().to(xyz.device)

    h_offset = h_offset.view(1, 1, -1).expand(batch_size, -1, knn)
    w_offset = w_offset.view(1, 1, -1).expand(batch_size, -1, knn)

    idx = torch.arange(depth * height * width).to(xyz.device)
    idx = idx.view(1, -1, 1).expand(batch_size, -1, knn)
    idx = idx + (width * h_offset) + w_offset

    idx = torch.clamp(idx, 0, depth * height * width - 1)

    return idx

def compute_visibility_mask(depth_map_i, depth_map_j, RT_i, RT_j, K_i, K_j, depth_threshold=2.0):
    """compute visibility mask of view i"""
    assert (depth_map_i.shape == depth_map_j.shape)
    height, width = depth_map_i.shape
    num_pts = height * width
    feature_grid = get_pixel_grids_np(depth_map_i.shape[0], depth_map_i.shape[1])
    uv = np.matmul(np.linalg.inv(K_i), feature_grid)
    gt_cam_points = uv * np.reshape(depth_map_i, (1, -1))
    R = RT_i[:3, :3]
    t = RT_i[:3, 3:4]
    R_inv = np.linalg.inv(R)
    gt_world_points = np.matmul(R_inv, gt_cam_points - t)
    R_j = RT_j[:3, :3]
    t_j = RT_j[:3, 3:4]
    gt_cam_points_j = np.matmul(R_j, gt_world_points) + t_j
    x = torch.tensor(gt_cam_points_j[0, :]).float()
    y = torch.tensor(gt_cam_points_j[1, :]).float()
    z = torch.tensor(gt_cam_points_j[2, :]).float()
    normal_uv = torch.stack([x / z, y / z, torch.ones_like(x)])
    uv = torch.matmul(torch.tensor(K_j).float(), normal_uv)
    uv = (uv[:2, :]).transpose(0, 1)
    grid = (uv - 0.5).view(1, num_pts, 1, 2)
    grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
    grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

    grid[:, z < 0, ...] = 100
    grid[torch.isnan(grid)] = 100
    depth_j_tensor = torch.tensor(depth_map_j).float().view(1, 1, height, width)
    fetch_depth = F.grid_sample(depth_j_tensor, grid, mode="nearest")
    fetch_depth = fetch_depth.view(height, width).numpy()
    proj_depth = z.view(height, width).numpy()

    visibility_mask = (np.abs(fetch_depth - proj_depth) < depth_threshold) * (proj_depth > 0) * (fetch_depth > 0)
    return visibility_mask
