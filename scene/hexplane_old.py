import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1] # 取出最后一维，这里是2，因为坐标是2D的

    if grid.dim() == grid_dim + 1: # 如果grid没有batch维度，则添加一个batch维度, 就像 out_dim, res_y, res_x] -> [1, out_dim, res_y, res_x]
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0) # 在第零维添加一个维度
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3: # 只支持2D或者3D的插值，如果不是，则报错
        grid_sampler = F.grid_sample # 这是python的内置函数， 用于在给定的网格上进行双线性插值
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int, # 平面的维度
        in_dim: int,  # 输入的总维度（如4， 表示x,y,z,t）
        out_dim: int, # 每个平输出的特征维度
        reso: Sequence[int], # 每个维度的分辨率 （如[64,64,64,25]）
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd)) # 例如 in_dim=4, grid_nd=2，则组合有 [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]，即 xy、xz、xt、yz、yt、zt。
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        # 创建一个未初始化的张量，形状为[1, out_dim, res_y, res_x]， 所以对于每个平面，每个像素位置学习32维
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]] # [1, out_dim, res_y, res_x]， coo_comb[::-1]表示反转顺序, 会得到(1,0), (2,0), (3,0)等等，然后从reso中取出对应的分辨率，先y后x
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            # 在初始化时，模型还没有学习到任何时间相关的特征，因此我们希望时间平面对特征的影响是“中性”的，因此初始化为1
            nn.init.ones_(new_grid_coef) # 如果有时间平面，且当前平面是时间平面，则初始化为1，
        else:
            # 空间平面是用来捕捉点云在空间维度上的静态特征， 空间特征通常具有较大的变化范围，因此需要通过随机初始化来提供多样性，避免初始特征过于平滑
            nn.init.uniform_(new_grid_coef, a=a, b=b) # 如果是空间平面，则初始化为均匀分布
        grid_coefs.append(new_grid_coef) # 6 * [1, out_dim, res_y, res_x]

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor, # [batch_size, num_points, 4]， 4表示x,y,z,t
                            ms_grids: Collection[Iterable[nn.Module]],  # 计算得到的多分辨率平面 [num_res * num_coo_comb = 4 * 6，1, out_dim, res_y, res_x]
                            grid_dimensions: int,  # 2
                            concat_features: bool, # 是否拼接特征
                            num_levels: Optional[int], # None
                            ) -> torch.Tensor:
    # 1. 生成所有2D平面组合（如xy、xz、xt、yz、yt、zt）
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    # 2. 如果没有指定num_levels，则用所有分辨率
    if num_levels is None:
        num_levels = len(ms_grids)
    # 3. 根据是否拼接特征，初始化输出变量
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList  # 类型注释，对实际运行没有影响
    # 4. 遍历每个分辨率（多尺度）
    for scale_id,  grid in enumerate(ms_grids[:num_levels]): # grid: [1, out_dim, res_y, res_x]
        interp_space = 1.
        # 5. 遍历每个2D平面
        for ci, coo_comb in enumerate(coo_combs): # [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            # 6. 在当前平面上插值，得到每个点的特征
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            # 7. 融合所有平面特征（乘积）
            interp_space = interp_space * interp_out_plane

        # combine over scales
        # 8. 多分辨率特征融合
        if concat_features:
            multi_scale_interp.append(interp_space) # 这里只是append了当前分辨率下的特征，还没有拼接
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    # 9. 如果拼接特征，把所有分辨率的特征拼接起来
    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig, # {'grid_dimensions': 2,'input_coordinate_dim': 4,'output_coordinate_dim': 32,'resolution': [64, 64, 64, 25]  # resolution of spatial grid and temporal grid, better to be half length of dynamic frames }
        multires # [1,2,4,8] 通过multires, 每个分辨率初始化一组2D平面， 如xy, xz, xt等
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False) # Parameter相比于tensor, 会被optimizer自动识别，具有更好的可训练性
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires # [1,2,4,8] 通过multires, 每个分辨率初始化一组2D平面， 如xy, xz, xt等
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList() # ModuleList是一个特殊的list, 具有更好的可训练性， 保存所有分辨率下的所有平面
        self.feat_dim = 0
        '''
        创建多分辨率的平面， 对于每个multires(多分辨率的缩放因子列表), 每套的分辨率是resolution的基础上乘以不同的缩放因子。
        [64*1, 64*1, 64*1, 25] → [64, 64, 64, 25]
        [64*2, 64*2, 64*2, 25] → [128, 128, 128, 25]
        [64*4, 64*4, 64*4, 25] → [256, 256, 256, 25]
        [64*8, 64*8, 64*8, 25] → [512, 512, 512, 25]
        '''
        for res in self.multiscale_res_multipliers: # res是缩放因子
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1] # 加上每个像素的特征维度out_dim
            else:
                self.feat_dim = gp[-1].shape[1] # 如果不拼接，则只取最后一个平面的特征维度
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb) # 先将输入点归一化到[-1,1]之间
        pts = torch.cat((pts, timestamps), dim=-1)  # 把时间戳拼接到空间坐标后面，得到4D输入

        pts = pts.reshape(-1, pts.shape[-1]) # 任意批量、任意采样数的空间+时间输入，统一展平成 [N, 4]，方便后续批量插值处理。
        features = interpolate_ms_features( # 多尺度插值，在所有分辨率所有2D平面上进行插值
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1: # 如果没有特征，则返回空张量
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
