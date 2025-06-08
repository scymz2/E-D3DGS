import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HexPlaneField(nn.Module):
    """多分辨率 2‑D 平面，在 3D+Time 空间做特征插值。"""

    def __init__(
        self,
        bound: float,
        plane_cfg: dict,
        multires: Sequence[int],  # e.g. [1,2,4]
    ) -> None:
        super().__init__()
        # aabb
        xyz_max = torch.tensor([ bound,  bound,  bound])
        xyz_min = torch.tensor([-bound, -bound, -bound])
        self.register_buffer("aabb_max", xyz_max)
        self.register_buffer("aabb_min", xyz_min)

        # meta
        self.grid_nd   = plane_cfg["grid_dimensions"]      # usually 2
        self.in_dim    = plane_cfg["input_coordinate_dim"] # 4 (x,y,z,t)
        self.out_dim   = plane_cfg["output_coordinate_dim"]
        base_reso      = plane_cfg["resolution"]           # [64,64,64,25]
        assert len(base_reso) == self.in_dim

        # 预生成所有 2‑D 维度组合，例如 (0,1)、(0,2)…
        self.coo_combs: List[Tuple[int, int]] = list(itertools.combinations(range(self.in_dim), self.grid_nd))
        n_planes = len(self.coo_combs)

        # ======================== 生成多尺度平面参数 =========================
        self.grids: nn.ModuleList[nn.ParameterList] = nn.ModuleList()
        self.concat_features = True
        self.feat_dim = 0

        for s in multires:
            # 只缩放空间 xyz 分辨率，时间分辨率保持不变
            reso = [r * s for r in base_reso[:3]] + base_reso[3:]
            gp = self._init_grid_param(self.grid_nd, self.in_dim, self.out_dim, reso)
            self.grids.append(gp)
            self.feat_dim += gp[0].shape[1]  # 因为 concat

    # ------------------------------------------------------------------
    # grid helpers
    # ------------------------------------------------------------------
    def _init_grid_param(
        self, grid_nd: int, in_dim: int, out_dim: int, reso: Sequence[int]
    ) -> nn.ParameterList:
        """生成一组 2‑D 平面权重 (未加 batch 维)。"""
        has_time = in_dim == 4
        planes = nn.ParameterList()
        for coo in self.coo_combs:
            p = nn.Parameter(torch.empty(1, out_dim, *[reso[c] for c in coo[::-1]]))
            if has_time and 3 in coo:
                nn.init.ones_(p)          # time planes neutral @ init
            else:
                nn.init.uniform_(p, a=0.1, b=0.5)
            planes.append(p)
        return planes
    
    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # pts:(N,3) t:(N,1)
        """批量查询；返回 (N, feat_dim)。"""
        # 1) 归一化到 [-1,1]
        pts_n = (pts - self.aabb_min) * (2.0 / (self.aabb_max - self.aabb_min)) - 1.0
        x = torch.cat((pts_n, t), dim=-1)           # (N,4)

        # 2) 多尺度插值
        feats: List[torch.Tensor] = []
        for grid_set in self.grids:                # 各分辨率
            prod_feat = 1.0
            for plane, coo in zip(grid_set, self.coo_combs):
                # build coords
                coord = x[:, coo].unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
                interp = F.grid_sample(
                    plane, coord, mode="bilinear", align_corners=True, padding_mode="border"
                ).squeeze().T  # (N, out_dim)
                prod_feat = prod_feat * interp               # 逐平面乘积
            feats.append(prod_feat)
        return torch.cat(feats, dim=-1)                       # (N, feat_dim)