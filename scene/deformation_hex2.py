import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init

from scene.hexplane import HexPlaneField

def kaiming_init_weights(m):
        if isinstance(m, nn.Linear):
            # Use Kaiming Normal initialization, best for ReLU activations
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                # Initialize bias to zero
                nn.init.zeros_(m.bias)

def fourier_encode(x: torch.Tensor, freq_buf: torch.Tensor) -> torch.Tensor:
    """向量化的 Fourier 特征编码 (无 Python for‑loop). *x* 和 *freq_buf* 必须同设备/同 dtype"""
    # x : [N, C]      freq_buf : [F]
    # out: [N, C + 2*C*F]
    emb = (x.unsqueeze(-1) * freq_buf).flatten(-2)  # [N, C*F]
    return torch.cat([x, emb.sin(), emb.cos()], dim=-1)

class deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None) -> None:
        super().__init__()
        self.grid = HexPlaneField(1.6, {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 100]
    }, [1,2,4])  # 传入外部已构建、可复用的 HexPlaneField 实例
        self.args = args
        self.D, self.W = D, W

        # ======================== 频谱表（register_buffer 免显式 to(device)） ========================
        self.register_buffer("time_freq", 2 ** torch.arange(4))
        self.register_buffer("pos_freq", 2 ** torch.arange(10))
        self.register_buffer("rot_scale_freq", 2 ** torch.arange(2))

        # ======================== 时序嵌入  ========================
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        self.c2f_temporal_iter = args.c2f_temporal_iter # 渐进式训练参数

        if args.zero_temporal: # 时间嵌入
            # 零初始化的时间嵌入
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            # 正态分布初始化的时间嵌入，避免梯度爆炸或者消失
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))  # hard coded the upper limit of the num cameras (adjust as necessary)

    
        # ======================== MLP 结构 ========================
        self.mlp_c = self._make_mlp(self.grid.feat_dim, self.W, self.D)
        self.mlp_f = self._make_mlp(args.temporal_embedding_dim + args.gaussian_embedding_dim, W, D, residual=True)

        # 属性头
        self.pos_head_c = self._make_head(W, 3)
        self.scale_head_c = self._make_head(W, 3)
        self.rot_head_c = self._make_head(W, 4)
        self.opa_head_c = self._make_head(W, 1)
        self.rgb_head_c = self._make_head(W, 16 * 3)

        self.pos_head_f = self._make_head(W, 3, residual=True)
        self.scale_head_f = self._make_head(W, 3, residual=True)
        self.rot_head_f = self._make_head(W, 4, residual=True)
        self.opa_head_f = self._make_head(W, 1, residual=True)
        self.rgb_head_f = self._make_head(W, 16 * 3, residual=True)

        # ======================== 可选 Compile ========================
        if getattr(args, "use_torch_compile", False):
            self.forward = torch.compile(self.forward, fullgraph=False, mode="reduce-overhead")

    # -------------------------------------------------------------------------
    # 网络组件
    # -------------------------------------------------------------------------
    @staticmethod
    def _make_mlp(in_dim: int, hidden: int, depth: int, residual: bool = False) -> nn.Module:
        layers = [nn.Linear(in_dim, hidden)]
        for _ in range(depth - 1):
            layers += [nn.ReLU(), nn.Linear(hidden, hidden)]
        mlp = nn.Sequential(*layers)
        if residual:
            mlp.apply(kaiming_init_weights)
        return mlp   # keep interface identical

    @staticmethod
    def _make_head(hidden: int, out_dim: int, residual: bool = False) -> nn.Module:
        head = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )
        if residual:
            head.apply(kaiming_init_weights)
        return head
    
    # ---------------------- Temporal embedding ------------------------
    # def _temb_linear(self, t_norm: torch.Tensor, n_T: int) -> torch.Tensor:
    #     # 确保 t_norm 是 (N,) 而不是 (N, 1)
    #     if t_norm.dim() > 1:
    #         t_norm = t_norm.squeeze(-1)  # (N, 1) -> (N,)
        
    #     idx_f = t_norm.clamp(0, 1) * (n_T - 1)
    #     idx0 = torch.floor(idx_f).long()
    #     idx1 = (idx0 + 1).clamp(max=n_T - 1)
    #     w = (idx_f - idx0.float()).unsqueeze(-1)
    #     return self.weight[idx0] * (1 - w) + self.weight[idx1] * w
    
    def _temb_linear(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                size=(current_num_embeddings, self.temporal_embedding_dim), 
                                mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]
        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2
        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        
        emb = emb.repeat(1,1,N,1).squeeze()
        return emb
    
    

    # ---------------------------- forward ------------------------------
    def forward(
        self,
        pts: torch.Tensor,          # (N,3)
        scales: torch.Tensor,       # (N,3)
        rotations: torch.Tensor,    # (N,4) 四元数
        opacity: torch.Tensor,      # (N,1)
        time: torch.Tensor,         # (N,1) 已归一化到 [0,1]
        cam_no: int,         # 相机编号 (0-29)，用于时间偏移
        pc=None,                    # 兼容旧接口，可忽略
        gaussian_emb=None, # (N,G) (= SH 系数或其他高斯特征)
        sh_coefs=None,  # (N,16,3) SH 系数或其他高斯特征
        iter=None,
        num_down_emb_c=30,              # 粗网络降采样嵌入数
        num_down_emb_f=30              # 细网络降采样嵌入数
    ):
        # 确保输入是float32类型
        pts = pts[:, :3].float()
        scales = scales[:, :3].float()
        rotations = rotations[:, :4].float()
        opacity = opacity[:, :1].float()
        time = time.float()
        
        # --------------------------- 预处理 ----------------------------
        # 先保存一份「原始输入」备份
        pts0, scl0, rot0, opa0 = pts, scales, rotations, opacity
        sh0 = sh_coefs if sh_coefs is None else sh_coefs

        # fourier 编码
        pts_fourier = fourier_encode(pts, self.pos_freq)  # (N, 3 + 2*10)
        scales_fourier = fourier_encode(scales, self.pos_freq)  # (N, 3 + 2*10)
        rotations_fourier = fourier_encode(rotations, self.rot_scale_freq)  # (N, 4 + 2*2)

        # 相机时间偏移（若有）
        if cam_no is not None:
            time = time + self.offsets[cam_no]
        else:
            # 使用 offsets 中非零值的均值（若存在），否则不偏移
            non_zero = self.offsets != 0
            if non_zero.any():
                time = time + self.offsets[non_zero].mean()

        # --------------------------- 退火系数 ---------------------------
        use_anneal = getattr(self.args, "use_anneal", False)
        if not use_anneal:
            coef_main = coef_c = coef_o = coef_s = 1.0
        else:
            # 1 000 step 线性生效，可根据需要改成 args 中的常量
            coef_main = min(iter / 1_000, 1.0)
            start = getattr(self.args, "deform_from_iter", 0)
            k = max(iter - start, 0)
            coef_c = coef_o = coef_s = min(k / 1_000, 1.0)

        #自动混精度
        use_amp = getattr(self.args, "use_amp", True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            grid_feat = self.grid(pts, time)  # (N, feat_dim)
            hidden_c = self.mlp_c(grid_feat)      # (N, W)

            # ------------------ coarse deform -----------------------------
            # ---- 位置 ----
            dx  = self.pos_head_c(hidden_c)
            pts_c = pts + dx * coef_main

            # ---- 尺度 ----
            ds = self.scale_head_c(hidden_c) if not getattr(self.args, "no_ds", False) else 0.0
            scl_c = scales + ds * coef_main * coef_s

            # ---- 旋转 ----
            dr = self.rot_head_c(hidden_c) if not getattr(self.args, "no_dr", False) else 0.0
            rot_c = rotations + dr * coef_main

            # ---- 透明度 ----
            do = self.opa_head_c(hidden_c) if not getattr(self.args, "no_do", False) else 0.0
            opa_c = opacity + do * coef_main * coef_o

            # ---- SH/RGB ----
            dc = (
                self.rgb_head_c(hidden_c).view(-1, 16, 3)
                if not getattr(self.args, "no_dc", False)
                else 0.0
            )
            sh_c = sh0 + dc * coef_main * coef_c

            # ------------------ temporal embed (fine) ----------------------
            if getattr(self.args, "no_c2f_temporal", False):
                nT = self.args.max_embeddings
            else:
                # 渐进式
                nT = int(
                    self.args.min_embeddings
                    + (self.args.max_embeddings - self.args.min_embeddings)
                    * min(iter, self.c2f_temporal_iter)
                    / self.c2f_temporal_iter
                )
            t_emb = self._temb_linear(time, nT)                  # (N,tself.temporal_embedding_dim)

            if gaussian_emb is None:
                gaussian_emb = pc.get_embedding


            # -------------- fine deform ------------------------
            # ---- Fine MLP ----
            hidden_f = self.mlp_f(torch.cat([t_emb, gaussian_emb], dim=-1))

            # ---- Fine 形变 (带退火) ----
            pts_f = pts_c + self.pos_head_f(hidden_f) * coef_main
            scl_f = scl_c + (
                self.scale_head_f(hidden_f) * coef_main * coef_s
                if not getattr(self.args, "no_ds", False)
                else 0.0
            )
            rot_f = rot_c + (
                self.rot_head_f(hidden_f) * coef_main
                if not getattr(self.args, "no_dr", False)
                else 0.0
            )
            opa_f = opa_c + (
                self.opa_head_f(hidden_f) * coef_main * coef_o
                if not getattr(self.args, "no_do", False)
                else 0.0
            )
            sh_f = sh_c + (
                self.rgb_head_f(hidden_f).view(-1, 16, 3) * coef_main * coef_c
                if not getattr(self.args, "no_dc", False)
                else 0.0
            )

            # 返回三种变形后的点云数据：原始的点云数据和经过粗粒度、细粒度变形后的点云数据
            return pts0, scl0, rot0, opa0, sh0, \
                ((pts_c, scl_c, rot_c, opa_c, sh_c), \
                    (pts_f, scl_f, rot_f, opa_f, sh_f)) \
            
        
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list

