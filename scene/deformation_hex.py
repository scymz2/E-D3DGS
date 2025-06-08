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

from torch.profiler import profile, record_function, ProfilerActivity


def kaiming_init_weights(m):
        """
        Applies He Kaiming normal initialization to linear layers.
        """
        if isinstance(m, nn.Linear):
            # Use Kaiming Normal initialization, best for ReLU activations
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                # Initialize bias to zero
                nn.init.zeros_(m.bias)

def poc_fre(input_data,poc_buf):
        '''
        对输入（坐标，尺度，旋转，时间等）做fourier特征编码
        '''
        input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
        input_data_sin = input_data_emb.sin()
        input_data_cos = input_data_emb.cos()
        input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
        return input_data_emb

class deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(deform_network, self).__init__()
        self.D = D
        self.W = W

        # 4DGaussian components
        self.grid = HexPlaneField(1.6, {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 100]
    }, [1,2,4])
        
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(4)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(10)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(2)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(2)]))

        self.args = args
        self.min_embeddings = min_embeddings # 最小的时间嵌入数量
        self.max_embeddings = max_embeddings # 最大的时间嵌入数量
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        self.c2f_temporal_iter = args.c2f_temporal_iter # 渐进式训练参数


        # 以下网络包含位置变形（三维）， 尺度变形（三维）， 旋转变形（四维四元数）， 不透明度变形（一维）， RGB变形（四十八维， 3*16球谐函数）
        # 粗粒度网络（coarse）
        self.feature_out_c, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c = self.create_net_hex()
        # 细粒度网络（fine）
        self.feature_out_f, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f = self.create_net(is_residual=True)

        if args.zero_temporal: # 时间嵌入
            # 零初始化的时间嵌入
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            # 正态分布初始化的时间嵌入，避免梯度爆炸或者消失
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))  # hard coded the upper limit of the num cameras (adjust as necessary)


    def create_net_hex(self):
        mlp_out_dim = 0
        grid_out_dim = self.grid.feat_dim
        self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]

        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        feature_out = nn.Sequential(*self.feature_out)

        return \
            feature_out,\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

        
        

    def create_net(self, is_residual=False):
        # 初始化一个包含单个线性层的列表，这个线性层的作用是将时间嵌入和高斯嵌入的维度组合并映射到W维度
        self.feature_out = [nn.Linear(self.temporal_embedding_dim + self.gaussian_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        feature_out = nn.Sequential(*self.feature_out) # 列表转为顺序容器, Sequential可以将多个层组合成一个层，层中自动执行forward
        
        if is_residual:
            return \
                feature_out.apply(kaiming_init_weights),\
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)).apply(kaiming_init_weights),\
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)).apply(kaiming_init_weights),\
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)).apply(kaiming_init_weights), \
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)).apply(kaiming_init_weights), \
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)).apply(kaiming_init_weights)
        else:
            return  \
                feature_out,\
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
                nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\

    # 修改get_temporal_embed函数
    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        with record_function("get_temporal_embed"):
            with record_function("interpolate"):
                emb_resized = F.interpolate(self.weight[None,None,...], 
                                        size=(current_num_embeddings, self.temporal_embedding_dim), 
                                        mode='bilinear', align_corners=True)
            
            N, _ = t.shape
            t = t[0,0]

            with record_function("grid_sample"):
                fdim = self.temporal_embedding_dim
                grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
                grid = (grid - 0.5) * 2
                emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
                
            with record_function("repeat"):
                emb = emb.repeat(1,1,N,1).squeeze()

            return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time_hex(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        '''
        融合空间和时间特征，输出隐藏特征， 若 no_grid=True,则拼接空间和时间， 否则用HexPlaneField插值空间-时间特征
        '''
        # 确保输入是 Float 类型（32位）
        rays_pts_emb = rays_pts_emb.float()
        time_emb = time_emb.float() if time_emb is not None else None
        with record_function("grid_query"): 
            # 只使用位置的前三个维度，并确保时间戳是正确的形状
            if time_emb is not None:
                grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            else:
                # 如果没有时间戳，创建一个全零时间戳
                zero_time = torch.zeros((rays_pts_emb.shape[0], 1), device=rays_pts_emb.device, dtype=torch.float32)
                grid_feature = self.grid(rays_pts_emb[:,:3], zero_time)
            
        # 确保特征是 Float 类型
        grid_feature = grid_feature.float()
        
        with record_function("network_forward"):  # 细分网络前向传播
            hidden = torch.cat([grid_feature], -1)
            hidden = self.feature_out_c(hidden)
        
        return hidden
    
    
    def query_time(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        # 添加对None的检查
        if time_emb is None:
            # 如果time_emb为None，创建一个全零时间戳
            time_emb = torch.zeros((pts.shape[0], 1), device=pts.device, dtype=torch.float32)
    
        # 第1步：提取时间信息
        t = time_emb[:,:1] # 获取所有批次的第一个时间戳，假设时间嵌入是一个形状为[N, 1]的张量， 所以t是一个形状为[N, 1]的张量

        # 第2步：根据不同策略获取时间嵌入
        if use_coarse_temporal_embedding:
            # 粗粒度：使用固定的较少嵌入数量
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                # 不使用渐进式：直接使用最大嵌入数量
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                # 渐进式：根据当前迭代次数动态调整嵌入数量
                # 从num_down_emb逐渐增加到max_embeddings
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
        
        # 第3步：拼接时间嵌入和高斯嵌入
        if type(pc) == type(None):
            h = torch.cat([h, embeddings], dim=-1) # 直接使用传入的embeddings
        else:
            h = torch.cat([h, pc.get_embedding], dim=-1) # 从点云对象获取embedding

        # 第4步：通过特征网络处理
        h = feature_out(h)
        return h

    def deform(self, hidden, pts, scales, rotations, opacity, sh_coefs, pos_deform, scales_deform, rotations_deform, opacity_deform, rgb_deform, scale=1., scale_c=1., scale_o=1., coef_s=1.):
        dx, ds, dr, do = pos_deform(hidden), None, None, None
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = scales_deform(hidden)
            scales = scales + ds * scale * coef_s  # scale 是主退火系数
        if not self.args.no_dr:
            dr = rotations_deform(hidden)
            rotations = rotations + dr * scale
        if not self.args.no_do:
            do = opacity_deform(hidden) 
            opacity = opacity + do * scale * scale_o
        if not self.args.no_dc:
            dc = rgb_deform(hidden) 
            sh_coefs = sh_coefs + dc.view(-1,16,3) * scale_c
        return pts, scales, rotations, opacity, sh_coefs
    
    def forward(self, point, scales=None, rotations=None, opacity=None, time_emb=None, cam_no=None, pc=None, embeddings=None, sh_coefs=None, iter=None, num_down_emb_c=30, num_down_emb_f=30):
        '''
        用空间+时间特征，预测动态点云的所有属性， 先用query_time得到隐藏特征，根据mask控制哪些点参与形变，分别通过各属性头预测位置，尺度，旋转，透明度和SH系数
        '''
        # 确保所有输入都是 Float 类型
        point = point.float()
        scales = scales.float() if scales is not None else None
        rotations = rotations.float() if rotations is not None else None
        opacity = opacity.float() if opacity is not None else None
        time_emb = time_emb.float() if time_emb is not None else None

        # 退火系数计算
        use_anneal = self.args.use_anneal
        coef = 1 if not use_anneal else np.clip(iter/1000,0,1)  # 主变形系数
        coef_c = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_o = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_s = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        
        point, scales, rotations, opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig = point, scales, rotations, opacity, sh_coefs

        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)

        hidden = self.query_time_hex(point_emb, scales_emb, rotations_emb, None, time_emb)
        dx = self.pos_deform_c(hidden)  # 位置变形
        
        # 修复：不应该将位移添加到傅里叶特征上，而应该添加到原始点上
        pts_sub = point + dx  # 修正：使用原始点坐标而不是point_emb
        
        ds = self.scales_deform_c(hidden) if not self.args.no_ds else None  # 尺度变形
        scales_sub = scales + ds if ds is not None else scales  # 尺度变形后的尺度
        dr = self.rotations_deform_c(hidden) if not self.args.no_dr else None  # 旋转变形
        rotations_sub = rotations + dr if dr is not None else rotations  # 旋转变形后的旋转
        do = self.opacity_deform_c(hidden) if not self.args.no_do else None  # 透明度变形
        opacity_sub = opacity + do if do is not None else opacity  # 透明度变形后的透明度
        dc = self.rgb_deform_c(hidden) if not self.args.no_dc else None  # RGB变形
        sh_coefs_sub = sh_coefs + dc.view(-1,16,3) if dc is not None else sh_coefs  # RGB变形后的SH系数

        # 修复：这里传递None作为time_emb是不正确的，应该传递原始time_emb
        hidden_f = self.query_time(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings, sh_coefs_sub, iter, self.feature_out_f, use_coarse_temporal_embedding=self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_f).float()
        pts, scales, rotations, opacity, sh_coefs = self.deform(hidden_f, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub,\
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f, coef, coef_c, coef_o, coef_s)

        return pts, scales, rotations, opacity, sh_coefs, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig))

    # def forward(self, point, scales=None, rotations=None, opacity=None, time_emb=None, cam_no=None, pc=None, embeddings=None, sh_coefs=None, iter=None, num_down_emb_c=30, num_down_emb_f=30):
    #     pts, scales, rotations, opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1]
    #     pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig = pts, scales, rotations, opacity, sh_coefs
        
    #     # 如果不存在cam_no, 则说明只有一个相机，不需要考虑不同相机间同步存在的时间偏移误差
    #     if type(cam_no) == type(None):
    #         offset = torch.masked_select(self.offsets, self.offsets.ne(0)).mean() # 计算非零偏移的平均值， 如果没有非零偏移，则返回0
    #         offset[torch.isnan(offset)] = 0  # 避免nan值，替换为0
    #     else:
    #         offset = self.offsets[cam_no]
    #     time_emb += offset

    #     # 退火系数计算
    #     use_anneal = self.args.use_anneal
    #     coef = 1 if not use_anneal else np.clip(iter/1000,0,1)  # 主变形系数
    #     coef_c = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
    #     coef_o = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
    #     coef_s = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)

    #     if self.args.no_coarse_deform:
    #         # 如果不使用粗粒度变形，则直接使用原始点云数据
    #         pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig
    #     else:
    #         hidden = self.query_time(pts, scales, rotations, time_emb, pc, embeddings, sh_coefs, iter, self.feature_out_c, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()        
    #         pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = self.deform(hidden, pts, scales, rotations, opacity, sh_coefs,\
    #             self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c, coef, coef_c, coef_o, coef_s)

    #     if self.args.no_fine_deform:
    #         # 如果不使用细粒度变形，则直接使用粗粒度变形后的点云数据
    #         pts, scales, rotations, opacity, sh_coefs = pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub
    #     else:
    #         hidden = self.query_time(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings, sh_coefs_sub, iter, self.feature_out_f, num_down_emb=num_down_emb_f).float()
    #         pts, scales, rotations, opacity, sh_coefs = self.deform(hidden, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub,\
    #             self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f, coef, coef_c, coef_o, coef_s)
    #     # 返回三种变形后的点云数据：原始的点云数据和经过粗粒度、细粒度变形后的点云数据
    #     return pts, scales, rotations, opacity, sh_coefs, \
    #         ((pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub), \
    #         (pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig))
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
