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


class deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(deform_network, self).__init__()
        self.D = D
        self.W = W

        self.args = args
        self.min_embeddings = min_embeddings # 最小的时间嵌入数量
        self.max_embeddings = max_embeddings # 最大的时间嵌入数量
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        self.c2f_temporal_iter = args.c2f_temporal_iter # 渐进式训练参数


        # 以下网络包含位置变形（三维）， 尺度变形（三维）， 旋转变形（四维四元数）， 不透明度变形（一维）， RGB变形（四十八维， 3*16球谐函数）
        # 粗粒度网络（coarse）
        self.feature_out_c, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c = self.create_net()
        # 细粒度网络（fine）
        self.feature_out_f, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f = self.create_net()

        if args.zero_temporal: # 时间嵌入
            # 零初始化的时间嵌入
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            # 正态分布初始化的时间嵌入，避免梯度爆炸或者消失
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))  # hard coded the upper limit of the num cameras (adjust as necessary)

    def create_net(self):
        # 初始化一个包含单个线性层的列表，这个线性层的作用是将时间嵌入和高斯嵌入的维度组合并映射到W维度
        self.feature_out = [nn.Linear(self.temporal_embedding_dim + self.gaussian_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        feature_out = nn.Sequential(*self.feature_out) # 列表转为顺序容器, Sequential可以将多个层组合成一个层，层中自动执行forward
        return  \
            feature_out,\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        # 动态调整时间嵌入表大小
        # self.weight shape: [max_embeddings, temporal_embedding_dim]
        # 通过双线性插值将其调整为 [current_num_embeddings, temporal_embedding_dim]
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        
        # 第2步：获取批次大小和时间戳
        N, _ = t.shape
        t = t[0,0]

        # 第三步：构建采样网格
        fdim = self.temporal_embedding_dim
        # 创建一个2D网格用于grid_sample，双线性归一化采样
        # x坐标：特征维度的归一化坐标 [0, 1]
        # y坐标：时间坐标，所有位置都是相同的时间t
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        # 第4步：将坐标从[0,1]转换到[-1,1]（grid_sample要求）
        grid = (grid - 0.5) * 2

        # 第5步：使用grid_sample进行双线性采样
        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        # 第6步：扩展到批次维度
        emb = emb.repeat(1,1,N,1).squeeze()

        return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
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
        pts, scales, rotations, opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig = pts, scales, rotations, opacity, sh_coefs
        
        # 如果不存在cam_no, 则说明只有一个相机，不需要考虑不同相机间同步存在的时间偏移误差
        if type(cam_no) == type(None):
            offset = torch.masked_select(self.offsets, self.offsets.ne(0)).mean() # 计算非零偏移的平均值， 如果没有非零偏移，则返回0
            offset[torch.isnan(offset)] = 0  # 避免nan值，替换为0
        else:
            offset = self.offsets[cam_no]
        time_emb += offset

        # 退火系数计算
        use_anneal = self.args.use_anneal
        coef = 1 if not use_anneal else np.clip(iter/1000,0,1)  # 主变形系数
        coef_c = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_o = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_s = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)

        if self.args.no_coarse_deform:
            # 如果不使用粗粒度变形，则直接使用原始点云数据
            pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig
        else:
            hidden = self.query_time(pts, scales, rotations, time_emb, pc, embeddings, sh_coefs, iter, self.feature_out_c, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()        
            pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = self.deform(hidden, pts, scales, rotations, opacity, sh_coefs,\
                self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c, coef, coef_c, coef_o, coef_s)

        if self.args.no_fine_deform:
            # 如果不使用细粒度变形，则直接使用粗粒度变形后的点云数据
            pts, scales, rotations, opacity, sh_coefs = pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub
        else:
            hidden = self.query_time(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings, sh_coefs_sub, iter, self.feature_out_f, num_down_emb=num_down_emb_f).float()
            pts, scales, rotations, opacity, sh_coefs = self.deform(hidden, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub,\
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f, coef, coef_c, coef_o, coef_s)
        # 返回三种变形后的点云数据：原始的点云数据和经过粗粒度、细粒度变形后的点云数据
        return pts, scales, rotations, opacity, sh_coefs, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig))
    
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
