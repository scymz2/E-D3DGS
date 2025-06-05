import warnings
warnings.filterwarnings("ignore")

import json
import os
import random
import numpy as np
import torch
from PIL import Image
import math
from tqdm import tqdm
from scene.utils import Camera
from typing import NamedTuple
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
import torch.nn.functional as F
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.pose_utils import smooth_camera_poses

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float
    mask: np.array

class Load_hyper_data(Dataset):
    def __init__(self, 
                 datadir, 
                 ratio=1.0,
                 use_bg_points=False,
                 split="train",
                 startime=0,
                 duration=None):
        
        datadir = os.path.expanduser(datadir)
        
        # 加载JSON配置文件
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        # 场景参数
        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        # 图像ID和验证ID
        self.all_img = dataset_json['ids']
        self.val_id = dataset_json.get('val_ids', [])
        
        # 时间范围设置
        self.startime = startime
        self.duration = len(self.all_img) // 2 if duration is None else duration
        
        # 根据时间范围裁剪数据
        end_time = self.startime + self.duration
        self.all_img = self.all_img[self.startime * 2 : end_time * 2]
        if self.val_id:
            self.val_id = self.val_id[self.startime:end_time]

        self.split = split
        self.ratio = ratio
        
        # 初始化索引
        self._init_indices(dataset_json)
        
        # 加载相机参数和时间信息
        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        
        self.max_time = max(self.all_time)
        self.min_time = min(self.all_time)
        
        # 加载相机参数
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            self.all_cam_params.append(camera)
        
        # 设置图像路径
        scale_factor = int(1/ratio)
        self.all_img_paths = [f'{datadir}/rgb/{scale_factor}x/{i}.png' for i in self.all_img]
        self.all_depth_paths = [f'{datadir}/depth/{scale_factor}x/{i}.npy' for i in self.all_img]
        
        # 获取图像尺寸
        self.h, self.w = self.all_cam_params[0].image_shape
        
        # 加载掩码路径（如果存在）
        covisible_dir = os.path.join(datadir, "covisible")
        if os.path.exists(covisible_dir):
            self.image_mask = [f'{covisible_dir}/2x/val/{i}.png' for i in self.all_img]
        else:
            self.image_mask = None
        
        # 加载一个示例图像用于视频生成
        if self.all_img_paths:
            self.sample_image = Image.open(self.all_img_paths[0])
        else:
            self.sample_image = None
        
        # 为视频模式生成路径
        if split == "video":
            self._generate_video_path()

    def _init_indices(self, dataset_json):
        """初始化训练和测试索引"""
        if len(self.val_id) == 0:
            # 如果没有验证ID，使用默认分割策略
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if i % 4 == 0])
            self.i_test = self.i_train + 2
            self.i_test = self.i_test[self.i_test < len(self.all_img)]  # 确保不超出范围
        else:
            # 使用提供的训练和验证ID
            self.train_id = dataset_json.get('train_ids', [])
            self.i_test = []
            self.i_train = []
            
            for i, img_id in enumerate(self.all_img):
                if img_id in self.val_id:
                    self.i_test.append(i)
                if img_id in self.train_id:
                    self.i_train.append(i)

    def _generate_video_path(self):
        """生成视频路径"""
        # 选择部分相机用于视频生成
        self.select_video_cams = [self.all_cam_params[i] for i in range(0, len(self.all_cam_params), max(1, len(self.all_cam_params) // 20))]
        
        if len(self.select_video_cams) > 1:
            self.video_cameras, self.video_times = smooth_camera_poses(self.select_video_cams, 10)
            # 限制视频长度
            max_video_length = 500
            self.video_cameras = self.video_cameras[:max_video_length]
            self.video_times = self.video_times[:max_video_length]

            # 确保视频时间戳在[0,1]范围内
            if len(self.video_times) > 0:
                min_time = min(self.video_times)
                max_time = max(self.video_times)
                if max_time > min_time:
                    self.video_times = [(t - min_time) / (max_time - min_time) for t in self.video_times]
                else:
                    self.video_times = [0.0] * len(self.video_times)
        else:
            self.video_cameras = self.select_video_cams
            self.video_times = [0.0]

    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            return len(getattr(self, 'video_cameras', []))
        return 0

    def __getitem__(self, index):
        if self.split == "train":
            return self._load_camera_info(self.i_train[index], index)
        elif self.split == "test":
            return self._load_camera_info(self.i_test[index], index)
        elif self.split == "video":
            return self._load_video_camera_info(index)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _load_camera_info(self, data_idx, uid):
        """加载相机信息"""
        camera = self.all_cam_params[data_idx]
        
        # 加载图像
        image = Image.open(self.all_img_paths[data_idx])
        w, h = image.size
        
        # 计算时间戳（归一化到[0,1]）
        time = self.all_time[data_idx]
        normalized_timestamp = (time - self.startime) / max(1, self.duration)
        normalized_timestamp = np.clip(normalized_timestamp, 0.0, 1.0)
        
        # 相机参数
        R = camera.orientation.T
        T = -camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        
        # 主点偏移
        cxr = (camera.principal_point[0] / self.w - 0.5)
        cyr = (camera.principal_point[1] / self.h - 0.5)
        
        # 路径信息
        image_path = "/".join(self.all_img_paths[data_idx].split("/")[:-1])
        image_name = self.all_img_paths[data_idx].split("/")[-1]
        
        # 加载掩码（仅测试时）
        mask = None
        if self.image_mask is not None and self.split == "test":
            try:
                mask_img = Image.open(self.image_mask[data_idx])
                mask = PILtoTorch(mask_img, None).to(torch.float32)[0:1, :, :]
                mask = F.interpolate(mask.unsqueeze(0), size=[self.h, self.w], 
                                   mode='bilinear', align_corners=False).squeeze(0)
            except Exception as e:
                print(f"Warning: Could not load mask for {self.image_mask[data_idx]}: {e}")
                mask = None
        
        return CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_name, width=w, height=h,
            near=self.near, far=self.far, timestamp=normalized_timestamp,
            pose=np.eye(4), hpdirecitons=np.array([0, 0, 1]), cxr=cxr, cyr=cyr,
            mask=mask
        )

    def _load_video_camera_info(self, index):
        """加载视频相机信息"""
        if not hasattr(self, 'video_cameras') or index >= len(self.video_cameras):
            raise IndexError(f"Video camera index {index} out of range")
            
        camera = self.video_cameras[index]
        time = self.video_times[index] if index < len(self.video_times) else 0.0
        normalized_timestamp = np.clip(time, 0.0, 1.0)

        # 使用示例图像的尺寸
        if self.sample_image:
            w, h = self.sample_image.size
            image = self.sample_image
        else:
            w, h = self.w, self.h
            image = None
        
        # 相机参数
        R = camera.orientation.T
        T = -camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        
        # 主点偏移
        cxr = (camera.principal_point[0] / self.w - 0.5)
        cyr = (camera.principal_point[1] / self.h - 0.5)
        
        return CameraInfo(
            uid=index, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path="", image_name=f"video_{index:04d}.png", width=w, height=h,
            near=self.near, far=self.far, timestamp=normalized_timestamp,
            pose=np.eye(4), hpdirecitons=np.array([0, 0, 1]), cxr=cxr, cyr=cyr,
            mask=None
        )

# 移除format_hyper_data函数，因为已经不需要了