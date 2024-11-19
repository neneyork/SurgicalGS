import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm
from scene.cameras import Camera
from typing import NamedTuple
from utils.graphics_utils import focal2fov, fov2focal
import glob
from torchvision import transforms as T
import open3d as o3d
from tqdm import trange
import imageio.v2 as iio
import cv2
import copy
import torch
import torch.nn.functional as F
from utils.general_utils import inpaint_depth, inpaint_rgb

def generate_se3_matrix(translation, rotation_rad):


    # Create rotation matrices around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_rad[0]), -np.sin(rotation_rad[0])],
                   [0, np.sin(rotation_rad[0]), np.cos(rotation_rad[0])]])

    Ry = np.array([[np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]])

    Rz = np.array([[np.cos(rotation_rad[2]), -np.sin(rotation_rad[2]), 0],
                   [np.sin(rotation_rad[2]), np.cos(rotation_rad[2]), 0],
                   [0, 0, 1]])

    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create S(3) matrix
    se3_matrix = np.eye(4)


    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = translation

    return se3_matrix

def update_extr(c2w, rotation_deg, radii_mm):
        rotation_rad = np.radians(rotation_deg)
        translation = np.array([-radii_mm * np.sin(rotation_rad) , 0, radii_mm * (1 - np.cos(rotation_rad))])
        # translation = np.array([0, 0, 10])
        se3_matrix = generate_se3_matrix(translation, (0,rotation_rad,0)) # transform_C_C'
        extr = np.linalg.inv(se3_matrix) @ np.linalg.inv(c2w) # transform_C'_W = transform_C'_C @ (transform_W_C)^-1
        
        return np.linalg.inv(extr) # c2w
    
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    Zfar: float
    Znear: float


class EndoNeRF_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        test_every=8
    ):
        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample 
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False

        self.load_meta()
        print(f"meta data loaded, total image:{len(self.image_paths)}")
        
        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every == 0]
        self.video_idxs = [i for i in range(n_frames)]
        
        self.maxtime = 1.0
        
    def load_meta(self):
        """
        Load meta data from the dataset.
        """

        # load poses
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = (focal, focal)
        self.K = np.array([[focal, 0 , W//2],
                            [0, focal, H//2],
                            [0, 0, 1]]).astype(np.float32)
        poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)
    
        # prepare poses
        self.image_poses = []
        self.image_times = []
        for idx in range(poses.shape[0]):
            pose = poses[idx]
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
            
            # # ======================Generate the novel view for infer (StereoMIS)==========================
            # thetas = np.linspace(0, 30, poses.shape[0], endpoint=False)
            # c2w = update_extr(c2w, rotation_deg=thetas[idx], radii_mm=30)
            # # =================================================================================
            
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1] #w2c
            R = np.transpose(R) #c2w
            self.image_poses.append((R, T))
            self.image_times.append(idx / poses.shape[0])
        
        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, filetype, "*.png")))
        self.image_paths = agg_fn("images")
        self.depth_paths = agg_fn("depth")
        # self.masks_paths = agg_fn("mask_combine")
        self.masks_paths = agg_fn("masks")

        # import pdb;pdb.set_trace
        assert len(self.image_paths) == poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.depth_paths) == poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.masks_paths) == poses.shape[0], "the number of masks should equal to the number of poses"
        
    def format_infos(self, split):
        cameras = []
        
        if split == 'train': idxs = self.train_idxs
        elif split == 'test': idxs = self.test_idxs
        else:
            idxs = self.video_idxs
        
        for idx in tqdm(idxs):
            # mask / depth
            mask_path = self.masks_paths[idx]
            mask = Image.open(mask_path)

            mask = 1 - np.array(mask) / 255.0
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path))
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth) 
            depth = torch.from_numpy(depth)
            mask = self.transform(mask).bool()
            # color
            color = np.array(Image.open(self.image_paths[idx]))/255.0
            image = self.transform(color)
            # times           
            time = self.image_times[idx]
            # poses
            R, T = self.image_poses[idx]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])
            cameras.append(Camera(colmap_id=idx, R=R, T=T, FoVx=FovX, FoVy=FovY,image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                          image_name=f"{idx}", uid=idx, data_device=torch.device("cuda"), time=time,
                          Znear=None, Zfar=None, K=self.K, h=self.img_wh[1], w=self.img_wh[0]))
        return cameras

    def get_pts_cam(self, depth, mask, color, disable_mask=False):
        W, H = self.img_wh
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        cx = self.K[0,-1]
        cy = self.K[1,-1]
        X_Z = (i-cx) / self.focal[0]
        Y_Z = (j-cy) / self.focal[1]
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
            color_valid = color[mask, :]
        else:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam
            color_valid = color
            color_valid[mask==0, :] = np.ones(3)
                    
        return pts_valid, color_valid, mask
        
    def get_maxtime(self):
        return self.maxtime

    def get_initialization(self, ):
        pts_total, colors_total = [], []
        motion_masks = []
        
        # frame 0
        color, depth_frame0, mask_frame0 = self.get_color_depth_mask(0)
        motion_masks.append(mask_frame0)
        pts, colors, _ = self.get_pts_cam(depth_frame0, mask_frame0, color, disable_mask=False)
        pts = self.get_pts_wld(pts, self.image_poses[0])
        num_pts = pts.shape[0]
        sel_idxs = np.random.choice(num_pts, 35000, replace=False)
        pts_sel, colors_sel = pts[sel_idxs], colors[sel_idxs]
        pts_total.append(pts_sel)
        colors_total.append(colors_sel)
        # import pdb;pdb.set_trace()
        # rest frame
        if len(self.image_poses) > 150: # in case long sequence
            interval = 2
        else:
            interval = 1
        mask_frame0_not = np.logical_not(mask_frame0)
        for idx in range(1,  len(self.image_poses), interval):
            color, depth, mask = self.get_color_depth_mask(idx)
            motion_mask = self.get_motion_masks(idx, depth_frame0)
            motion_masks.append(motion_mask)
            # import torchvision
            # torchvision.utils.save_image(torch.from_numpy(fuse_mask).float(), os.path.join("/root/autodl-tmp/tmp/4d-gaussian-splatting/output/pushing/pushing_init", '{0:05d}'.format(idx) + ".png"))
            
            fuse_mask_tmp = np.logical_or(motion_mask,mask_frame0_not)
            fuse_mask = np.logical_and(mask, fuse_mask_tmp)
            pts, colors, _ = self.get_pts_cam(depth, fuse_mask, color, disable_mask=False)
            pts = self.get_pts_wld(pts, self.image_poses[idx])
            
            num_pts = pts.shape[0]
            num_sel = int(5000 / len(self.image_poses))
            sel_idxs = np.random.choice(num_pts, num_sel, replace=False)
            pts_sel, colors_sel = pts[sel_idxs], colors[sel_idxs]
            pts_total.append(pts_sel)
            colors_total.append(colors_sel)
        # import pdb;pdb.set_trace()
        pts_total = np.concatenate(pts_total)
        colors_total = np.concatenate(colors_total)
        # sel_idxs = np.random.choice(pts_total.shape[0], 30_000, replace=True)
        # pts, colors = pts_total[sel_idxs], colors_total[sel_idxs]
        normals = np.zeros((pts_total.shape[0], 3))
        
        return pts_total, colors_total, normals, motion_masks

    def get_motion_masks(self, idx, depth_frame0):
        color, depth, mask = self.get_color_depth_mask(idx) 
        diff_map = np.abs(depth - depth_frame0)
        diff_thrshold = 40
        
        motion_mask = diff_map > diff_thrshold
        
        return motion_mask

    def get_pts_wld(self, pts, pose):
        R, T = pose
        R = np.transpose(R)
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        pts_cam_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
        pts_wld = pts_wld[:, :3]
        return pts_wld
    
    def get_color_depth_mask(self, idx):
        depth = np.array(Image.open(self.depth_paths[idx]))
        close_depth = np.percentile(depth[depth!=0], 3.0)
        inf_depth = np.percentile(depth[depth!=0], 99.8)
        depth = np.clip(depth, close_depth, inf_depth)
        mask = Image.open(self.masks_paths[idx])
        mask = 1 - np.array(mask) / 255.0
        color = np.array(Image.open(self.image_paths[idx]))/255.0
        return color, depth, mask
