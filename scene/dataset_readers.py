#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
import torchvision.transforms as transforms
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm

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
    time : float
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}



def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w

    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        fovx = template_json["camera_angle_x"]
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time))
    return cam_infos

def readEndoNeRFInfo(datadir):
    # load camera infos
    from scene.endo_loader import EndoNeRF_Dataset
    endo_dataset = EndoNeRF_Dataset(
        datadir=datadir,
        downsample=1.0,
    )
    train_cam_infos = endo_dataset.format_infos(split="train")
    test_cam_infos = endo_dataset.format_infos(split="test")
    video_cam_infos = endo_dataset.format_infos(split="video")
    
    # get normalizations
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # initialize sparse point clouds
    ply_path = os.path.join(datadir, "points3d.ply")
    xyz, rgb, normals, _ = endo_dataset.get_initialization()
    
    normals = np.random.random((xyz.shape[0], 3))
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)
    storePly(ply_path, xyz,rgb*255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    # get the maximum time
    maxtime = endo_dataset.get_maxtime()
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=maxtime)

    return scene_info
    


sceneLoadTypeCallbacks = {
    "endonerf": readEndoNeRFInfo,
}
