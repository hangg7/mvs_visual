#!/usr/bin/env python3
#
# File   : example.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 04/15/2021
#
# Distributed under terms of the MIT license.

import cv2
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as sR

from mvs_visual import PerspectiveCamera, plot_scene


def circular_c2ws_around_y(c2w, num_poses=30):
    """
    Create circular c2ws around y axis from existed c2w.
    """

    # absolute camera up axis in world space.
    rotvecs = (
        c2w[None, :3, 1]
        * np.linspace(0, 2 * np.pi, num=num_poses + 1)[:num_poses, None]
    )
    rotmats = sR.from_rotvec(rotvecs).as_matrix()
    transforms = np.eye(4)[None].repeat(num_poses, axis=0)
    transforms[:, :3, :3] = rotmats
    circular_c2ws = transforms @ c2w
    return circular_c2ws


def focal_from_fov(fov, s):
    return (s / 2) / np.tan(fov / 2)


# numer of views for visualziation.
N = 3
img_wh = (400, 400)
assert img_wh[0] == img_wh[1], 'Only isotropic imaging is supported.'
yfov = np.pi / 3
focal_length = (
    focal_from_fov(yfov, img_wh[0]),
    focal_from_fov(yfov, img_wh[1]),
)
principal_point = (200, 200)

# bunny mesh has following orientation for +x/y/z axes: left/up/forward.
bunny_tmesh = trimesh.load('assets/bunny.ply')
# recenter and rescale mesh such that the longest bound has length 1.
bunny_tmesh.vertices -= bunny_tmesh.vertices.mean(axis=0, keepdims=True)
bounds = bunny_tmesh.bounds
bunny_tmesh.vertices /= (bounds[1] - bounds[0]).max()

# assume an opengl camera model: +x/y/z -> right/up/back.
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3, aspectRatio=1)

# put the camera at (0, 0, 1.5) in world frame with up +y facing -z.
start_c2w = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.5], [0, 0, 0, 1]])
# and rotate it around +y.
c2ws = circular_c2ws_around_y(start_c2w, num_poses=N)

# put the light at (0, 1.5, 0) in world frame with up +z facing -y.
light_c2w = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 1.5], [0, -1, 0, 0], [0, 0, 0, 1]]
)
light = pyrender.PointLight(color=np.ones(3), intensity=5.0)

r = pyrender.OffscreenRenderer(*img_wh)
scene = pyrender.Scene()
mesh_node = pyrender.Node(
    mesh=pyrender.Mesh.from_trimesh(bunny_tmesh), matrix=np.eye(4)
)
camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
light_node = pyrender.Node(light=light, matrix=light_c2w)
scene.add_node(mesh_node)
scene.add_node(camera_node)
scene.add_node(light_node)

plot_dict = dict(bunny=dict(mesh=bunny_tmesh))

for i, c2w in enumerate(c2ws):
    scene.set_pose(camera_node, pose=c2w)
    img = r.render(scene)[0]
    cv2.imwrite(f'view{i}.png', img[..., ::-1])
    plot_dict['bunny'][f'view{i}'] = (
        PerspectiveCamera(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=img_wh,
            c2w=c2w,
        ),
        img,
    )
fig = plot_scene(plot_dict, camera_scale=0.3)
fig.write_html('example.html')
