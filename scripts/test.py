import sys
sys.path.append('.')

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from lietorch import SE3
import raft3d.projective_ops as pops
from data_readers import frame_utils
from utils import show_image, normalize_image


DEPTH_SCALE = 1

def prepare_images_and_depths(image1, image2, depth1, depth2):
    """ padding, normalization, and scaling """

    image1 = F.pad(image1, [0,0,0,4], mode='replicate')
    image2 = F.pad(image2, [0,0,0,4], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,0,0,4], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,0,0,4], mode='replicate')[:,0]

    depth1 = (DEPTH_SCALE * depth1).float()
    depth2 = (DEPTH_SCALE * depth2).float()
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    return image1, image2, depth1, depth2


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()


@torch.no_grad()
def demo(args):
    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D
    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model), strict=False)

    model.eval()
    model.cuda()

    fx, fy, cx, cy, b = (417.903625, 417.903625, 157.208288, 143.735811, -5.196155)
    limg1 = "/home/avinash/Desktop/datasets/endo/depth/rectified15/cropped/image01/0000000010.jpg"
    limg2 = "/home/avinash/Desktop/datasets/endo/depth/rectified15/cropped/image01/0000000015.jpg"
    ldisp1 = "/home/avinash/Desktop/datasets/endo/depth/rectified15/cropped/depth01/0000000010.npy"
    ldisp2 = "/home/avinash/Desktop/datasets/endo/depth/rectified15/cropped/depth01/0000000015.npy"
    img1 = cv2.imread(limg1)
    img2 = cv2.imread(limg2)
    disp1 = frame_utils.read_gen(ldisp1)
    disp2 = frame_utils.read_gen(ldisp2)

    print(disp1)

    depth1 = torch.from_numpy(fx*b/ disp1).float().cuda().unsqueeze(0)
    depth2 = torch.from_numpy(fx*b/ disp2).float().cuda().unsqueeze(0)
    image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0)
    image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0)
    intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0)

    image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)
    Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
    
    # compute 2d and 3d from from SE3 field (Ts)
    flow2d, flow3d, _ , X0, X1 = pops.induced_flow(Ts, depth1, intrinsics)

    import open3d as o3d

    # Assuming X0 and X1 are numpy arrays representing the 3D coordinates
    # X0.shape = (N, 3), X1.shape = (N, 3), where N is the number of points

    # Create Open3D point cloud objects
    pcd0 = o3d.geometry.PointCloud()

    pcd1 = o3d.geometry.PointCloud()

    p1 = X0.cpu().detach().numpy()
    p2 = X1.cpu().detach().numpy()

    X0 = p1[0].reshape(-1,3)
    X1 = p2[0].reshape(-1,3)

    print(X0.shape)

    # Set the point cloud data
    pcd0.points = o3d.utility.Vector3dVector(X0)
    pcd1.points = o3d.utility.Vector3dVector(X1)

    # Visualize the point clouds
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd0)
    vis.add_geometry(pcd1)
    vis.run()
    vis.destroy_wind()

    # extract rotational and translational components of Ts
    # tau, phi = Ts.log().split([3,3], dim=-1)
    # tau = tau[0].cpu().numpy()
    # phi = phi[0].cpu().numpy()

    # # undo depth scaling
    # flow3d = flow3d / DEPTH_SCALE

    # display(img1, tau, phi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft3d.pth', help='checkpoint to restore')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    args = parser.parse_args()

    demo(args)

    


