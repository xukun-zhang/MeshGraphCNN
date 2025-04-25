import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import kornia
import random
import torchvision.transforms

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import (
        Rotate, Translate, so3_log_map, so3_exp_map, Transform3d,
        euler_angles_to_matrix,
        matrix_to_rotation_6d,
        rotation_6d_to_matrix
)

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, SoftPhongShader, PointLights, TexturesVertex,
)
import utils


class Model(nn.Module):
    def __init__(self, 
            meshes, 
            renderer, 
            dilate_labels=True,
            initial_obj_pos=(0,0,0), 
            initial_obj_rot=torch.eye(3),
            device="cuda",
        ):
        super().__init__()

        self.meshes = meshes
        self.renderer = renderer
        self.dilate_labels = dilate_labels
        self.device = device
        self.initial_obj_pos = initial_obj_pos
        self.initial_obj_rot = initial_obj_rot

        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        textures_gray = torch.ones_like(verts).unsqueeze(0).to(device) * 0.5
        self.mesh_gray = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(textures_gray))
        
        # 设置顶点颜色为白色
        textures_white = torch.ones_like(verts).unsqueeze(0).to(device)  # 所有顶点颜色为 [1.0, 1.0, 1.0]

        # 创建带有白色纹理的网格对象
        self.mesh_white = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(textures_white))

        self.textures_label = meshes.textures.clone()

        # continuous 6D representation of rotation for easier learning 
        self.initial_obj_rot = self.initial_obj_rot.unsqueeze(0)
        self.R = matrix_to_rotation_6d(self.initial_obj_rot).detach().to(self.device)
        self.T = torch.tensor(initial_obj_pos).unsqueeze(0).detach().to(self.device)  
        self.R.requires_grad = True
        self.T.requires_grad = True

        # upper edge detection kernel: [in_channel, out_channel, kernel_height, kernel_width]
        # kernel = [-1, 0, 1].T
        self.silhouette_kernel = torch.zeros((3,3,3,1)).to(self.device)
        self.silhouette_kernel[1, 1, 0, 0] = -1
        self.silhouette_kernel[1, 1, 2, 0] = 1
        

    def sihlouette_detection(self, image):
        # detect sihlouette using the rendered grey mesh
        tmp = image[...,:3].permute(0,3,1,2)
        tmp = torch.nn.functional.pad(tmp,
                pad=(0,0,1,1),     # pad height, but not width, by 1
                mode="replicate",
        )
        edge = torch.nn.functional.conv2d(tmp, self.silhouette_kernel, padding=0)
        # set pixels below threshold to black
        edge[edge < 0.1] = 0

        # dilate edge
        if self.dilate_labels:
            edge = kornia.morphology.dilation(
                edge,
                kernel = torch.ones((11,11)).to(self.device)
            )
        edge = edge.permute(0,2,3,1)
        # Scale the result, because the grayscale image may not have been 0 - 1 but 0 - 0.5
        # (for example), and we want the green to reach from 0 to 1:
        if edge.max() > 0:
            edge = edge/edge.max()

        alpha_channel = torch.ones_like(edge[...,0])
        image_silhouette = torch.cat((edge, alpha_channel.unsqueeze(3)), dim=3)
        return image_silhouette


    def initial_alignment(self, image_ref):
        # Attempt to align the images by their ligaments (i.e., the blue regions) and the ridge (i.e., the red regions)
        with torch.no_grad():
            print("初始位置时label- image_ref.shape:", image_ref.shape)
            # Render image from the current position/direction
            rendered_lbl, rendered_img, rendered_edge, R_matrix, T_matrix = self.forward()
            transform = torchvision.transforms.ToPILImage()
            img_cpu = transform(rendered_lbl.squeeze().permute(2,0,1))
            img_cpu.save("blue_search.png")
            

            # Get the pixels that are blue (ligament)
            blue_pixels = rendered_lbl[:,:,:,2] > 0.5
            blue_pixel_inds = blue_pixels.nonzero()
            if blue_pixel_inds.shape[0] < 1150:     # 如果出现的韧带点太少，说明位置还是不对的；
                print("No blue pixels found in rendered image - skip alignment!")
                return False

            # Get the pixels that are red (ridge)
            red_pixels = rendered_lbl[:,:,:,0] > 0.5 
            red_pixels_ref = image_ref[:,:,:,0] > 0.5

            red_combine = rendered_lbl[:,:,:,0] + image_ref[:,:,:,0]
            red_combine_number = red_combine>1
            red_number = red_combine_number.sum()
            print("red_pixels.sum():", red_pixels.sum(), red_pixels_ref.sum(), red_number, rendered_lbl[:,:,:,0].max(), image_ref[:,:,:,0].max(), red_combine.max())
            
            red_pixel_inds = red_pixels.nonzero()

            if red_pixel_inds.shape[0] < 1500 or red_number<50:     # 如果出现的脊的点太少，说明位置也是不对的；
                print("No red pixels found in rendered image - skip alignment!")
                return False
            # Calculate bounding box center for blue (ligament) and red (ridge) regions in the rendered image
            min_x_blue = blue_pixel_inds[:,2].min()
            max_x_blue = blue_pixel_inds[:,2].max()
            min_y_blue = blue_pixel_inds[:,1].min()
            max_y_blue = blue_pixel_inds[:,1].max()
            center_blue = ((max_x_blue+min_x_blue)*0.5, (max_y_blue+min_y_blue)*0.5)

            min_x_red = red_pixel_inds[:,2].min()
            max_x_red = red_pixel_inds[:,2].max()
            min_y_red = red_pixel_inds[:,1].min()
            max_y_red = red_pixel_inds[:,1].max()
            center_red = ((max_x_red+min_x_red)*0.5, (max_y_red+min_y_red)*0.5)

            # Get the blue (ligament) region in the reference image
            blue_pixels_ref = image_ref[:,:,:,2] > 0.25
            blue_pixel_inds_ref = blue_pixels_ref.nonzero()
            if blue_pixel_inds_ref.shape[0] == 0:
                print("No blue pixels found in reference image - skip alignment!")
                return False

            # Get the red (ridge) region in the reference image
            red_pixels_ref = image_ref[:,:,:,0] > 0.25
            red_pixel_inds_ref = red_pixels_ref.nonzero()
            if red_pixel_inds_ref.shape[0] == 0:
                print("No red pixels found in reference image - skip alignment!")
                return False

            # Calculate bounding box center for blue (ligament) and red (ridge) regions in the reference image
            min_x_blue_ref = blue_pixel_inds_ref[:,2].min()
            max_x_blue_ref = blue_pixel_inds_ref[:,2].max()
            min_y_blue_ref = blue_pixel_inds_ref[:,1].min()
            max_y_blue_ref = blue_pixel_inds_ref[:,1].max()
            center_blue_ref = ((max_x_blue_ref+min_x_blue_ref)*0.5, (max_y_blue_ref+min_y_blue_ref)*0.5)

            min_x_red_ref = red_pixel_inds_ref[:,2].min()
            max_x_red_ref = red_pixel_inds_ref[:,2].max()
            min_y_red_ref = red_pixel_inds_ref[:,1].min()
            max_y_red_ref = red_pixel_inds_ref[:,1].max()
            center_red_ref = ((max_x_red_ref+min_x_red_ref)*0.5, (max_y_red_ref+min_y_red_ref)*0.5)

            # Combine the centers to get a more accurate estimate of the required translation
            combined_center = (center_blue[0] + center_red[0]) * 0.5, (center_blue[1] + center_red[1]) * 0.5
            combined_center_ref = (center_blue_ref[0] + center_red_ref[0]) * 0.5, (center_blue_ref[1] + center_red_ref[1]) * 0.5

            # Estimate depth (this part is rough and assumes depth is constant)
            depth = torch.linalg.vector_norm(self.T)

            # Project the bounding box centers back into 3D space using 'depth'
            xy_depth_lbl = [
                2*(combined_center[0]/rendered_lbl.shape[2]-0.5),
                2*(combined_center[1]/rendered_lbl.shape[1]-0.5),
                depth
            ]
            xy_depth_ref = [
                2*(combined_center_ref[0]/rendered_lbl.shape[2]-0.5),
                2*(combined_center_ref[1]/rendered_lbl.shape[1]-0.5),
                depth
            ]

            xy_depth_lbl = torch.Tensor(xy_depth_lbl).to(self.device).unsqueeze(0)
            xy_depth_ref = torch.Tensor(xy_depth_ref).to(self.device).unsqueeze(0)

            cameras = self.renderer.rasterizer.cameras
            pos3D_lbl = cameras.unproject_points(xy_depth_lbl, world_coordinates=True)
            pos3D_ref = cameras.unproject_points(xy_depth_ref, world_coordinates=True)

            # Calculate the difference in position and apply it as an offset to the camera position
            diff = pos3D_ref - pos3D_lbl
            self.T.data -= diff.squeeze()

            # Re-render the image and save
            rendered_img, rendered_lbl, rendered_edge, R_matrix, T_matrix = self.forward()

            transform = torchvision.transforms.ToPILImage()
            img_cpu = transform(rendered_lbl.squeeze().permute(2,0,1))
            img_cpu.save("blue_search_aligned.png")
            
            transform = torchvision.transforms.ToPILImage()
            img_cpu = transform(rendered_edge.squeeze().permute(2,0,1))
            img_cpu.save("blue_search_aligned-1.png")

            transform = torchvision.transforms.ToPILImage()
            img_cpu = transform(rendered_img.squeeze().permute(2,0,1))
            img_cpu.save("blue_search_aligned_2D.png")

            transform = torchvision.transforms.ToPILImage()
            img_cpu = transform(image_ref.squeeze().permute(2,0,1))
            img_cpu.save("blue_search_reference.png")

            return True


    def forward(self):
        # 计算 3D mesh 的质心
        center = self.meshes.verts_packed().mean(0)  # 质心的坐标
        cx, cy, cz = center
        T = self.T.squeeze(0)  # 变为 [3]

        # Step 1: 将 mesh 平移到质心位置
        transform = Transform3d(device=self.device).translate(-cx, -cy, -cz)
        # Step 2: 应用旋转
        print("self.R:", self.R, T)
        R = rotation_6d_to_matrix(self.R)
        # print("R:", R)
        transform = transform.rotate(R)
        # Step 3: 将 mesh 平移回原位置，并应用平移 T
        transform = transform.translate(cx + T[0], cy + T[1], cz + T[2])
        # Step 4: 将变换应用到 mesh 顶点
        verts = transform.transform_points(self.meshes.verts_packed())
        mesh = Meshes(
            verts=[verts], 
            faces=[self.meshes.faces_packed()],
            textures=self.textures_label,
        )
        # Step 5: 创建灰度 mesh
        mesh_gray = Meshes(
            verts=[verts],
            faces=[self.meshes.faces_packed()],
            textures=self.mesh_gray.textures     # textures=self.mesh_white.textures
        )

        # 创建一个只显示边缘的纹理的新 mesh 
        edges = torch.ones_like(verts) * 255  # 设置颜色为白色
        vertex_colors = TexturesVertex(verts_features=edges.unsqueeze(0))
        # 创建带边缘纹理的mesh
        mesh_edge = Meshes(
            verts=[verts],
            faces=[self.meshes.faces_packed()],
            textures=vertex_colors  # 使用边缘纹理
        )
        # 渲染 mesh
        R_cam = torch.eye(3).to(self.device).unsqueeze(0)
        T_cam = torch.FloatTensor([0, 0, 0]).to(self.device).unsqueeze(0)
        mesh_label_rendered = self.renderer(meshes_world=mesh, R=R_cam, T=T_cam)
        mesh_gray_rendered = self.renderer(meshes_world=mesh_gray, R=R_cam, T=T_cam)
        # 渲染边缘纹理新 mesh 
        mesh_edge_rendered_1 = self.renderer(meshes_world=mesh, R=R_cam, T=T_cam)
        # 设定透明背景
        alpha_channel = torch.zeros_like(mesh_edge_rendered_1[..., 0])  # 初始化为透明
        # 保留三角形边缘，设置alpha通道
        alpha_channel[mesh_edge_rendered_1[..., 0] > 0] = 1  # 边缘设置为不透明
        # 将边缘图像和透明度合并
        edge_image = torch.cat([mesh_edge_rendered_1[..., :3], alpha_channel.unsqueeze(-1)], dim=-1)

        # Optional: 膨胀标签图像
        if self.dilate_labels:
            mesh_label_rendered = utils.dilate_image(mesh_label_rendered)
        # 检测轮廓
        mesh_silhouette = self.sihlouette_detection(mesh_gray_rendered)
        # 合并轮廓到标签图像
        mesh_label_rendered = mesh_label_rendered + mesh_silhouette
        
        # 不合并轮廓到标签图像，仅保留韧带和肝脊 
        # mesh_label_rendered = mesh_label_rendered
        return mesh_label_rendered, mesh_gray_rendered, edge_image, R, T



