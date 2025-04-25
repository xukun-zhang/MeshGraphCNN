import torch
import torchvision.transforms as T
from pytorch3d.structures import Meshes
from PIL import Image
from torch import nn
from torch.nn import Parameter
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, look_at_view_transform, PerspectiveCameras, SoftPhongShader,
    TexturesVertex, PointLights, BlendParams,
)
from pytorch3d.transforms import (
    Rotate, Translate, euler_angles_to_matrix, 
    quaternion_multiply, quaternion_to_matrix, 
    axis_angle_to_quaternion,
    rotation_6d_to_matrix, matrix_to_quaternion, matrix_to_rotation_6d,
)
from pytorch3d.utils import (
    cameras_from_opencv_projection,
)
from lightless_shader import LightlessShader
from model import Model
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
import matplotlib.pyplot as plt
import utils
import optuna
import nibabel as nib
import os
import math
from loss import overlap_loss, combined_loss
import random
import datetime
import torchvision.transforms as T
# 设定设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import torchvision.transforms

def load_data(
    path_mesh = "/home/XXX/P2ILF22_patient7.obj",
    path_image = "/home/XXX/P2ILF22_patient7_14.png",
    path_camera_param = "/home/XXX/calibration.xml",
    scale=1e-3,
    dilate=True,
    image_width=1920
):
    image_scale_factor = image_width / 1920
    print("image_scale_factor:", image_scale_factor)
    # 加载 3D 模型和 2D 图像
    mesh = load_objs_as_meshes([path_mesh], device=device)

    """
    P2ILF-7 Landmarks
    """
    ligament_indices = [709, 756, 783, 710, 815, 801, 855, 894, 895, 962, 912, 936, 1054, 1093, 1009, 935, 1075, 1148, 1128, 1176, 1107, 1146, 1106, 1236, 1235, 1258, 1224, 1278, 1325, 1324, 1373, 1303, 1340, 1338, 1339, 1417, 1476, 1418, 1477, 1557, 1606, 1680, 1603, 1604, 1654, 1755, 1655, 1753, 1774, 1818, 1851, 1982, 1926, 2172, 2254, 2320, 2301, 2422, 2333, 2542, 2477, 2438, 2523, 2498, 2666]
    ridge_indices = [7, 8, 9, 25, 24, 32, 34, 18, 19, 39, 40, 33, 31, 30, 46, 42, 44, 43, 61, 62, 45, 78, 70, 69, 80, 99, 73, 90, 91, 112, 114, 131, 115, 108, 147, 133, 159, 188, 169, 208, 195, 222, 216, 236, 217, 269, 233, 256, 234, 238, 235, 257, 237, 240, 241, 271, 297, 310, 300, 311, 319, 301, 320, 333, 370, 379, 357, 380, 368, 454, 422, 468, 420, 466, 464, 484, 465, 503, 557, 604, 558, 603, 601, 560, 566, 559, 562, 568, 561, 567, 574, 571, 572, 600, 602, 639, 670, 707, 735, 712, 736, 711, 739, 758, 915, 940, 913, 2290, 2311, 3126, 3127]

    # """
    # P2ILF-10
    # """
    # ligament_indices = [1573, 1758, 1695, 1796, 1825, 1827, 1900, 1901, 1934, 1975, 1870, 4021, 1935, 2050, 2025, 1902, 2023, 2022, 2087, 2024, 2088, 2156, 2158, 2157, 2192, 2193, 2228, 2257, 2226, 2285, 2350, 2415, 2310, 2413, 2351, 2414, 2410, 2411, 2512, 2561, 2490, 2564, 2591, 2633, 2567, 2634, 2635, 2697, 2696, 2655, 2694, 2734, 2789, 2836, 2790, 2917, 2916, 2878, 2979, 2980, 2981, 2984, 3075, 3148, 3149, 3146]
    # ridge_indices = [106, 121, 141, 140, 142, 154, 159, 155, 190, 162, 170, 196, 175, 181, 195, 191, 194, 209, 246, 220, 235, 245, 271, 278, 276, 258, 277, 288, 299, 312, 323, 322, 356, 370, 364, 371, 379, 407, 396, 436, 437, 481, 504, 480, 490, 513, 512, 527, 505, 537, 557, 570, 602, 538, 568, 600, 588, 623, 642, 599, 640, 686, 681, 711, 710, 682, 680, 730, 709, 683, 684, 685, 748, 679, 728, 729, 814, 753, 807, 806, 826, 727, 750, 781, 752, 834, 784, 773, 778, 751, 813, 772, 776, 774, 804, 775, 808, 827, 777, 810, 829, 780, 811, 812, 892, 805, 809, 917, 881, 876, 825, 879, 891, 882, 884, 918, 943, 1007, 883, 1081, 890, 944, 945, 1073, 976, 977, 1076, 1046, 1137, 1074, 1045, 1186, 1252, 1182, 1080, 1078, 1136, 1285, 1184, 1189, 1286, 1287, 4010, 1402, 1401, 1465, 1695, 4011]
    
    # 将顶点颜色设置为灰色
    colors = torch.full_like(mesh.verts_packed(), 0.0, device=device)  # 灰色，RGB值为0.5
    colors[ridge_indices] = torch.tensor([1.0, 0.0, 0.0], device=device)  # 脊顶点着红色
    colors[ligament_indices] = torch.tensor([0.0, 0.0, 1.0], device=device)  # 韧带顶点着蓝色

    # 创建一个包含颜色信息的Textures对象
    vertex_colors = TexturesVertex(verts_features=colors.unsqueeze(0))
    # mesh.textures = vertex_colors
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    verts = utils.center_mesh(verts)

    mesh = Meshes(
        verts=[verts * scale],
        faces=[faces],
        textures=vertex_colors
    )

    # ... 加载 2D 图像代码 ...
    if path_image.endswith(".jpg"):

        image = Image.open(path_image).convert('RGB')  # 确保转换为RGB模式
        # print("image.size:", image.size)
        # 这里需要使用 torchvision 的 ToTensor() 转换
        transform = T.Compose([T.ToTensor()])
        image_label = transform(image).to(device)
        # print("image_label.shape:", image_label.shape)
        image_label = image_label.unsqueeze(0)
        # 现在 permute 操作
        image_label = image_label.permute(0, 2, 3, 1)  # 注意这里的顺序
        

        
    elif path_image.endswith(".png"):
        # print("path_image:", path_image)
        image = Image.open(path_image).convert('RGB')  # 确保转换为RGB模式
        # print("image.size:", image.size)
        # 这里需要使用 torchvision 的 ToTensor() 转换
        transform = T.Compose([T.ToTensor()])
        image_label = transform(image).to(device)
        image_label = image_label.unsqueeze(0)
        # 现在 permute 操作
        image_label = image_label.permute(0, 2, 3, 1)  # 注意这里的顺序
        
    if dilate:
        image_label = utils.dilate_image(image_label)
    if image_scale_factor < 1.0:
        img = image_label.permute(0, 3, 1, 2)
        img = torch.nn.functional.interpolate(
            img,
            scale_factor=image_scale_factor,
            mode="bilinear",
        )
        image_label = img.permute(0, 2, 3, 1)


    camera_params = utils.load_camera_parameters_xml(path_camera_param, scale_factor=image_scale_factor)


    return mesh, image_label, camera_params



def setup_render(
    camera_params,
    # image_size=(1080, 1920),
    R=torch.eye(3).unsqueeze(0),
    tvec=torch.zeros(1, 3)
):
    image_size = [int(camera_params["height"]), int(camera_params["width"])]

    # Create a perspective camera
    camera_matrix =  utils.construct_camera_matrix(camera_params )
    cameras = cameras_from_opencv_projection( 
            R=R, 
            tvec = tvec, 
            camera_matrix = camera_matrix,
            image_size=torch.Tensor(image_size).unsqueeze(0)
        ).to(device)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0,0,0))


    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0, 
        # blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=10,
    )
    # We can add a point light in front of the object. 
    #lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=LightlessShader(blend_params=blend_params, device=device)#, cameras=cameras)#, lights=lights)
        # shader=SoftSilhouetteShader(blend_params=blend_params),
    )

    return renderer


def random_init_mesh_pos():

    init_obj_T = [
        random.uniform(-0.05, 0.05),
        random.uniform(-0.05, 0.05),
        random.uniform(0.13, 0.22),
    ]
    init_obj_T = torch.tensor(init_obj_T) #.detach().to(device)


    """完全随机化，但是需要大量的初始化才能找到对应的正确位置"""
    ang_x = random.uniform(0, 2 * math.pi)  # 0 到 360 度
    ang_y = random.uniform(0, 2 * math.pi)  # 0 到 360 度
    ang_z = random.uniform(0, 2 * math.pi)  # 0 到 360 度


    euler_x = torch.Tensor((ang_x,0,0)).unsqueeze(0)
    euler_y = torch.Tensor((0,ang_y,0)).unsqueeze(0)
    euler_z = torch.Tensor((0,0,ang_z)).unsqueeze(0)
    rot_x = axis_angle_to_quaternion(euler_x)
    rot_y = axis_angle_to_quaternion(euler_y)
    rot_z = axis_angle_to_quaternion(euler_z)

    init_obj_R = quaternion_multiply(quaternion_multiply( rot_x, rot_z ), rot_y)
    init_obj_R = quaternion_to_matrix( init_obj_R )

    # init_obj_R = init_obj_R.to(device)

    return init_obj_R, init_obj_T



def register(
        n_trials,
        n_iter,
        output_folder,
        log_step=30,
):
    output_folder = os.path.join(output_folder, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    mesh, image_label, camera_params = load_data(
        path_mesh, 
        path_image, 
        path_camera_param,
        image_width=382,
        )
    renderer = setup_render(camera_params,)

    losses = []
    minminlossindex = 0
    for idx_trial in range(n_trials):
        print("===========================================================Trial:", idx_trial)
        init_obj_R, init_obj_T = random_init_mesh_pos()
        print("init_obj_R:", init_obj_R)
        print("init_obj_T:", init_obj_T)

        model = Model(
            meshes=mesh, 
            renderer=renderer, 
            dilate_labels=True,
            initial_obj_pos=init_obj_T, #(0,0,0), 
            initial_obj_rot=init_obj_R, #torch.eye(3),
            device=device,
        )
        model = model.to(device)
        # 执行初始对齐，并检查返回值
        if not model.initial_alignment(image_label):
            print("Alignment failed, skipping to the next trial.")
            losses.append(1000000)
            continue  # 如果对齐失败，则跳过当前循环的剩余部分
        
        # model.initial_alignment(image_label)
        lr_T = 1e-9
        lr_R = 1e-5
        optimizer = torch.optim.SGD([
            {"params": [model.T], "lr": lr_T},
            {"params": [model.R], "lr": lr_R},
        ])

        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_iter)

        
        trial_dir = os.path.join(output_folder, "trial_{:02d}".format(idx_trial))
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)


        loss_list = []
        for idx_iter in range(n_iter):
            if idx_iter%5 == 0:
                print("--------------------Iteration:", idx_iter)
            mesh_label_rendered, mesh_gray_rendered, mesh_edge_1  = model()
            
            """
            重新保存每一个打印的图片
            """
            output_path_lab = os.path.join(trial_dir, "test_rendered_image_{:03d}_lab.png".format(idx_iter))
            transform = torchvision.transforms.ToPILImage()
            img_cpu = transform(mesh_label_rendered.squeeze().permute(2,0,1))
            img_cpu.save(output_path_lab)

            output_path_img = os.path.join(trial_dir, "test_rendered_image_{:03d}_img.png".format(idx_iter))
            transform = torchvision.transforms.ToPILImage()
            img_cpu = transform(mesh_gray_rendered.squeeze().permute(2,0,1))
            img_cpu.save(output_path_img)
            
            

            loss, _ = overlap_loss(
                img = mesh_label_rendered, 
                ref = image_label, 
                ligament_weight=5, silhouette_weight=2, black_mask=True
            )
            

            print("loss:", loss)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list.append(loss.item())
            print("min loss:", min(loss_list))
            if len(losses) > 0:
                if loss.item() < min(losses):
                    print("idx_iter:", idx_iter)
                    minminlossindex = idx_iter


        losses.append(min(loss_list))

    print("losses:", losses)
    best_trials = torch.topk(
        torch.tensor(losses), 
        k=int(0.2 * n_trials) if n_trials > 5 else 1, 
        largest=False, 
        sorted=True,
        )
    print("best trials:", best_trials)
    print("minminlossindex:", minminlossindex)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="/home/XXX/P2ILF22_patient7.obj")
    parser.add_argument("--label_3d", type=str, default="")

    parser.add_argument("--label_2d", type=str, default="/home/XXX/P2ILF22_patient7_14.png")
    parser.add_argument("--path_camera_param", type=str, default="/home/XXX/calibration.xml")
    parser.add_argument("--output_folder", type=str, default="./output")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_iter", type=int, default=150)

    args = parser.parse_args()
    path_mesh = args.mesh
    path_mesh_label = args.label_3d
    path_image = args.label_2d
    path_camera_param = args.path_camera_param
    output_folder = args.output_folder
    n_iter = args.n_iter
    n_trials = args.n_trials


    register(
        n_trials=n_trials,
        n_iter=n_iter,
        output_folder=output_folder,
    )




