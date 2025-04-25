import json
import torch
import kornia


def load_camera_parameters_xml(camera_param_path, scale_factor=1.0):
    import xml.etree.ElementTree as ET
    tree = ET.parse( camera_param_path )
    root = tree.getroot()

    fx = float(root.find("fx").text)
    fy = float(root.find("fy").text)
    cx = float(root.find("cx").text)
    cy = float(root.find("cy").text)
    calib = {}
    params = ["width", "height", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "skew"]
    for name in params:
        val = float(root.find(name).text)
        calib[name] = val

    calib["fx"] = scale_factor*calib["fx"]
    calib["fy"] = scale_factor*calib["fy"]
    calib["cx"] = scale_factor*calib["cx"]
    calib["cy"] = scale_factor*calib["cy"]
    calib["width"] = scale_factor*calib["width"]
    calib["height"] = scale_factor*calib["height"]

    #print("camera params:", fx, fy, cx, cy)
    return calib


def load_camera_parameters_json(camera_param_path):
    with open(camera_param_path, "r") as f:
        content = json.load(f)
        camera_parameters = {}
        for k, val in content.items():
            try:
                camera_parameters[k] = float(val)
            except ValueError:
                camera_parameters[k] = val
    return camera_parameters


def construct_camera_matrix( params, ):

    K = torch.zeros((1,4,4), dtype=torch.float32)
    K[:, 0, 0] = params["fx"]
    K[:, 1, 1] = params["fy"]
    K[:, 2, 2] = 1
    K[:, 3, 3] = 1
    K[:, 0, 2] = params["cx"]
    K[:, 1, 2] = params["cy"]
    return K



def dilate_image(image):
    image = image.permute(0,3,1,2)
    image = kornia.morphology.dilation(
            image,
            kernel = torch.ones((3, 3)).to("cuda:1")
        )
    image = image.permute(0,2,3,1)
    return image

# def dilate_image(image):
#     # 添加批量维度
#     image = image.unsqueeze(0)
#     # 现在 permute 操作
#     image = image.permute(0, 2, 3, 1)  # 注意这里的顺序
#     image = kornia.morphology.dilation(
#         image,
#         kernel=torch.ones((11, 11)).to("cuda:1")
#     )
#     # image = image.permute(0, 2, 3, 1)  # 恢复原始顺序
#     # # 移除批量维度
#     # image = image.squeeze(0)

#     return image



def center_mesh(verts):
    verts_offset = [0,0,0]
    for dim in range(3):
        dim_min = verts[:,dim].min()
        dim_max = verts[:,dim].max()
        center = (dim_max+dim_min)*0.5
        verts[:,dim] = verts[:,dim] - center
        verts_offset[dim] = center
    # print("-------verts_offset:", verts_offset)
    return verts


