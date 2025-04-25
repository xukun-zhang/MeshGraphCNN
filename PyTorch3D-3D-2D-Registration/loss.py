import torch
import torchvision.transforms as T
from PIL import Image

def save_tensor_as_png(ref_tensor, file_path):
    # 确保张量在 CPU 上，并从计算图中分离
    image_tensor = ref_tensor.detach().cpu()
    # 去除批次维度，形状变为 (H, W, 3)
    image_tensor = image_tensor.squeeze(0)
    # 由于 torchvision.transforms.ToPILImage() 期望输入形状为 (C, H, W)，需要调整维度
    image_tensor = image_tensor.permute(2, 0, 1)  # 形状变为 (3, H, W)
    # 将张量的像素值范围调整为 [0, 255]
    # 如果您的像素值已经在 [0, 1]，需要乘以 255 并转换为 uint8 类型
    image_tensor = image_tensor * 255.0
    image_tensor = image_tensor.type(torch.uint8)
    # 使用 torchvision.transforms 将张量转换为 PIL 图像
    transform = T.ToPILImage()
    image = transform(image_tensor)
    # 保存图像为 PNG 文件
    image.save(file_path)
    # print(f"图像已保存至 {file_path}")



def overlap_loss(img, ref, ligament_weight=5, silhouette_weight=2, black_mask=True ):
    save_tensor_as_png(ref, 'ref_image.png')
    inds = torch.argmax( ref, dim=3 )
    
    c_sum = ref[..., 0] + ref[..., 1] + ref[..., 2]
    o_sum = img[..., 0] + img[..., 1] + img[..., 2]
    save_tensor_as_png(img[..., :3], 'output_image.png')
    diff = img[..., :3] - ref[..., :3]

    if ligament_weight != 1:
        inds = torch.argmax( ref, dim=3 )
        lig_mask = inds == 2
        lig_mask[c_sum<0.5]=0

    if silhouette_weight != 1:
        inds = torch.argmax( ref, dim=3 )
        sil_mask = inds == 1
        sil_mask[c_sum<0.5]=0

    if silhouette_weight != 1:
        inds = torch.argmax( ref, dim=3 )
        rid_mask = inds == 0
        rid_mask[c_sum<0.5]=0

    land_loss = 3*torch.sum((diff[lig_mask]) ** 2) + 2*torch.sum((diff[rid_mask]) ** 2) + 2*torch.sum((diff[sil_mask]) ** 2)
    # liver_loss = 0.0001 * torch.sum((diff_liver) ** 2)
    loss = land_loss  #+ liver_loss
    print("land_loss:", land_loss)
    return loss, diff

import torch
import torch.nn.functional as F
from chamferdist import ChamferDistance

def extract_foreground_points(image, color_index):
    # 计算颜色索引掩码
    mask = torch.argmax(image[..., :3], dim=3) == color_index
    points = torch.nonzero(mask).float()  # 获取前景像素的坐标
    return points

def compute_chamfer_distance(img, ref, color_index):
    chamfer_dist = ChamferDistance()
    img_points = extract_foreground_points(img, color_index)
    ref_points = extract_foreground_points(ref, color_index)
    
    if img_points.numel() == 0 or ref_points.numel() == 0:
        return torch.tensor(0.0, device=img.device)  # 如果某个颜色区域没有前景点，返回0损失

    # 计算 Chamfer 距离
    loss = chamfer_dist(img_points.unsqueeze(0), ref_points.unsqueeze(0))
    return loss

def combined_loss(img, ref, ligament_weight=5, silhouette_weight=2, black_mask=True):
    # 计算 L2 损失
    diff = img[..., :3] - ref[..., :3]
    inds = torch.argmax(ref, dim=3)

    lig_mask = inds == 2
    # sil_mask = inds == 1
    rid_mask = inds == 0

    diff[lig_mask] *= 5
    # diff[sil_mask] *= 2
    diff[rid_mask] *= 2

    if black_mask:
        mask = ref.sum(dim=3, keepdim=True) > 0.001
        mask = mask.repeat(1, 1, 1, 3)
        diff[mask == False] *= 1e-2
    l2_loss = torch.sum(diff ** 2)
    # 计算 Chamfer Distance 损失
    chamfer_loss_ridge = compute_chamfer_distance(img, ref, color_index=0)  # 红色 (肝脊)
    # chamfer_loss_silhouette = compute_chamfer_distance(img, ref, color_index=1)  # 绿色 (肝脏上面边缘)
    chamfer_loss_ligament = compute_chamfer_distance(img, ref, color_index=2)  # 蓝色 (韧带)
    # print("l2_loss, chamfer_loss_ridge, chamfer_loss_silhouette, chamfer_loss_ligament:", l2_loss, chamfer_loss_ridge, chamfer_loss_silhouette, chamfer_loss_ligament)
    # 合并损失
    total_loss = l2_loss + (chamfer_loss_ridge + chamfer_loss_ligament)
    # total_loss = (chamfer_loss_ridge + chamfer_loss_silhouette + chamfer_loss_ligament)
    return total_loss
