import cv2
import numpy as np
import torch

import image_dataset
import is_model

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载保存的模型
model = is_model.UNet(in_channels=3, out_channels=3)  # 初始化模型
model.load_state_dict(torch.load('saved_model/unet_model.pth', weights_only=True))  # 加载模型状态字典
model.to(device)  # 将模型移动到GPU（如果可用）
model.eval()  # 设置模型为评估模式

# 对模型使用的训练子集以外的数据集中的其中一个图像进行预测
with torch.no_grad():
    image, mask = image_dataset.full_train_dataset[5000]
    image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到GPU
    output = model(image)

# 将预测结果转换为numpy数组以便显示
output_np = output.squeeze().cpu().numpy() * 255  # 将结果移动到CPU
output_np = output_np.astype(np.uint8)
output_np = np.transpose(output_np, (1, 2, 0))  # 将通道顺序从CHW转换为HWC

# 将原始图像转换为numpy数组以便显示
image_np = (image.squeeze().cpu().permute(1, 2, 0).numpy() * 255)  # 将结果移动到CPU
image_np = image_np.astype(np.uint8)

# 将掩码转换为numpy数组以便显示
mask_np = (mask.squeeze().cpu().permute(1, 2, 0).numpy() * 255)  # 将结果移动到CPU
mask_np = mask_np.astype(np.uint8)

# 显示原始图像、预测掩码和真实掩码
cv2.imshow('Original Image', image_np)
cv2.imshow('Predicted Mask', output_np)
cv2.imshow('True Mask', mask_np)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('saved_image/original_image.jpg', image_np)
cv2.imwrite('saved_image/predicted_mask.jpg', output_np)
cv2.imwrite('saved_image/true_mask.jpg', mask_np)

