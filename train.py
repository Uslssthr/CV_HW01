import torch
import torch.nn as nn
import torch.optim as optim

import image_dataset
import is_model

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型、损失函数和优化器
model = is_model.UNet(in_channels=3, out_channels=3).to(device)  # 修改输出通道数为3
criterion = nn.MSELoss()  # 使用均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}] beginning')
    model.train()
    running_loss = 0.0
    for images, masks in image_dataset.train_loader:
        images, masks = images.to(device), masks.to(device)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(image_dataset.train_loader):.4f}')

# 保存训练结束后的模型
model_path = 'saved_model/unet_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

