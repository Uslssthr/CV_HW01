from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# 自定义数据集类
class FloorPlanDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['plans']
        mask = self.dataset[idx]['colors']  # 假设数据集中有mask字段

        # 确保图像只有3个通道
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if mask.mode == 'RGBA':
            mask = mask.convert('RGB')  # 保持掩码为RGB格式

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 加载数据集
dataset = load_dataset('zimhe/pseudo-floor-plan-12k')

# 创建数据集实例
full_train_dataset = FloorPlanDataset(dataset['train'], transform=transform)

# 选择前300个样本
subset_indices = list(range(300))  # 选择前300个样本
train_dataset = Subset(full_train_dataset, subset_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


