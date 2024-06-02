# 1. 导入所需的模块
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl


# 2. 定义编码器和解码器
# 2.1 定义基础编码器Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


# 2.2 定义基础解码器Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


# 3. 定义LightningModule
class LitAutoEncoder(pl.LightningModule):

    # 3.1 加载基础模型
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 3.2 训练过程设置
    def training_step(self, batch, batch_idx):  # 每一个batch数据运算计算loss
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    # 3.3 优化器设置
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# 4. 定义训练数据
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

# 5. 实例化模型
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# 6. 开始训练
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)