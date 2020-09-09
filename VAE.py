# v1 简单的损失函数

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_size=120, hidden_size=100, laten_size=8):
        # encoder， decoder都是两个隐藏层
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.en_miu = nn.Linear(hidden_size, laten_size)
        # self.en_std = nn.Linear(hidden_size, laten_size)
        self.en_std = nn.Sequential(
            nn.Linear(hidden_size, laten_size),
            nn.Softplus()
        )
        self.epsilon = 0.0001

        self.decoder = nn.Sequential(
            nn.Linear(laten_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.de_miu = nn.Linear(hidden_size, input_size)
        # self.de_std = nn.Linear(hidden_size, input_size)
        self.de_std = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Softplus()
        )

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)




    def forward(self, x):
        # Variational
        out_z_miu, out_z_std = self.encoder(x), self.encoder(x)
        z_miu = self.en_miu(out_z_miu)
        z_std = self.en_std(out_z_std) + self.epsilon
        z = z_miu + z_std * torch.randn(z_miu.shape[0], z_miu.shape[1])  # 重参数化采样

        # Generative
        out_x_miu, out_x_std = self.decoder(z), self.decoder(z)
        x_miu = self.de_miu(out_x_miu)
        x_std = self.de_std(out_x_std) + self.epsilon
        # gen_x = torch.normal(mean=x_miu, std=x_std)  # 直接采样
        return z, x_miu, x_std, z_miu, z_std





