# v1 简单的损失函数

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_size=120, hidden_size=100, laten_size=8):
        # encoder， decoder都是两个隐藏层
        super(VAE, self).__init__()
        self.win = input_size
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




    def forward(self, x, n_sample=1):
        """
        :param x:
        :param n_sample: z的采样次数
        :return:
        """
        if self.training:
            # 训练时z的采样次数为1
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

        else:
            out_z_miu, out_z_std = self.encoder(x), self.encoder(x)
            z_miu = self.en_miu(out_z_miu)
            z_std = self.en_std(out_z_std) + self.epsilon
            # 测试时z的采样次数为 n_sample, 直接采样，不用重参数化
            batch_size = z_miu.shape[0]
            z_miu = z_miu.repeat(n_sample, 1).view(n_sample, batch_size, -1)  # size: n_sample * batch_size * win
            z_std = z_std.repeat(n_sample, 1).view(n_sample, batch_size, -1)
            z = torch.normal(mean=z_miu, std=z_std)  # size: n_sample * batch_size * win

            # Generative
            out_x_miu, out_x_std = self.decoder(z), self.decoder(z)
            x_miu = self.de_miu(out_x_miu)
            x_std = self.de_std(out_x_std) + self.epsilon
            # gen_x = torch.normal(mean=x_miu, std=x_std)  # 直接采样
            return z, x_miu, x_std, z_miu, z_std






