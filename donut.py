import torch
from VAE import VAE
from data import TsDataset
from torch.optim import Adam
import math


class Donut:
    def __init__(self):
        self._vae = VAE()


    def m_elbo_loss(self, train_x, train_y, z, x_miu, x_std, z_miu, z_std):
        """
        采用蒙特卡洛估计， 根据论文，设置采样次数 L=1
        :param train_x: batch_size * win, 样本
        :param train_y: batch_size * 1, 标签， 0代表正常， 1代表异常
        :param z: batch_size * latent_size, 观察到的隐变量
        :param x_miu: batch_size * win, x服从正态分布的均值
        :param x_std: batch_size * win, x服从正态分布的标准差
        :param z_miu: batch_size * latent_size, 后验z服从正态分布的均值
        :param z_std: batch_size * latent_size, 后验z服从正态分布的标准差
        :param z_prior_mean: int, 先验z服从正态分布的均值， 一般设置为0
        :param z_prior_std: int, 先验z服从正态分布的标准差， 一般设置为1
        :return:
        """
        z_prior_mean = torch.zeros(size=z_miu.shape)
        z_prior_std = torch.ones(size=z_miu.shape)

        # 以下蒙特卡洛估计的采样次数均为1
        # 蒙特卡洛估计 log p(x|z)。在重构的x的正态分布上， 取值为train_x时，概率密度函数的log值； batch_size * win
        log_p_xz = - torch.log(math.sqrt(2 * math.pi) * x_std) - ((train_x - x_miu) ** 2) / (2 * x_std ** 2)

        # 蒙特卡洛估计 log p(z). p(z)为先验分布， 一般设置为标准正态分布； batch_size * latent_size
        log_p_z = - torch.log(math.sqrt(2 * math.pi) * z_prior_std) - ((z - z_prior_mean) ** 2) / (2 * z_prior_std ** 2)

        # 蒙特卡洛估计 log q(z|x). q(z|x)为z的后验分布; batch_size * latent_size
        log_q_zx = - torch.log(math.sqrt(2 * math.pi) * z_std) - ((z - z_miu) ** 2) / (2 * z_std ** 2)

        # 去除缺失点的影响，也是m-elbo的精髓
        normal = 1 - train_y  # batch_size * win
        log_p_xz = normal * log_p_xz  # batch_size * win

        # beta, log_p_z 的放缩系数
        beta = torch.sum(normal, dim=1) / normal.shape[1]  # size = batch_size

        # m-elbo的值
        m_elbo = torch.sum(log_p_xz, dim=1) + beta * torch.sum(log_p_z, dim=1) - torch.sum(log_q_zx, dim=1)
        m_elbo = torch.mean(m_elbo) * (-1)
        return m_elbo


    def fit(self, x, y, n_epoch=30, valid_x=None, valid_y=None):
        '''
        如果在实际应用中没有标签， 可以把大多数样本当做正常样本， 即把y全部设置为0
        :param x:
        :param y:
        :param n_epoch:
        :param valid_x:
        :param valid_y:
        :return:
        '''
        # todo missing injection
        train_dataset = TsDataset(x, y)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

        if valid_x is not None:
            valid_dataset = TsDataset(valid_x, valid_y)
            valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=0)

        optimizer = Adam(self._vae.parameters(), lr=0.001)  # todo 动态学习率
        for epoch in range(n_epoch):
            for train_x, train_y in train_iter:
                optimizer.zero_grad()
                z, x_miu, x_std, z_miu, z_std = self._vae(train_x)  # 前向传播
                l = self.m_elbo_loss(train_x, train_y, z, x_miu, x_std, z_miu, z_std)
                l.backward()
                optimizer.step()
            if valid_x is not None:
                with torch.no_grad():
                    for v_x, v_y in valid_iter:
                        z, x_miu, x_std, z_miu, z_std = self._vae(v_x)
                        v_l = self.m_elbo_loss(v_x, v_y, z, x_miu, x_std, z_miu, z_std)
                print("train loss %.4f,  valid loss %.4f" %(l.item(), v_l.item()))
            else:
                print("loss", l.item())














            # break



        # pass


    def evaluate(self):
        pass