import numpy as np
import torch

def proprocess(x, y, slide_win=120):
    """
    Standardize(zero mean ??) and fill missing with zero

    :param sliding_window: sliding window size
    :param x: 1-D array
        origin kpi
    :param y: 1-D array
        label
    :return: zero mean standardized kpi,
    """
    # todo 标准化和用0补充缺失点
    ret_x, ret_y = slide_sampling(x, y, slide_win=slide_win)
    return ret_x, ret_y


def slide_sampling(x, y, slide_win):
    ret_x = []
    ret_y = []
    for i in range(len(x) - slide_win + 1):
        ret_x.append(x[i: i + slide_win])
        # ret_y.append(y[i + slide_win - 1])
        ret_y.append(y[i: i + slide_win])
    ret_x = np.array(ret_x)
    ret_y = np.array(ret_y)
    return ret_x, ret_y


class TsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).int()

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return data, label

    def __len__(self):
        return self.x.shape[0]
