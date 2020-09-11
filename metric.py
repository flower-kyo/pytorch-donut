# author:30331
# contact: 16221137@bjtu.edu.cn
# datetime:2020/9/9 17:23
# software: PyCharm
import numpy as np
from sklearn.metrics import precision_recall_curve
from  sklearn.metrics import plot_precision_recall_curve

def best_threshold(y_true, scores):
    """
    :param y_true: np.array
    :param scores: np.array
    :return:
    """
    # scores 标准化
    _min = np.min(scores)
    _max = np.max(scores)
    scores_ = (scores - _min) / (_max - _min + 1e-8)
    pr, re, thrs = precision_recall_curve(y_true, scores_)
    fs = 2.0 * pr * re / np.clip(pr + re, a_min=1e-4, a_max=None)
    # plot_precision_recall_curve(y_true, scores_)
    print("best f score", max(fs))
    return max(fs)







