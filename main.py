import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import proprocess
from donut import Donut
# 准备数据， 读取数据, 只使用第一条 kpi
# kpi_df1 = pd.read_csv("C:\\Users\\30331\\Desktop\\异常检测组\\codes\\PyADTS\\tests\\data\\kpi\\phase2_train.csv")
# # kpi_df2 = pd.read_dhf("C:\Users\\30331\Desktop\异常检测组\codes\PyADTS\tests\data\kpi\phase2_ground_truth.hdf")
# kpi = kpi_df1[kpi_df1["KPI ID"].apply(str) == "da10a69f-d836-3baa-ad40-3e548ecf1fbd"]
# 读取数据
kpi = pd.read_csv("./Test.csv")
x, y = kpi['value'].values, kpi['label'].values
assert len(x) == len(y)
train_spilt, val_spilt = 0.6, 0.8
train_x, valid_x, test_x = x[: int(len(x) * train_spilt)], x[int(len(x) * train_spilt): int(len(x) * val_spilt)], x[int(len(x) * val_spilt):]
train_y, valid_y, test_y = y[: int(len(y) * train_spilt)], y[int(len(y) * train_spilt): int(len(y) * val_spilt)], y[int(len(y) * val_spilt):]
train_x, train_y = proprocess(train_x, train_y)
valid_x, valid_y = proprocess(valid_x, valid_y)
test_x, test_y = proprocess(test_x, test_y)

# 模型训练
model = Donut()
model.fit(train_x, train_y, n_epoch=300, valid_x=valid_x, valid_y=valid_y)
model.evaluate(test_x, test_y)

