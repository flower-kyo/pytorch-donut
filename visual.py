# author:30331
# contact: 16221137@bjtu.edu.cn
# datetime:2020/9/10 16:11
# software: PyCharm
import time
import matplotlib.pyplot as plt
def restruct_compare_plot(x, con_x):
    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.ylabel('orngin')

    plt.subplot(2, 1, 2)
    plt.plot(con_x)
    plt.ylabel('con x')

    plt.savefig("./images/" +str(time.time())+ ".jpg")
    plt.cla()
    plt.close("all")
    # plt.show()

