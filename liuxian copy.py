import numpy as np
import pydicom
import pyvista as pv
import matplotlib.pyplot as plt
from mayavi import mlab

# 读取 DICOM 文件
def read_dicom_array(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    return arr, ds

# 路径
FHP_path = 'S0/S11040/I11'
RLP_path = 'S0/S11070/I11'
APP_path = 'S0/S11050/I11'

# 读取数据
FHP_raw, info_FHP = read_dicom_array(FHP_path)
RLP_raw, info_RLP = read_dicom_array(RLP_path)
APP_raw, info_APP = read_dicom_array(APP_path)

num_slices = 80
num_timepoints = int(1600 / num_slices)  # 20
shape = (num_timepoints, num_slices, 128, 128)

FHP = FHP_raw.reshape(shape)
RLP = RLP_raw.reshape(shape)
APP = APP_raw.reshape(shape)

# 选取某一帧（如第1帧）进行三维速度场重建
t = 0
Vx = FHP[t, :, :, :]  # (80, 128, 128)
Vy = RLP[t, :, :, :]
Vz = APP[t, :, :, :]

# 转换为 (X, Y, Z) 顺序
Vx = np.transpose(Vx, (2, 1, 0))  # (128, 128, 80)
Vy = np.transpose(Vy, (2, 1, 0))
Vz = np.transpose(Vz, (2, 1, 0))

# 归一化参数
P0 = 2048  # 中心点
SCALE = 2048
VENC = 150
# 归一化
FHP = (FHP_raw - P0) / SCALE * VENC
RLP = (RLP_raw - P0) / SCALE * VENC
APP = (APP_raw - P0) / SCALE * VENC


# 计算速度幅值
Vmag = np.sqrt(Vx**2 + Vy**2 + Vz**2)
print(Vmag.shape)
print('速度幅值 min:', Vmag.min(), 'max:', Vmag.max())

# 构建网格
x, y, z = np.arange(128), np.arange(128), np.arange(80)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 展示速度幅值的三维等值面
mlab.figure(bgcolor=(1,1,1))
mlab.contour3d(Vmag, contours=8, opacity=0.4)
mlab.show()

