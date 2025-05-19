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

from toool import speed_calculate
Vmag = speed_calculate(FHP_raw, RLP_raw, APP_raw)
Vmag = Vmag[:, 4, :, :]

# 构建网格
x, y, z = np.arange(80), np.arange(128), np.arange(128)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 创建点云数据
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
values = Vmag.flatten()

# 设置速度阈值，只显示速度大于阈值的点
speed_threshold = 50  # 可以根据需要调整阈值
mask = values > speed_threshold
points = points[mask]
values = values[mask]

# 创建可视化窗口
mlab.figure(bgcolor=(0.3,0.3,0.3), size=(1200, 800))

# 使用点云方式显示速度值
pts = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], values,
                   scale_mode='none',
                   scale_factor=1.0,
                   colormap='jet',
                   opacity=0.4)

# 添加颜色条
mlab.colorbar(pts, title='速度幅值', orientation='vertical')

# 设置视角
mlab.view(azimuth=45, elevation=45, distance='auto')

# 添加坐标轴标签
mlab.text3d(X.max(), 0, 0, 'X', scale=5)
mlab.text3d(0, Y.max(), 0, 'Y', scale=5)
mlab.text3d(0, 0, Z.max(), 'Z', scale=5)

# 显示图形
mlab.show()
