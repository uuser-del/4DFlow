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

# 展示速度幅值的三维等值面
mlab.figure(bgcolor=(1,1,1))
# 使用网格坐标和归一化后的速度幅值
mlab.contour3d(X, Y, Z, Vmag, contours=32, opacity=0.4)
mlab.colorbar(title='速度幅值', orientation='vertical')
mlab.show()

