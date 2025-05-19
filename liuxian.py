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
shape = (num_slices,num_timepoints,  128, 128)

FHP = FHP_raw.reshape(shape)
RLP = RLP_raw.reshape(shape)
APP = APP_raw.reshape(shape)

# 选取某一帧（如第1帧）进行三维速度场重建
t = 0
Vx = FHP[ :,t, :, :]  # (80, 128, 128)
Vy = RLP[ :,t, :, :]
Vz = APP[ :,t, :, :]

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

# 画速度幅值的三个正交切片
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Vmag[64, :, :], cmap='coolwarm', origin='lower')
axes[0].set_title('Sagittal (X=64)')
axes[1].imshow(Vmag[:, 64, :], cmap='coolwarm', origin='lower')
axes[1].set_title('Coronal (Y=64)')
axes[2].imshow(Vmag[:, :, 40], cmap='coolwarm', origin='lower')
axes[2].set_title('Axial (Z=40)')
# plt.show()

# 构建三维坐标网格
x, y, z = np.mgrid[0:128, 0:128, 0:80]

# 画三维流线
mlab.figure(bgcolor=(1,1,1), size=(800, 600))
mlab.flow(x, y, z, Vx, Vy, Vz, seedtype='sphere', integration_direction='both', line_width=2, colormap='cool')
mlab.outline()
mlab.title('3D Streamlines of Velocity Field')
mlab.show()

# 用 ImageData 替代 UniformGrid
grid = pv.UniformGrid()
grid.dimensions = (128, 128, 80)
grid.spacing = (1, 1, 1)
grid.origin = (0, 0, 0)
grid['vectors'] = np.transpose(np.array([Vx.flatten(), Vy.flatten(), Vz.flatten()]), (1, 0)).reshape(-1, 3, order='F')

# 生成流线
streamlines = grid.streamlines('vectors', n_points=100, max_time=200.0, initial_step_length=1.0, terminal_speed=1e-3)

