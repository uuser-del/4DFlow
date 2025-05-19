import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

# 读取 DICOM 文件
def read_dicom_array(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float64)
    return arr, ds

# 路径
FHP_path = 'S0/S11040/I11'
RLP_path = 'S0/S11070/I11'
APP_path = 'S0/S11050/I11'
MM_path  = 'S0/S11030/I11'

# 读取数据
FHP_raw, info_FHP = read_dicom_array(FHP_path)
RLP_raw, info_RLP = read_dicom_array(RLP_path)
APP_raw, info_APP = read_dicom_array(APP_path)
MM_raw,  info_MM  = read_dicom_array(MM_path)

VENC = 150  # 检查 VENC 信息
num_slices = 80
num_timepoints = int(1600 / num_slices)  # 或根据实际情况调整

# 读取并重塑数据
FHP = FHP_raw.reshape(num_timepoints, num_slices, 128, 128)
RLP = RLP_raw.reshape(num_timepoints, num_slices, 128, 128)
APP = APP_raw.reshape(num_timepoints, num_slices, 128, 128)
MM  = MM_raw.reshape(num_timepoints, num_slices, 128, 128)

# 获取 DICOM 文件的位深度
bit_depth = info_FHP.BitsStored if hasattr(info_FHP, 'BitsStored') else None

# 归一化每个时间点
FHP_norm = np.zeros_like(FHP)
RLP_norm = np.zeros_like(RLP)
APP_norm = np.zeros_like(APP)

for t in range(num_timepoints):
    FHP_min, FHP_max = FHP[t, :, :, :].min(), FHP[t, :, :, :].max()
    RLP_min, RLP_max = RLP[t, :, :, :].min(), RLP[t, :, :, :].max()
    APP_min, APP_max = APP[t, :, :, :].min(), APP[t, :, :, :].max()
    FHP_norm[t, :, :, :] = (FHP[t, :, :, :] - FHP_min) / (FHP_max - FHP_min) * 2 - 1
    RLP_norm[t, :, :, :] = (RLP[t, :, :, :] - RLP_min) / (RLP_max - RLP_min) * 2 - 1
    APP_norm[t, :, :, :] = (APP[t, :, :, :] - APP_min) / (APP_max - APP_min) * 2 - 1

# 计算流速
Vx = FHP_norm * VENC
Vy = RLP_norm * VENC
Vz = APP_norm * VENC

# 计算总流速
V_mag = np.sqrt(Vx**2 + Vy**2 + Vz**2)

# 遍历每个时间点和切片，保存速度幅值图像
output_root = 'quan'
os.makedirs(output_root, exist_ok=True)

for t_index in range(num_timepoints):
    timepoint_folder = os.path.join(output_root, f'timepoint_{t_index+1}')
    os.makedirs(timepoint_folder, exist_ok=True)
    for slice_index in range(num_slices):
        V_mag_slice = V_mag[t_index, slice_index, :, :]
        plt.figure()
        plt.imshow(V_mag_slice, vmin=0, vmax=VENC, cmap='jet')
        plt.colorbar()
        plt.title(f'V_mag，时间点：{t_index+1}，切片：{slice_index+1}')
        plt.xlabel('列索引')
        plt.ylabel('行索引')
        plt.savefig(os.path.join(timepoint_folder, f'slice_{slice_index+1}.png'))
        plt.close()