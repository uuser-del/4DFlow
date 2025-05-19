import numpy as np
import pydicom
import os
from toool import speed_calculate

def read_dicom_array(path):
    """读取DICOM文件"""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    return arr, ds

# 路径
FHP_path = 'S0/S11040/I11'
RLP_path = 'S0/S11070/I11'
APP_path = 'S0/S11050/I11'

# 使用固定的空间参数
pixel_spacing = 1.9531  # 像素间距 (mm)
slice_thickness = 2.409  # 层厚 (mm)

# 构建网格（考虑实际物理尺寸）
x = np.arange(80) * slice_thickness
y = np.arange(128) * pixel_spacing
z = np.arange(128) * pixel_spacing
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 读取数据
print("正在读取DICOM数据...")
FHP_raw, info_FHP = read_dicom_array(FHP_path)
RLP_raw, info_RLP = read_dicom_array(RLP_path)
APP_raw, info_APP = read_dicom_array(APP_path)

# 计算速度幅值
print("正在计算速度幅值...")
Vmag = speed_calculate(FHP_raw, RLP_raw, APP_raw)

# 加载掩码
print("正在加载掩码...")
mask_path = os.path.join('mask_data', 'aorta_mask.npy')
if not os.path.exists(mask_path):
    raise FileNotFoundError("找不到掩码文件，请先运行 calculate_mask.py")
aorta_mask = np.load(mask_path)

# 创建点云数据
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

# 设置速度阈值，只保存速度大于阈值的点
speed_threshold = 0  # 可以根据需要调整阈值

# 创建保存目录
save_dir = 'point_cloud_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 处理所有时间点的数据
num_timepoints = Vmag.shape[0]  # 获取时间点数量
print(f"开始处理 {num_timepoints} 个时间点的数据...")

for t in range(num_timepoints):
    print(f"\n处理时间点 {t+1}/{num_timepoints}")
    
    # 获取当前时间点的速度幅值
    current_vmag = Vmag[t]
    
    # 应用掩码到当前时间点的Vmag数据
    masked_vmag = current_vmag.copy()
    masked_vmag[~aorta_mask] = 0
    
    # 创建点云数据
    masked_values = masked_vmag.flatten()
    
    # 创建掩码
    mask_masked = masked_values > speed_threshold
    
    # 获取分割后的点和速度值
    masked_points = points[mask_masked]
    masked_velocities = masked_values[mask_masked]
    
    # 创建时间点目录
    timepoint_dir = os.path.join(save_dir, f'timepoint_{t+1}')
    if not os.path.exists(timepoint_dir):
        os.makedirs(timepoint_dir)
    
    # 保存分割后的点云数据
    np.save(os.path.join(timepoint_dir, 'masked_points.npy'), masked_points)
    np.save(os.path.join(timepoint_dir, 'masked_velocities.npy'), masked_velocities)
    
    print(f"时间点 {t+1} 的数据已保存到 {timepoint_dir}")

print("\n所有时间点的数据处理完成！") 