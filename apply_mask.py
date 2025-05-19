import numpy as np
import pydicom
from mayavi import mlab
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

# 设置速度阈值，只显示速度大于阈值的点
speed_threshold = 0  # 可以根据需要调整阈值

# 处理所有时间点的数据
num_timepoints = Vmag.shape[0]  # 获取时间点数量
print(f"开始处理 {num_timepoints} 个时间点的数据...")
print("按'q'键可以停止程序")

for t in range(num_timepoints):
    print(f"\n处理时间点 {t+1}/{num_timepoints}")
    
    # 获取当前时间点的速度幅值
    current_vmag = Vmag[t]
    
    # 应用掩码到当前时间点的Vmag数据
    masked_vmag = current_vmag.copy()
    masked_vmag[~aorta_mask] = 0
    
    # 创建点云数据
    original_values = current_vmag.flatten()
    masked_values = masked_vmag.flatten()
    
    # 创建掩码
    mask_original = original_values > speed_threshold
    mask_masked = masked_values > speed_threshold
    
    # 创建可视化窗口
    fig = mlab.figure(bgcolor=(0.3,0.3,0.3), size=(1200, 800))
    
    # 显示原始Vmag点云（灰色）
    pts_original = mlab.points3d(points[mask_original, 0], 
                               points[mask_original, 1], 
                               points[mask_original, 2], 
                               original_values[mask_original],
                               scale_mode='none',
                               scale_factor=1.0,
                               colormap='gray',
                               opacity=0.2)
    
    # 显示分割后的Vmag点云（红色）
    pts_masked = mlab.points3d(points[mask_masked, 0], 
                              points[mask_masked, 1], 
                              points[mask_masked, 2], 
                              masked_values[mask_masked],
                              scale_mode='none',
                              scale_factor=1.0,
                              colormap='hot',
                              opacity=0.4)
    
    # 添加颜色条
    mlab.colorbar(pts_original, title='原始速度幅值', orientation='vertical')
    mlab.colorbar(pts_masked, title='分割后速度幅值', orientation='vertical')
    
    # 设置视角
    mlab.view(azimuth=45, elevation=45, distance='auto')
    
    # 添加坐标标签
    mlab.text3d(X.max(), 0, 0, 'X', scale=5, color=(1,1,1))
    mlab.text3d(0, Y.max(), 0, 'Y', scale=5, color=(1,1,1))
    mlab.text3d(0, 0, Z.max(), 'Z', scale=5, color=(1,1,1))
    
    # 添加时间点信息
    mlab.title(f'时间点 {t+1}', size=0.5)
    
    # 显示图形并等待用户交互
    @mlab.animate(delay=100)
    def anim():
        while True:
            if mlab.get_engine().scenes[0].scene.interactor.GetKeySym() == 'q':
                mlab.close(all=True)
                print("\n程序已停止")
                return
            yield
    
    anim()
    mlab.show() 