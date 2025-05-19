import os
import numpy as np
import pydicom
from skimage import measure
import pyvista as pv

def process_input_path(input_path):
    """处理输入的路径"""
    # 移除可能的引号和r前缀
    path = input_path.strip().strip("'").strip('"')
    if path.startswith('r'):
        path = path[1:].strip("'").strip('"')
    
    # 统一路径分隔符
    path = path.replace('\\', '/')
    
    # 如果是相对路径，添加基础路径
    if not os.path.isabs(path):
        base_path = 'D:/4dflow/zhangyaozhong/MR_yaozhong'  # 基础路径
        path = os.path.join(base_path, path)
    
    return path

# 获取用户输入
print("请输入DICOM文件路径：")
# print("支持格式：")
# print("1. 相对路径：S0/S11030/I11")
# print("2. 绝对路径：D:/4dflow/zhangyaozhong/MR_yaozhong/S0/S11030/I11")
mm_path = process_input_path(input().strip())
print("请输入输出文件夹路径（提前建好）：")
output_dir = process_input_path(input().strip())
print(output_dir)

# 读取MM DICOM文件
mm_raw = pydicom.dcmread(mm_path).pixel_array.astype(np.float32)
print("原始数据形状:", mm_raw.shape)
print("总帧数:", mm_raw.shape[0])

# 参数设置
num_timepoints = 20  # 时间点数
num_slices = int(mm_raw.shape[0] / num_timepoints)
print("时间点数:", num_timepoints)
print("切片数:", num_slices)

# 重塑为4D
MM = mm_raw.reshape((num_slices, num_timepoints, mm_raw.shape[1], mm_raw.shape[2]))
print("重塑后形状:", MM.shape)

# # 创建输出目录
# os.makedirs(output_dir, exist_ok=True)

# 遍历每一帧，重建三维模型
for t in range(num_timepoints):
    print(f"正在处理第 {t+1}/{num_timepoints} 帧")
    # 取出当前帧的三维体数据
    mm_3d = MM[:, t, :, :]
    # 转换为 (Z, Y, X) 顺序
    mm_3d = np.transpose(mm_3d, (2, 1, 0))
    # 选择阈值（可根据实际情况调整）
    threshold = np.percentile(mm_3d, 70)  # 取较高分位数，避免噪声
    # print(threshold)
    verts, faces, normals, values = measure.marching_cubes(mm_3d, level=threshold)
    # verts, faces, normals, values = measure.marching_cubes(mm_3d)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(verts, faces_pv)
    # print(mesh)
    # 保存为STL
    folder_name = os.path.basename(output_dir)
    stl_path = os.path.join(output_dir, f'{folder_name}_frame_{t+1}.stl')
    mesh.save(stl_path)
    print(f'已保存: {stl_path}', mesh)
    # 体绘制可视化
    volume = pv.wrap(mm_3d)

    plotter = pv.Plotter()
    plotter.add_volume(volume, cmap="bone", opacity="sigmoid_6", shade=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.set_background('white')
    plotter.show()
    break