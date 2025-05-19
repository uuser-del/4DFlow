import os
import numpy as np
import pydicom
from skimage import measure
import pyvista as pv


# 读取MM DICOM文件
mm_path = 'S0/S11030/I11'
mm_raw = pydicom.dcmread(mm_path).pixel_array.astype(np.float32)
# print(mm_raw.shape)(1600, 128, 128)
# 参数（根据实际情况调整）
num_slices = 80
num_timepoints = int(1600 / num_slices)  # 20
print(num_timepoints)
# 重塑为4D
MM = mm_raw.reshape((num_slices, num_timepoints, 128, 128))
print(MM.shape) # (20, 80, 128, 128)
# 输出目录
output_dir = 'mm_3d_models'
os.makedirs(output_dir, exist_ok=True)

# 遍历每一帧，重建三维模型
for t in range(20):
    print(t)
    # 取出当前帧的三维体数据
    mm_3d = MM[:, t, :,  :]  # (20, 80, 128, 128)
    # 转换为 (Z, Y, X) 顺序
    mm_3d = np.transpose(mm_3d, (2, 1, 0))
    # 选择阈值（可根据实际情况调整）
    threshold = np.percentile(mm_3d, 70)  # 取较高分位数，避免噪声
    print(threshold)
    # verts, faces, normals, values = measure.marching_cubes(mm_3d, level=threshold)
    verts, faces, normals, values = measure.marching_cubes(mm_3d)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(verts, faces_pv)
    # print(mesh)
    # 保存为STL
    # stl_path = os.path.join(output_dir, f'mm_frame_{t+1}.stl')
    # mesh.save(stl_path)
    # print(f'已保存: {stl_path}', mesh)
    # 可视化（可选）
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='red', opacity=0.5)
    plotter.show_grid()
    plotter.show()
    break