import numpy as np
import trimesh
from scipy.spatial import cKDTree
from mayavi import mlab

# 读取STL文件
def read_stl(stl_path):
    mesh = trimesh.load(stl_path)
    return mesh.vertices, mesh.faces

# 路径
mm_stl_path = r'D:\4dflow\zm\zm_frame_1.stl'
aorta_stl_path = r'D:\4dflow\zm\zm2203.stl'

# 使用固定的空间参数
pixel_spacing = 1.9531  # 像素间距 (mm)
slice_thickness = 2.409  # 层厚 (mm)

# 读取两个STL模型
mm_vertices, mm_faces = read_stl(mm_stl_path)
aorta_vertices, aorta_faces = read_stl(aorta_stl_path)

# 对主动脉模型进行坐标轴交换（Z轴与-Y轴互换）
aorta_vertices_swapped = aorta_vertices.copy()
aorta_vertices_swapped[:, 1] = -aorta_vertices[:, 2]  # Y轴取Z轴的负值
aorta_vertices_swapped[:, 2] = aorta_vertices[:, 1]   # Z轴取Y轴的值

# 对MM模型应用固定空间参数
mm_vertices_scaled = mm_vertices.copy()
mm_vertices_scaled[:, 0] *= slice_thickness  # X轴（切片方向）
mm_vertices_scaled[:, 1] *= pixel_spacing   # Y轴
mm_vertices_scaled[:, 2] *= pixel_spacing   # Z轴

# 计算两个模型的尺寸信息
mm_x_min, mm_x_max = mm_vertices_scaled[:, 0].min(), mm_vertices_scaled[:, 0].max()
mm_y_min, mm_y_max = mm_vertices_scaled[:, 1].min(), mm_vertices_scaled[:, 1].max()
mm_z_min, mm_z_max = mm_vertices_scaled[:, 2].min(), mm_vertices_scaled[:, 2].max()
mm_center = np.array([(mm_x_max + mm_x_min)/2, 
                      (mm_y_max + mm_y_min)/2, 
                      (mm_z_max + mm_z_min)/2])

# aorta_x_min, aorta_x_max = aorta_vertices[:, 0].min(), aorta_vertices[:, 0].max()
# aorta_y_min, aorta_y_max = aorta_vertices[:, 1].min(), aorta_vertices[:, 1].max()
# aorta_z_min, aorta_z_max = aorta_vertices[:, 2].min(), aorta_vertices[:, 2].max()
# aorta_center = np.array([(aorta_x_max + aorta_x_min)/2,
#                          (aorta_y_max + aorta_y_min)/2,
#                          (aorta_z_max + aorta_z_min)/2])

aorta_x_min, aorta_x_max = aorta_vertices_swapped[:, 0].min(), aorta_vertices_swapped[:, 0].max()
aorta_y_min, aorta_y_max = aorta_vertices_swapped[:, 1].min(), aorta_vertices_swapped[:, 1].max()
aorta_z_min, aorta_z_max = aorta_vertices_swapped[:, 2].min(), aorta_vertices_swapped[:, 2].max()
aorta_center = np.array([(aorta_x_max + aorta_x_min)/2,
                         (aorta_y_max + aorta_y_min)/2,
                         (aorta_z_max + aorta_z_min)/2])

print("\nMM模型尺寸信息 (应用空间参数后):")
print(f"X范围: {mm_x_min:.2f} 到 {mm_x_max:.2f} mm")
print(f"Y范围: {mm_y_min:.2f} 到 {mm_y_max:.2f} mm")
print(f"Z范围: {mm_z_min:.2f} 到 {mm_z_max:.2f} mm")
print(f"中心点: {mm_center} mm")

print("\n主动脉模型尺寸信息:")
print(f"X范围: {aorta_x_min:.2f} 到 {aorta_x_max:.2f} mm")
print(f"Y范围: {aorta_y_min:.2f} 到 {aorta_y_max:.2f} mm")
print(f"Z范围: {aorta_z_min:.2f} 到 {aorta_z_max:.2f} mm")
print(f"中心点: {aorta_center} mm")

# 计算偏移量
offset = mm_center - aorta_center

# 坐标变换函数
def transform_coordinates(x, y, z):
    # 只应用偏移，不进行缩放
    point = np.array([x, y, z])
    return point + offset

# 变换主动脉顶点（使用交换后的顶点）
transformed_aorta_vertices = np.array([transform_coordinates(x, y, z) 
                                     for x, y, z in aorta_vertices_swapped])

# 可视化
mlab.figure(bgcolor=(1,1,1))

# 显示MM模型（灰色，使用缩放后的顶点）
mlab.triangular_mesh(mm_vertices_scaled[:, 0], mm_vertices_scaled[:, 1], mm_vertices_scaled[:, 2],
                    mm_faces, color=(0.8, 0.8, 0.8), opacity=0.3)
# 显示主动脉模型（红色）
# mlab.triangular_mesh(aorta_vertices[:, 0], aorta_vertices[:, 1], aorta_vertices[:, 2],
#                     aorta_faces, color=(1, 0, 0), opacity=0.3)
# 显示原始主动脉模型（红色，使用交换后的顶点）
mlab.triangular_mesh(aorta_vertices_swapped[:, 0], aorta_vertices_swapped[:, 1], aorta_vertices_swapped[:, 2],
                    aorta_faces, color=(1, 0, 0), opacity=0.3)

# 显示变换后的主动脉模型（绿色）
mlab.triangular_mesh(transformed_aorta_vertices[:, 0], 
                    transformed_aorta_vertices[:, 1], 
                    transformed_aorta_vertices[:, 2],
                    aorta_faces, color=(0, 1, 0), opacity=0.3)

# 添加坐标标签
mlab.text3d(mm_x_max, 0, 0, 'X', scale=5)
mlab.text3d(0, mm_y_max, 0, 'Y', scale=5)
mlab.text3d(0, 0, mm_z_max, 'Z', scale=5)

mlab.show() 