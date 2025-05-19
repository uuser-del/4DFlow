import numpy as np
import pydicom
import trimesh
from mayavi import mlab

# 读取 DICOM 文件
def read_dicom_array(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    return arr, ds

# 读取STL文件
def read_stl(stl_path):
    mesh = trimesh.load(stl_path)
    return mesh.vertices, mesh.faces

# 路径
FHP_path = 'S0/S11040/I11'
RLP_path = 'S0/S11070/I11'
APP_path = 'S0/S11050/I11'
mm_stl_path = r'D:\4dflow\mm_3d_models\mm_frame_1.stl'

# 读取数据
FHP_raw, info_FHP = read_dicom_array(FHP_path)
RLP_raw, info_RLP = read_dicom_array(RLP_path)
APP_raw, info_APP = read_dicom_array(APP_path)

# 计算速度幅值
from toool import speed_calculate
Vmag = speed_calculate(FHP_raw, RLP_raw, APP_raw)
Vmag = Vmag[:, 4, :, :]  # 选择第4个时间点

# 构建Vmag网格（使用原始网格坐标）
x = np.arange(80)
y = np.arange(128)
z = np.arange(128)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 读取STL模型
mm_vertices, mm_faces = read_stl(mm_stl_path)

# 交换STL模型的Z轴和X轴
mm_vertices_swapped = mm_vertices.copy()
mm_vertices_swapped[:, 0] = mm_vertices[:, 2]  # X轴取Z轴的值
mm_vertices_swapped[:, 2] = mm_vertices[:, 0]  # Z轴取X轴的值

# 计算Vmag模型的尺寸信息
vmag_x_min, vmag_x_max = X.min(), X.max()
vmag_y_min, vmag_y_max = Y.min(), Y.max()
vmag_z_min, vmag_z_max = Z.min(), Z.max()
vmag_center = np.array([(vmag_x_max + vmag_x_min)/2, 
                        (vmag_y_max + vmag_y_min)/2, 
                        (vmag_z_max + vmag_z_min)/2])

# 计算STL模型的尺寸信息（使用交换后的顶点）
mm_x_min, mm_x_max = mm_vertices_swapped[:, 0].min(), mm_vertices_swapped[:, 0].max()
mm_y_min, mm_y_max = mm_vertices_swapped[:, 1].min(), mm_vertices_swapped[:, 1].max()
mm_z_min, mm_z_max = mm_vertices_swapped[:, 2].min(), mm_vertices_swapped[:, 2].max()
mm_center = np.array([(mm_x_max + mm_x_min)/2,
                      (mm_y_max + mm_y_min)/2,
                      (mm_z_max + mm_z_min)/2])

print("\nVmag模型尺寸信息:")
print(f"X范围: {vmag_x_min:.2f} 到 {vmag_x_max:.2f}")
print(f"Y范围: {vmag_y_min:.2f} 到 {vmag_y_max:.2f}")
print(f"Z范围: {vmag_z_min:.2f} 到 {vmag_z_max:.2f}")
print(f"中心点: {vmag_center}")

print("\nSTL模型尺寸信息（交换Z轴和X轴后）:")
print(f"X范围: {mm_x_min:.2f} 到 {mm_x_max:.2f}")
print(f"Y范围: {mm_y_min:.2f} 到 {mm_y_max:.2f}")
print(f"Z范围: {mm_z_min:.2f} 到 {mm_z_max:.2f}")
print(f"中心点: {mm_center}")

# 计算偏移量
offset = vmag_center - mm_center

# 坐标变换函数
def transform_coordinates(x, y, z):
    # 只应用偏移
    point = np.array([x, y, z])
    return point + offset

# 变换STL顶点（使用交换后的顶点）
transformed_mm_vertices = np.array([transform_coordinates(x, y, z) 
                                  for x, y, z in mm_vertices_swapped])

# 可视化
mlab.figure(bgcolor=(1,1,1))

# 显示Vmag等值面（灰色）
mlab.contour3d(X, Y, Z, Vmag, contours=8, opacity=0.3, color=(0.7, 0.7, 0.7))

# 显示原始STL模型（红色，使用交换后的顶点）
mlab.triangular_mesh(mm_vertices_swapped[:, 0], mm_vertices_swapped[:, 1], mm_vertices_swapped[:, 2],
                    mm_faces, color=(1, 0, 0), opacity=0.3)

# 显示变换后的STL模型（绿色）
mlab.triangular_mesh(transformed_mm_vertices[:, 0], 
                    transformed_mm_vertices[:, 1], 
                    transformed_mm_vertices[:, 2],
                    mm_faces, color=(0, 1, 0), opacity=0.3)

# 添加坐标标签
mlab.text3d(X.max(), 0, 0, 'X', scale=5)
mlab.text3d(0, Y.max(), 0, 'Y', scale=5)
mlab.text3d(0, 0, Z.max(), 'Z', scale=5)

mlab.colorbar(title='速度幅值', orientation='vertical')
mlab.show()
