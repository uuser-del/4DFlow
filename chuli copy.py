import numpy as np
import pydicom
import trimesh
from mayavi import mlab
from scipy.spatial import cKDTree
import os

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
aorta_stl_path = r'D:\4dflow\mm_3d_models\ast.stl'

# 使用固定的空间参数
pixel_spacing = 1.9531  # 像素间距 (mm)
slice_thickness = 2.409  # 层厚 (mm)

# 读取数据
FHP_raw, info_FHP = read_dicom_array(FHP_path)
RLP_raw, info_RLP = read_dicom_array(RLP_path)
APP_raw, info_APP = read_dicom_array(APP_path)

# 计算速度幅值
from toool import speed_calculate
Vmag = speed_calculate(FHP_raw, RLP_raw, APP_raw)

# 构建Vmag网格（考虑实际物理尺寸）
x = np.arange(80) * slice_thickness
y = np.arange(128) * pixel_spacing
z = np.arange(128) * pixel_spacing
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 读取STL模型
mm_vertices, mm_faces = read_stl(mm_stl_path)
aorta_vertices, aorta_faces = read_stl(aorta_stl_path)

# 第一步：使用peizhun.py的方法配准mm和vmag
# 交换STL模型的Z轴和X轴
mm_vertices_swapped = mm_vertices.copy()
mm_vertices_swapped[:, 0] = mm_vertices[:, 2]  # X轴取Z轴的值
mm_vertices_swapped[:, 2] = mm_vertices[:, 0]  # Z轴取X轴的值

# 对mm模型应用空间参数
mm_vertices_scaled = mm_vertices_swapped.copy()
mm_vertices_scaled[:, 0] *= slice_thickness  # X轴（切片方向）
mm_vertices_scaled[:, 1] *= pixel_spacing   # Y轴
mm_vertices_scaled[:, 2] *= pixel_spacing   # Z轴

# 计算Vmag模型的尺寸信息
vmag_x_min, vmag_x_max = X.min(), X.max()
vmag_y_min, vmag_y_max = Y.min(), Y.max()
vmag_z_min, vmag_z_max = Z.min(), Z.max()
vmag_center = np.array([(vmag_x_max + vmag_x_min)/2, 
                        (vmag_y_max + vmag_y_min)/2, 
                        (vmag_z_max + vmag_z_min)/2])

# 计算STL模型的尺寸信息（使用缩放后的顶点）
mm_x_min, mm_x_max = mm_vertices_scaled[:, 0].min(), mm_vertices_scaled[:, 0].max()
mm_y_min, mm_y_max = mm_vertices_scaled[:, 1].min(), mm_vertices_scaled[:, 1].max()
mm_z_min, mm_z_max = mm_vertices_scaled[:, 2].min(), mm_vertices_scaled[:, 2].max()
mm_center = np.array([(mm_x_max + mm_x_min)/2,
                      (mm_y_max + mm_y_min)/2,
                      (mm_z_max + mm_z_min)/2])

# 计算偏移量
offset = vmag_center - mm_center

# 坐标变换函数
def transform_coordinates(x, y, z):
    # 只应用偏移
    point = np.array([x, y, z])
    return point + offset

# 变换STL顶点（使用缩放后的顶点）
transformed_mm_vertices = np.array([transform_coordinates(x, y, z) 
                                  for x, y, z in mm_vertices_scaled])

# 第二步：使用stl_register.py的方法配准主动脉和已配准的mm
# 交换主动脉模型的Z轴和-Y轴
aorta_vertices_swapped = aorta_vertices.copy()
aorta_vertices_swapped[:, 1] = -aorta_vertices[:, 2]  # Y轴取Z轴的负值
aorta_vertices_swapped[:, 2] = aorta_vertices[:, 1]   # Z轴取Y轴的值

# 计算主动脉模型的尺寸信息（直接使用交换后的顶点，不进行缩放）
aorta_x_min, aorta_x_max = aorta_vertices_swapped[:, 0].min(), aorta_vertices_swapped[:, 0].max()
aorta_y_min, aorta_y_max = aorta_vertices_swapped[:, 1].min(), aorta_vertices_swapped[:, 1].max()
aorta_z_min, aorta_z_max = aorta_vertices_swapped[:, 2].min(), aorta_vertices_swapped[:, 2].max()
aorta_center = np.array([(aorta_x_max + aorta_x_min)/2,
                         (aorta_y_max + aorta_y_min)/2,
                         (aorta_z_max + aorta_z_min)/2])

# 计算主动脉到已配准mm的偏移量
aorta_offset = vmag_center - aorta_center

# 主动脉坐标变换函数
def transform_aorta_coordinates(x, y, z):
    # 只应用偏移
    point = np.array([x, y, z])
    return point + aorta_offset

# 变换主动脉顶点（直接使用交换后的顶点，不进行缩放）
transformed_aorta_vertices = np.array([transform_aorta_coordinates(x, y, z) 
                                     for x, y, z in aorta_vertices_swapped])

# 计算变换后主动脉模型的中心点
transformed_aorta_center = np.mean(transformed_aorta_vertices, axis=0)

# 对变换后的主动脉模型进行镜面对称
mirrored_aorta_vertices = transformed_aorta_vertices.copy()
mirrored_aorta_vertices[:, 0] = -transformed_aorta_vertices[:, 0]  # 对x坐标取反

# 计算镜像后模型的中心点
mirrored_aorta_center = np.mean(mirrored_aorta_vertices, axis=0)

# 计算偏移量，使镜像模型与原始模型中心点一致
mirror_offset = transformed_aorta_center - mirrored_aorta_center

# 应用偏移到镜像模型
mirrored_aorta_vertices = mirrored_aorta_vertices + mirror_offset

# 计算镜像模型在Y轴方向上的范围
mirrored_y_min = mirrored_aorta_vertices[:, 1].min()
mirrored_y_max = mirrored_aorta_vertices[:, 1].max()

# 计算需要向Y轴正方向偏移的距离
y_offset = Y.max() - mirrored_y_max

# 创建Y轴方向的偏移向量
y_offset_vector = np.array([0, y_offset, 0])

# 应用Y轴方向的偏移
mirrored_aorta_vertices = mirrored_aorta_vertices + y_offset_vector

# 创建网格点（使用物理尺寸）
grid_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

# 创建主动脉网格对象
aorta_mesh = trimesh.Trimesh(vertices=mirrored_aorta_vertices, faces=aorta_faces)

# 使用射线投射法判断点是否在主动脉内部
print("正在判断点是否在主动脉内部...")
aorta_mask = np.zeros(len(grid_points), dtype=bool)

# 创建射线方向（X轴正方向）
ray_directions = np.tile(np.array([1.0, 0.0, 0.0]), (len(grid_points), 1))

# 使用trimesh的ray模块进行射线投射
ray_origins = grid_points
ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(aorta_mesh)

# 分批处理点以避免内存问题
batch_size = 5000
for i in range(0, len(grid_points), batch_size):
    print(f"处理点 {i} 到 {min(i+batch_size, len(grid_points))}")
    batch_origins = ray_origins[i:i+batch_size]
    batch_directions = ray_directions[i:i+batch_size]
    
    # 获取射线与网格的交点
    locations, index_ray, index_tri = ray_intersector.intersects_location(
        batch_origins, batch_directions)
    
    # 计算每个射线的交点数量
    if len(index_ray) > 0:
        unique_rays, counts = np.unique(index_ray, return_counts=True)
        # 如果交点数量为奇数，则点在内部
        aorta_mask[i + unique_rays] = (counts % 2) == 1

print("判断完成！")

# 将掩码重塑为3D形状
aorta_mask = aorta_mask.reshape(X.shape)

# 创建保存数据的目录
save_dir = 'point_cloud_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 处理所有时间点的数据
num_timepoints = Vmag.shape[0]  # 获取时间点数量 (20)
print(f"开始处理 {num_timepoints} 个时间点的数据...")

for t in range(num_timepoints):
    print(f"\n处理时间点 {t+1}/{num_timepoints}")
    
    # 获取当前时间点的速度幅值
    current_vmag = Vmag[t]  # 维度为 (80, 128, 128)
    
    # 应用掩码到当前时间点的Vmag数据
    masked_vmag = current_vmag.copy()
    masked_vmag[~aorta_mask] = 0
    
    # 创建点云数据
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    masked_values = masked_vmag.flatten()
    
    # 将aorta_mask展平以匹配points的维度
    aorta_mask_flat = aorta_mask.flatten()
    masked_points = points[aorta_mask_flat]  # 只保存分割后的点云坐标
    masked_velocities = masked_values[aorta_mask_flat]  # 只保存分割后的速度值
    
    # 保存当前时间点的数据
    timepoint_dir = os.path.join(save_dir, f'timepoint_{t+1}')
    if not os.path.exists(timepoint_dir):
        os.makedirs(timepoint_dir)
    
    np.save(os.path.join(timepoint_dir, 'masked_points.npy'), masked_points)
    np.save(os.path.join(timepoint_dir, 'masked_velocities.npy'), masked_velocities)
    
    print(f"时间点 {t+1} 的数据已保存到 {timepoint_dir}")

print("\n所有时间点的数据处理完成！")

# # 设置速度阈值，只显示速度大于阈值的点
# speed_threshold = 0  # 可以根据需要调整阈值
# mask_original = original_values > speed_threshold
# mask_masked = masked_values > speed_threshold

# # 创建可视化窗口
# mlab.figure(bgcolor=(0.3,0.3,0.3), size=(1200, 800))

# # 显示原始Vmag点云（灰色）
# pts_original = mlab.points3d(points[mask_original, 0], 
#                            points[mask_original, 1], 
#                            points[mask_original, 2], 
#                            original_values[mask_original],
#                            scale_mode='none',
#                            scale_factor=1.0,
#                            colormap='gray',
#                            opacity=0.2)

# # 显示分割后的Vmag点云（红色）
# pts_masked = mlab.points3d(points[mask_masked, 0], 
#                           points[mask_masked, 1], 
#                           points[mask_masked, 2], 
#                           masked_values[mask_masked],
#                           scale_mode='none',
#                           scale_factor=1.0,
#                           colormap='hot',
#                           opacity=0.4)



# # 添加坐标轴
# mlab.quiver3d(0, 0, 0, X.max(), 0, 0, color=(1,0,0), mode='arrow', scale_factor=1)
# mlab.quiver3d(0, 0, 0, 0, Y.max(), 0, color=(0,1,0), mode='arrow', scale_factor=1)
# mlab.quiver3d(0, 0, 0, 0, 0, Z.max(), color=(0,0,1), mode='arrow', scale_factor=1)

# # 显示已配准的mm模型（蓝色）
# mlab.triangular_mesh(transformed_mm_vertices[:, 0], 
#                     transformed_mm_vertices[:, 1], 
#                     transformed_mm_vertices[:, 2],
#                     mm_faces, color=(0, 0, 1), opacity=0.3)

# # 显示变换后的主动脉模型（绿色）
# mlab.triangular_mesh(transformed_aorta_vertices[:, 0], 
#                     transformed_aorta_vertices[:, 1], 
#                     transformed_aorta_vertices[:, 2],
#                     aorta_faces, color=(0, 1, 0), opacity=0.3)

# # 显示镜像并偏移后的主动脉模型（蓝色）
# mlab.triangular_mesh(mirrored_aorta_vertices[:, 0], 
#                     mirrored_aorta_vertices[:, 1], 
#                     mirrored_aorta_vertices[:, 2],
#                     aorta_faces, color=(0, 0, 1), opacity=0.3)

# # 添加颜色条
# mlab.colorbar(pts_original, title='原始速度幅值', orientation='vertical')
# mlab.colorbar(pts_masked, title='分割后速度幅值', orientation='vertical')

# # 设置视角
# mlab.view(azimuth=45, elevation=45, distance='auto')

# mlab.show() 