import numpy as np
import trimesh
import os

def read_stl(stl_path):
    """读取STL文件"""
    mesh = trimesh.load(stl_path)
    return mesh.vertices, mesh.faces

# 路径
aorta_stl_path = r'D:\4dflow\mm_3d_models\fangzhen.stl'

# 使用固定的空间参数
pixel_spacing = 1.9531  # 像素间距 (mm)
slice_thickness = 2.409  # 层厚 (mm)

# 构建网格（考虑实际物理尺寸）
x = np.arange(80) * slice_thickness
y = np.arange(128) * pixel_spacing
z = np.arange(128) * pixel_spacing
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 读取主动脉STL模型
aorta_vertices, aorta_faces = read_stl(aorta_stl_path)

# 交换主动脉模型的Z轴和-Y轴
aorta_vertices_swapped = aorta_vertices.copy()
aorta_vertices_swapped[:, 1] = -aorta_vertices[:, 2]  # Y轴取Z轴的负值
aorta_vertices_swapped[:, 2] = aorta_vertices[:, 1]   # Z轴取Y轴的值

# 计算主动脉模型的尺寸信息
aorta_x_min, aorta_x_max = aorta_vertices_swapped[:, 0].min(), aorta_vertices_swapped[:, 0].max()
aorta_y_min, aorta_y_max = aorta_vertices_swapped[:, 1].min(), aorta_vertices_swapped[:, 1].max()
aorta_z_min, aorta_z_max = aorta_vertices_swapped[:, 2].min(), aorta_vertices_swapped[:, 2].max()
aorta_center = np.array([(aorta_x_max + aorta_x_min)/2,
                         (aorta_y_max + aorta_y_min)/2,
                         (aorta_z_max + aorta_z_min)/2])

# 计算Vmag模型的尺寸信息
vmag_x_min, vmag_x_max = X.min(), X.max()
vmag_y_min, vmag_y_max = Y.min(), Y.max()
vmag_z_min, vmag_z_max = Z.min(), Z.max()
vmag_center = np.array([(vmag_x_max + vmag_x_min)/2, 
                        (vmag_y_max + vmag_y_min)/2, 
                        (vmag_z_max + vmag_z_min)/2])

# 计算主动脉到Vmag的偏移量
aorta_offset = vmag_center - aorta_center

# 主动脉坐标变换函数
def transform_aorta_coordinates(x, y, z):
    # 只应用偏移
    point = np.array([x, y, z])
    return point + aorta_offset

# 变换主动脉顶点
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
print("正在计算主动脉掩码...")
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

print("掩码计算完成！")

# 将掩码重塑为3D形状
aorta_mask = aorta_mask.reshape(X.shape)

# 创建保存目录
save_dir = 'mask_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存掩码
np.save(os.path.join(save_dir, 'aorta_mask.npy'), aorta_mask)
print(f"掩码已保存到 {save_dir}/aorta_mask.npy") 