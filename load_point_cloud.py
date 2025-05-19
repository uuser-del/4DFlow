import numpy as np
from mayavi import mlab
import os
from scipy.spatial import ConvexHull

def load_point_cloud_data():
    """加载分割后的点云数据"""
    save_dir = 'point_cloud_data'
    
    # 加载分割后的点云数据
    masked_points = np.load(os.path.join(save_dir, 'masked_points.npy'))
    masked_velocities = np.load(os.path.join(save_dir, 'masked_velocities.npy'))
    
    return masked_points, masked_velocities

# def calculate_slice_velocity(points, velocities, y_coord, tolerance=0.1):
#     """计算指定Y坐标处的点的平均速度"""
#     # 找到Y坐标在指定范围内的点
#     y_mask = np.abs(points[:, 1] - y_coord) < tolerance
#     slice_points = points[y_mask]
#     slice_velocities = velocities[y_mask]
    
#     if len(slice_points) == 0:
#         print(f"在Y={y_coord}处没有找到点")
#         return
    
#     # 计算平均速度
#     avg_velocity = np.mean(slice_velocities)
    
#     # 打印每个点的信息
#     print(f"\nY={y_coord}处的点信息：")
#     print(f"找到{len(slice_points)}个点")
#     print("\n点坐标及速度值：")
#     print("X\t\tY\t\tZ\t\t速度(cm/s)")
#     print("-" * 50)
#     for point, vel in zip(slice_points, slice_velocities):
#         print(f"{point[0]:.1f}\t\t{point[1]:.1f}\t\t{point[2]:.1f}\t\t{vel:.1f}")
    
#     print(f"\n平均速度: {avg_velocity:.1f} cm/s")
#     return avg_velocity

# def get_outer_points(points, num_points=10):
#     """获取点云外层的特征点"""
#     # 使用凸包算法获取外层点
#     hull = ConvexHull(points)
#     outer_indices = np.unique(hull.vertices)
    
#     # 如果外层点太多，随机选择一部分
#     if len(outer_indices) > num_points:
#         outer_indices = np.random.choice(outer_indices, num_points, replace=False)
    
#     return points[outer_indices]

selected_indices = []
pts = None  # 声明全局变量
pts_outside = None  # 声明全局变量

def picker_callback(picker_obj):
    global pts
    if picker_obj.actor in pts.actor.actors:
        x, y, z = picker_obj.pick_position
        # 找到最近的点索引
        distances = np.linalg.norm(masked_points - np.array([x, y, z]), axis=1)
        idx = np.argmin(distances)
        selected_indices.append(idx)
        print(f"选中点坐标: {masked_points[idx]}, 速度: {masked_velocities[idx]:.2f} cm/s")
        if len(selected_indices) > 1:
            avg_speed = np.mean(masked_velocities[selected_indices])
            print(f"当前选中点的平均速度: {avg_speed:.2f} cm/s")

def visualize_point_cloud(masked_points, masked_velocities, speed_threshold=0, y_slice=None):
    """可视化分割后的点云数据
    Args:
        masked_points: 点云坐标
        masked_velocities: 点云速度
        speed_threshold: 速度阈值
        y_slice: Y坐标切片位置，如果指定则只显示该位置附近的点
    """
    global pts, pts_outside
    
    def update_slice(y_pos):
        """更新切片位置并重新显示点云"""
        global pts, pts_outside
        # 清除之前的点
        if pts is not None:
            pts.remove()
        if pts_outside is not None:
            pts_outside.remove()
            
        # 创建新的切片掩码
        slice_mask = np.abs(masked_points[:, 1] - y_pos) < 1.0
        inside_mask = mask & slice_mask
        outside_mask = mask & ~slice_mask
        
        # 显示切片内的点
        pts = mlab.points3d(masked_points[inside_mask, 0], 
                           masked_points[inside_mask, 1], 
                           masked_points[inside_mask, 2], 
                           masked_velocities[inside_mask],
                           scale_mode='none',
                           scale_factor=1.0,
                           colormap='viridis',
                           opacity=0.8)
        
        # 显示切片外的点（更透明）
        pts_outside = mlab.points3d(masked_points[outside_mask, 0], 
                                  masked_points[outside_mask, 1], 
                                  masked_points[outside_mask, 2], 
                                  masked_velocities[outside_mask],
                                  scale_mode='none',
                                  scale_factor=1.0,
                                  colormap='viridis',
                                  opacity=0.1)
        
        # 打印切片信息
        print(f"\n切片位置: Y = {y_pos:.2f}")
        print(f"切片内点数: {np.sum(inside_mask)}")
        print(f"切片内平均速度: {np.mean(masked_velocities[inside_mask]):.2f} cm/s")
        print("\n切片内点的详细信息：")
        print("X\t\tY\t\tZ\t\t速度(cm/s)")
        print("-" * 50)
        for i in np.where(inside_mask)[0]:
            print(f"{masked_points[i,0]:.1f}\t\t{masked_points[i,1]:.1f}\t\t{masked_points[i,2]:.1f}\t\t{masked_velocities[i]:.1f}")
    
    # 创建掩码
    mask = masked_velocities > speed_threshold
    
    # 创建可视化窗口
    mlab.figure(bgcolor=(0.3,0.3,0.3), size=(1200, 800))
    
    # 初始化切片
    if y_slice is None:
        y_slice = masked_points[:, 1].mean()  # 默认使用Y坐标的平均值
    update_slice(y_slice)
    
    # 添加坐标轴
    mlab.quiver3d(0, 0, 0, masked_points[:, 0].max(), 0, 0, color=(1,0,0), mode='arrow', scale_factor=1)
    mlab.quiver3d(0, 0, 0, 0, masked_points[:, 1].max(), 0, color=(0,1,0), mode='arrow', scale_factor=1)
    mlab.quiver3d(0, 0, 0, 0, 0, masked_points[:, 2].max(), color=(0,0,1), mode='arrow', scale_factor=1)
    
    # 添加颜色条
    mlab.colorbar(pts, title='速度幅值 (cm/s)', orientation='vertical')
    
    # 设置视角
    mlab.view(azimuth=45, elevation=45, distance='auto')
    
    # 添加键盘事件处理
    def keyboard_callback(vtk_obj, event):
        nonlocal y_slice
        if event == 'KeyPressEvent':
            key = vtk_obj.GetKeySym()
            if key == 'Up':
                y_slice += 1.0
                update_slice(y_slice)
            elif key == 'Down':
                y_slice -= 1.0
                update_slice(y_slice)
            elif key == 'Left':
                y_slice -= 0.1
                update_slice(y_slice)
            elif key == 'Right':
                y_slice += 0.1
                update_slice(y_slice)
    
    # 注册键盘事件
    mlab.gcf().scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
    
    # 修改picker_callback，只选择切片内的点
    def new_picker_callback(picker_obj):
        if picker_obj.actor in pts.actor.actors:  # 只处理切片内的点
            x, y, z = picker_obj.pick_position
            # 找到最近的点索引
            distances = np.linalg.norm(masked_points - np.array([x, y, z]), axis=1)
            idx = np.argmin(distances)
            selected_indices.append(idx)
            print(f"选中点坐标: {masked_points[idx]}, 速度: {masked_velocities[idx]:.2f} cm/s")
            if len(selected_indices) > 1:
                avg_speed = np.mean(masked_velocities[selected_indices])
                print(f"当前选中点的平均速度: {avg_speed:.2f} cm/s")
    
    mlab.gcf().on_mouse_pick(new_picker_callback)
    
    # 添加说明文本
    mlab.text(0.01, 0.95, '使用方向键调整切片位置:\n上下键: ±1.0\n左右键: ±0.1', width=0.3)
    
    mlab.show()

if __name__ == '__main__':
    # 加载数据
    masked_points, masked_velocities = load_point_cloud_data()
    
    # # 计算Y=246.1处的平均速度
    # calculate_slice_velocity(masked_points, masked_velocities, 246.1)
    
    # 可视化点云，可以指定Y坐标来创建切片
    # 例如：y_slice=246.1 会显示Y=246.1附近的点
    visualize_point_cloud(masked_points, masked_velocities, y_slice=146.1) 