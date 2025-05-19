import numpy as np
from mayavi import mlab
import os
from scipy.spatial import ConvexHull

def load_point_cloud_data(timepoint=None):
    """加载分割后的点云数据
    Args:
        timepoint: 时间点编号（1-20），如果为None则加载所有时间点
    """
    save_dir = 'point_cloud_data'
    
    if timepoint is None:
        # 加载所有时间点的数据
        all_points = []
        all_velocities = []
        for t in range(1, 21):
            timepoint_dir = os.path.join(save_dir, f'timepoint_{t}')
            points = np.load(os.path.join(timepoint_dir, 'masked_points.npy'))
            velocities = np.load(os.path.join(timepoint_dir, 'masked_velocities.npy'))
            all_points.append(points)
            all_velocities.append(velocities)
        return all_points, all_velocities
    else:
        # 加载指定时间点的数据
        timepoint_dir = os.path.join(save_dir, f'timepoint_{timepoint}')
        masked_points = np.load(os.path.join(timepoint_dir, 'masked_points.npy'))
        masked_velocities = np.load(os.path.join(timepoint_dir, 'masked_velocities.npy'))
        return masked_points, masked_velocities

def calculate_slice_velocity(points, velocities, y_coord, tolerance=0.1):
    """计算指定Y坐标处的点的平均速度"""
    # 找到Y坐标在指定范围内的点
    y_mask = np.abs(points[:, 1] - y_coord) < tolerance
    slice_points = points[y_mask]
    slice_velocities = velocities[y_mask]
    
    if len(slice_points) == 0:
        print(f"在Y={y_coord}处没有找到点")
        return
    
    # 计算平均速度
    avg_velocity = np.mean(slice_velocities)
    
    # 打印每个点的信息
    print(f"\nY={y_coord}处的点信息：")
    print(f"找到{len(slice_points)}个点")
    print("\n点坐标及速度值：")
    print("X\t\tY\t\tZ\t\t速度(cm/s)")
    print("-" * 50)
    for point, vel in zip(slice_points, slice_velocities):
        print(f"{point[0]:.1f}\t\t{point[1]:.1f}\t\t{point[2]:.1f}\t\t{vel:.1f}")
    
    print(f"\n平均速度: {avg_velocity:.1f} cm/s")
    return avg_velocity

def get_outer_points(points, num_points=30):
    """获取点云外层的特征点"""
    # 使用凸包算法获取外层点
    hull = ConvexHull(points)
    outer_indices = np.unique(hull.vertices)
    
    # 如果外层点太多，随机选择一部分
    if len(outer_indices) > num_points:
        outer_indices = np.random.choice(outer_indices, num_points, replace=False)
    
    return points[outer_indices]

def visualize_point_cloud(masked_points, masked_velocities, speed_threshold=0, timepoint=None):
    """可视化分割后的点云数据"""
    # 创建掩码
    mask = masked_velocities > speed_threshold
    
    # 创建可视化窗口
    mlab.figure(bgcolor=(0.3,0.3,0.3), size=(1200, 800))
    
    # 显示分割后的点云
    pts = mlab.points3d(masked_points[mask, 0], 
                       masked_points[mask, 1], 
                       masked_points[mask, 2], 
                       masked_velocities[mask],
                       scale_mode='none',
                       scale_factor=1.0,
                       colormap='hot',
                       opacity=0.4)
    
    # 获取并显示外层点
    outer_points = get_outer_points(masked_points[mask])
    for point in outer_points:
        x, y, z = point
        mlab.text3d(x, y, z, f'({x:.1f},{y:.1f},{z:.1f})', 
                   scale=2, color=(1,1,1))
    
    # 添加坐标轴
    mlab.quiver3d(0, 0, 0, masked_points[:, 0].max(), 0, 0, color=(1,0,0), mode='arrow', scale_factor=1)
    mlab.quiver3d(0, 0, 0, 0, masked_points[:, 1].max(), 0, color=(0,1,0), mode='arrow', scale_factor=1)
    mlab.quiver3d(0, 0, 0, 0, 0, masked_points[:, 2].max(), color=(0,0,1), mode='arrow', scale_factor=1)
    
    # 添加颜色条
    mlab.colorbar(pts, title='速度幅值', orientation='vertical')
    
    # 设置视角
    mlab.view(azimuth=45, elevation=45, distance='auto')
    
    # 添加时间点信息
    if timepoint is not None:
        mlab.title(f'时间点 {timepoint}', size=0.5)
    
    mlab.show()

def visualize_all_timepoints():
    """可视化所有时间点的点云数据"""
    # 加载所有时间点的数据
    all_points, all_velocities = load_point_cloud_data()
    
    # 为每个时间点创建可视化
    for t in range(20):
        print(f"\n正在显示时间点 {t+1}/20")
        visualize_point_cloud(all_points[t], all_velocities[t], timepoint=t+1)

if __name__ == '__main__':
    # 可视化所有时间点的数据
    visualize_all_timepoints() 