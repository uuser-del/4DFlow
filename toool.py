import numpy as np
import pydicom
# import pyvista as pv
# import matplotlib.pyplot as plt
# from mayavi import mlab

# 读取 DICOM 文件
def read_dicom_array(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    return arr, ds

# 路径
FHP_path = 'S0/S11040/I11'
RLP_path = 'S0/S11070/I11'
APP_path = 'S0/S11050/I11'

def speed_calculate(FHP_raw: np.ndarray, RLP_raw: np.ndarray, APP_raw: np.ndarray):
    num_timepoints = 20
    num_slices = int(FHP_raw.shape[0] / num_timepoints) 
    num_height = FHP_raw.shape[1]
    shape = (num_slices, num_timepoints, num_height, num_height)  # 使用num_height替代硬编码的128

    # 重塑数组维度为 (时间点, 切片, 高度, 宽度)
    FHP = FHP_raw.reshape(shape)
    RLP = RLP_raw.reshape(shape)
    APP = APP_raw.reshape(shape)
    
    print("原始数据范围:")
    print(f"FHP: min={FHP.min()}, max={FHP.max()}")
    print(f"RLP: min={RLP.min()}, max={RLP.max()}")
    print(f"APP: min={APP.min()}, max={APP.max()}")
    
    # 归一化参数
    P0 = 2048  # DICOM 中心值
    SCALE = 2048
    VENC = 200
    
    # 对重塑后的数组进行归一化
    FHP = (FHP - P0) / SCALE * VENC
    RLP = (RLP - P0) / SCALE * VENC
    APP = (APP - P0) / SCALE * VENC
    
    print("\n归一化后范围:")
    print(f"FHP: min={FHP.min()}, max={FHP.max()}")
    print(f"RLP: min={RLP.min()}, max={RLP.max()}")
    print(f"APP: min={APP.min()}, max={APP.max()}")
    
    # 计算速度幅值，保持维度不变 (20, 80, 128, 128)
    Vmag = np.sqrt(FHP**2 + RLP**2 + APP**2)
    # / np.sqrt(3)
    
    print("\n速度幅值范围:")
    print(f"Vmag: min={Vmag.min()}, max={Vmag.max()}")
    
    # 验证速度幅值在合理范围内
    assert Vmag.min() >= 0, "速度幅值不应为负"
    # assert Vmag.max() <= VENC, "速度幅值不应超过最大可能值"
    
    return Vmag

if __name__ == '__main__':
    FHP_raw, info_FHP = read_dicom_array(FHP_path)
    RLP_raw, info_RLP = read_dicom_array(RLP_path)
    APP_raw, info_APP = read_dicom_array(APP_path)
    Vmag = speed_calculate(FHP_raw, RLP_raw, APP_raw)
    print(Vmag.shape)
    print('速度幅值 min:', Vmag.min(), 'max:', Vmag.max())


