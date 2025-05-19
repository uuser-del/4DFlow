import pydicom
import numpy as np

def get_specific_dicom_info(dicom_path):
    """
    读取DICOM文件中的特定标签信息
    
    参数:
        dicom_path: DICOM文件路径
    
    返回:
        dict: 包含特定标签信息的字典
    """
    try:
        ds = pydicom.dcmread(dicom_path)
        
        # 获取帧数 (标签2001,1017)
        frames = None
        if hasattr(ds, '0x20011017'):
            frames = ds['0x20011017'].value
        elif hasattr(ds, 'NumberOfFrames'):
            frames = ds.NumberOfFrames
            
        # 获取FPS (Nominal Cardiac Trigger Delay Time)
        fps = None
        if hasattr(ds, 'NominalCardiacTriggerDelayTime'):
            fps = ds.NominalCardiacTriggerDelayTime
            
        # 获取像素大小和切片厚度
        pixel_spacing = getattr(ds, 'PixelSpacing', 'N/A')
        slice_thickness = getattr(ds, 'SliceThickness', 'N/A')
        
        # 获取图像信息
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
            image_info = {
                'PixelType': str(pixel_array.dtype),
                'PixelMin': float(np.min(pixel_array)),
                'PixelMax': float(np.max(pixel_array)),
                'ImageShape': pixel_array.shape
            }
        else:
            image_info = {}
        
        info = {
            'Frames': frames,
            'FPS': fps,
            'PixelSpacing': pixel_spacing,
            'SliceThickness': slice_thickness,
            **image_info
        }
        
        return info
        
    except Exception as e:
        print(f"读取DICOM文件时出错: {str(e)}")
        return None

def print_specific_info(info):
    """
    打印特定DICOM信息
    
    参数:
        info: 包含标签信息的字典
    """
    if info is None:
        print("无法获取DICOM信息")
        return
        
    print("\n=== DICOM文件特定信息 ===")
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    # 使用指定的DICOM文件路径
    dicom_path = r"D:\4dflow\dh\zhaodihua_MR\S28040\I11"
    info = get_specific_dicom_info(dicom_path)
    print_specific_info(info) 