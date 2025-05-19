import pydicom

# 读取 DICOM 文件
ds = pydicom.dcmread("S0/S11070/I11")

# 打印基本信息
# print(ds)
# 获取像素数据
pixel_array = ds.pixel_array
print("像素数据形状:", pixel_array.shape)
# print("像素数据形状:", pixel_array)

import matplotlib.pyplot as plt 

for i in range(20):
    group = pixel_array[i:i+80, :, :]
    plt.imshow(group, cmap=plt.cm.gray)
    plt.show()
