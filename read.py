import pydicom
import numpy as np

ds = pydicom.dcmread('S0/S11040/I11')
print(ds)
