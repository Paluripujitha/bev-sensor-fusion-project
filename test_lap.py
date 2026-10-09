import sys
sys.path.append('d:/major_project')
import cv2
import numpy as np
bgr = cv2.imread(r"D:\major_project\data\samples\CAM_BACK\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg")
grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
lap  = cv2.Laplacian(grey, cv2.CV_64F).var()
print(f"Clear image lap: {lap}")
