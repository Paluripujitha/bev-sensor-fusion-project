import sys
sys.path.append('d:/major_project')
from backend.model.pipeline import _get_rcnn, _detect_2d_and_lift
import numpy as np
import torch
from PIL import Image
img = np.array(Image.open(r"D:\major_project\data\samples\CAM_BACK\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg").convert('RGB'))
pts_2d = np.zeros((3, 10))
depths = np.ones(10) * 10
pts_3d = np.zeros((3, 10))
dets = _detect_2d_and_lift(img, pts_2d, depths, pts_3d, 1.0)
print(f"Detections: {len(dets)}")
if len(dets) > 0: print(dets[0])
