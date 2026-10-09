import sys
sys.path.append('d:/major_project')
from backend.utils.weather_sim import apply_weather
rgb, cq, cw, lw = apply_weather(r"D:\major_project\data\samples\CAM_BACK\n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg", 'clear', 0.0)
print(f"cam_quality={cq:.3f}, cam_weight={cw:.3f}, lidar_weight={lw:.3f}")
