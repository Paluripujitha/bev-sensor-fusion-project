import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

class NuScenesDataLoader:
    def __init__(self, dataroot, version='v1.0-mini'):
        self.dataroot = dataroot
        self.version = version
        print(f"Loading NuScenes ({version}) from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        print(f"NuScenes loaded — {len(self.nusc.scene)} scenes, {len(self.nusc.sample)} samples.")

    # ------------------------------------------------------------------ scenes
    def get_scenes(self):
        scenes = []
        for scene in self.nusc.scene:
            scenes.append({
                'token': scene['token'],
                'name': scene['name'],
                'description': scene['description'],
                'nbr_samples': scene['nbr_samples'],
                'first_sample_token': scene['first_sample_token']
            })
        return scenes

    # --------------------------------------------------------------- samples
    def get_scene_samples(self, scene_token):
        """Return ordered list of sample tokens for a scene."""
        scene = self.nusc.get('scene', scene_token)
        samples = []
        token = scene['first_sample_token']
        while token:
            s = self.nusc.get('sample', token)
            samples.append({'token': s['token'], 'timestamp': s['timestamp']})
            token = s['next']
        return samples

    # ------------------------------------------------------- per-sample data
    def get_sample_info(self, sample_token):
        """Return all camera images + lidar path + metadata for a sample."""
        sample = self.nusc.get('sample', sample_token)

        cameras = {}
        for ch in CAM_CHANNELS:
            if ch not in sample['data']:
                continue
            sd = self.nusc.get('sample_data', sample['data'][ch])
            cameras[ch] = {
                'path': os.path.join(self.dataroot, sd['filename']),
                'token': sample['data'][ch],
                'filename': sd['filename']
            }

        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.dataroot, lidar_sd['filename'])

        return {
            'sample_token': sample_token,
            'timestamp': sample['timestamp'],
            'cameras': cameras,
            'lidar_token': lidar_token,
            'lidar_path': lidar_path,
            'lidar_filename': lidar_sd['filename'],
            'anns': sample['anns']
        }

    # ------------------------------------------------ LiDAR → camera frustum
    def get_lidar_projected_to_camera(self, sample_token, cam_channel='CAM_FRONT'):
        sample = self.nusc.get('sample', sample_token)

        lidar_token = sample['data']['LIDAR_TOP']
        cam_token   = sample['data'][cam_channel]

        lidar_sd = self.nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(os.path.join(self.dataroot, lidar_sd['filename']))

        # lidar → ego
        cs_lidar = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_lidar['rotation']).rotation_matrix)
        pc.translate(np.array(cs_lidar['translation']))

        # ego → global
        ep_lidar = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        pc.rotate(Quaternion(ep_lidar['rotation']).rotation_matrix)
        pc.translate(np.array(ep_lidar['translation']))

        cam_sd = self.nusc.get('sample_data', cam_token)

        # global → cam ego
        ep_cam = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        pc.translate(-np.array(ep_cam['translation']))
        pc.rotate(Quaternion(ep_cam['rotation']).rotation_matrix.T)

        # cam ego → cam sensor
        cs_cam = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        pc.translate(-np.array(cs_cam['translation']))
        pc.rotate(Quaternion(cs_cam['rotation']).rotation_matrix.T)

        depths = pc.points[2, :]
        intrinsic = np.array(cs_cam['camera_intrinsic'])
        pts2d = view_points(pc.points[:3, :], intrinsic, normalize=True)

        mask = depths > 0.1
        return pts2d[:, mask], depths[mask], pc.points[:3, mask]

    # ------------------------------------------------------- raw point cloud
    def load_lidar_points(self, sample_token):
        """Return raw (N,5) float32 array: x y z intensity ring"""
        sample  = self.nusc.get('sample', sample_token)
        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        path = os.path.join(self.dataroot, lidar_sd['filename'])
        pts  = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        return pts

    # ------------------------------------------------ ground-truth boxes
    def get_gt_boxes(self, sample_token):
        """Return list of GT annotation dicts (class, position, size, rotation)."""
        sample = self.nusc.get('sample', sample_token)
        boxes = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            boxes.append({
                'token': ann_token,
                'category': ann['category_name'],
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation'],
                'num_lidar_pts': ann['num_lidar_pts']
            })
        return boxes
