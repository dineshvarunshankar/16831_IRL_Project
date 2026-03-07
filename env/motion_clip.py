"""
MotionClip — Reference Motion Loader for G1

Loads a pre-retargeted G1 motion CSV and provides
per-frame qpos and qvel vectors compatible with MuJoCo's G1 model.

CSV format (36 columns, 30 FPS, no header):
    cols 0:3  — root xyz position
    cols 3:7  — root quaternion (qx, qy, qz, qw)  [reordered to MuJoCo convention]
    cols 7:36 — 29 joint angles matching G1 actuator order

Usage:
    mc = MotionClip('data/lafan1_retargeted/g1/walk1_subject1.csv')
    qpos = mc.get_qpos(frame_idx)   # (36,) for data.qpos
    qvel = mc.get_qvel(frame_idx)   # (35,) for data.qvel
"""

import numpy as np

class MotionClip:
    def __init__(self, csv_path, fps=30):
        raw = np.loadtxt(csv_path, delimiter = ",")
        self.fps = fps
        self.dt = 1.0 / fps
        self.n_frames = raw.shape[0]

        self.root_pos = raw[:, 0:3]
        root_quat_xyzw = raw[:, 3:7]

        # convert to mujoco format (w,x,y,z)
        self.root_quat = np.concatenate([
            root_quat_xyzw[:, 3:4],
            root_quat_xyzw[:, 0:3]
        ], axis = 1)

        self.joint_pos = raw[:, 7:36]

        self.joint_vel = np.zeros_like(self.joint_pos)
        self.joint_vel[1:-1] = (self.joint_pos[2:] - self.joint_pos[:-2]) / (2 * self.dt)
        self.joint_vel[0]    = (self.joint_pos[1]  - self.joint_pos[0])   / self.dt
        self.joint_vel[-1]   = (self.joint_pos[-1] - self.joint_pos[-2])  / self.dt

        self.root_vel = np.zeros_like(self.root_pos)
        self.root_vel[1:-1] = (self.root_pos[2:] - self.root_pos[:-2]) / (2 * self.dt)
        self.root_vel[0]    = (self.root_pos[1]  - self.root_pos[0])   / self.dt
        self.root_vel[-1]   = (self.root_pos[-1] - self.root_pos[-2])  / self.dt

    def get_qpos(self, frame_idx):
        frame_idx = frame_idx % self.n_frames
        qpos = np.zeros(36) # 3 (root pos) + 4 (quaternion) + 29 (joints)
        qpos[0:3] = self.root_pos[frame_idx]
        qpos[3:7] = self.root_quat[frame_idx]
        qpos[7:36] = self.joint_pos[frame_idx]
        return qpos

    def get_qvel(self, frame_idx):
        frame_idx = frame_idx % self.n_frames
        qvel = np.zeros(35) # 3 (root lin) + 3 (root ang) + 29 (joints)
        qvel[0:3] = self.root_vel[frame_idx]
        qvel[3:6] = 0.0 # angular velocity set to 0 
        qvel[6:35] = self.joint_vel[frame_idx]
        return qvel

    def __len__(self):
        return self.n_frames






