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

        # convert to mujoco format (w,x,y,z) - hamiltoninan convention
        self.root_quat = np.concatenate([
            root_quat_xyzw[:, 3:4],
            root_quat_xyzw[:, 0:3]
        ], axis = 1)

        self.joint_pos = raw[:, 7:36]

        # compute joint velocities using central difference theorem (CSV only has poses)
        self.joint_vel = np.zeros_like(self.joint_pos)
        # central difference for all frames except first and last
        self.joint_vel[1:-1] = (self.joint_pos[2:] - self.joint_pos[:-2]) / (2 * self.dt)
        # forward difference for first frame
        self.joint_vel[0]    = (self.joint_pos[1]  - self.joint_pos[0])   / self.dt
        # backward difference for last frame
        self.joint_vel[-1]   = (self.joint_pos[-1] - self.joint_pos[-2])  / self.dt

        self.root_vel = np.zeros_like(self.root_pos)
        self.root_vel[1:-1] = (self.root_pos[2:] - self.root_pos[:-2]) / (2 * self.dt)
        self.root_vel[0]    = (self.root_pos[1]  - self.root_pos[0])   / self.dt
        self.root_vel[-1]   = (self.root_pos[-1] - self.root_pos[-2])  / self.dt

        # compute root angular velocity from quaternion differences (body/local frame)
        # MuJoCo's qvel[3:6] for free joints is in the local body frame
        # ω_body = 2 * (q_prev⁻¹ ⊗ q_curr).xyz / dt
        self.root_angvel = np.zeros((self.n_frames, 3))
        for i in range(1, self.n_frames):
            q_prev = self.root_quat[i - 1]  # [w, x, y, z]
            q_curr = self.root_quat[i]

            # ensure shortest path (quaternion double cover: q and -q are the same rotation)
            if np.dot(q_prev, q_curr) < 0:
                q_curr = -q_curr

            # body-frame angular velocity
            q_diff = self._quat_multiply(self._quat_conjugate(q_prev), q_curr)
            self.root_angvel[i] = 2.0 * q_diff[1:4] / self.dt

        self.root_angvel[0] = self.root_angvel[1]  # copy forward for first frame

    @staticmethod
    def _quat_conjugate(q):
        """q = [w, x, y, z] → q* = [w, -x, -y, -z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_multiply(q1, q2):
        """Hamilton product: q1 ⊗ q2, both in [w, x, y, z] format"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

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
        qvel[3:6] = self.root_angvel[frame_idx]  # computed from quaternion differences
        qvel[6:35] = self.joint_vel[frame_idx]
        return qvel

    def __len__(self):
        return self.n_frames






