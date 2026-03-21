"""
LocoMimicEnv — Motion Imitation Environment for Humanoid Locomotion

A Gymnasium environment where a humanoid robot learns to imitate a reference
walking motion through reinforcement learning. At each timestep the robot
receives a reward based on how closely its pose, velocity, and end effector
positions match the reference motion. The robot is controlled via a PD
controller that takes normalized joint position targets as actions.

Inspired by DeepMimic (Peng et al. 2018) but simplified and generalized —
not tied to a specific robot, reward weights are fixed rather than tuned
per motion, and termination is relative to the reference rather than absolute.

Reference motion is loaded via MotionClip from a pre-retargeted CSV file.
Robot model is loaded from a MuJoCo XML file.

Usage:
    env = LocoMimicEnv('data/lafan1_retargeted/g1/walk1_subject1.csv')
    obs, _ = env.reset()
    obs, reward, terminated, truncated, _ = env.step(action)
"""

import numpy as np 
import mujoco 
import gymnasium as gym 
from env.motion_clip import MotionClip
import mujoco.viewer
import time

G1_XML = "mujoco_menagerie/unitree_g1/scene.xml"

class LocoMimicEnv(gym.Env):
    def __init__(self, motion_clip_path, render_mode=None):

        """
        Sets up the simulation environment, robot model, reference motion,
        PD controller, and RL spaces. Everything the environment needs to
        run episodes is initialized here.
        """

        # load mujoco model
        self.model = mujoco.MjModel.from_xml_path(G1_XML)
        self.data = mujoco.MjData(self.model)
        self.ref_data = mujoco.MjData(self.model)  # separate data for reference FK

        # load motion reference clip
        self.motion = MotionClip(motion_clip_path)

        # sim parameters
        self.fps = 30
        self.dt = 1.0/ self.fps
        self.n_substeps = int(round(self.dt / self.model.opt.timestep))

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.default_joint_pos = self.data.qpos[7:].copy()

        # pd control params
        #self.model.nu - number of actuators
        self.kp = np.full(self.model.nu, 500.0)
        self.kd = 2.0 * np.sqrt(self.kp)

        #each component of action only shifts the joint target by +/- 0.5 radians atmostfrom the default position (per dimension)
        self.action_scale = 0.5

        # action space
        #box - continuous values
        self.action_space = gym.spaces.Box(
            low = -1.0, high = 1.0,
            shape = (self.model.nu,),
            dtype = np.float32 
        )

        # observation space
        """
        robot root height (1,)
        robot root quat   (4,)
        robot joint pos   (29,)
        robot joint vel   (29,)
        robot root vel    (3,)
        robot root angvel (3,)
        ref root height   (1,)
        ref root quat     (4,)
        ref joint pos     (29,)
        ref joint vel     (29,)
        ref root vel      (3,)
        ref root angvel   (3,)
        phase             (1,)
        ─────────────────────
        total             (139,)


        phase - current index of the reference motion clip
        """

        obs_dim = 139
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # state stuff to track across timesteps
        self.phase = 0
        self.last_action = np.zeros(self.model.nu)
        self.n_steps = 0 # counts env steps in current episode 
        self.max_steps = 1000 # corresponds to about 33 seconds at 30 fps

        # render stuff
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        #reward 
        # nbody - number of rigid bodies in the model
        # range(1, self.model.nbody) - returns a list of body indices starting from 1 to nbody-1. 0 - is the world body(scene, ground, etc. which doesnt have to imitate anything)
        self.target_bodies = list(range(1, self.model.nbody))

        #reward weights
        self.w_pos = 1.0
        self.w_ori = 1.0
        self.w_vel = 1.0
        self.w_angv = 1.0
        self.w_action = -0.1
        #joint angles outside limits penalty
        self.w_limit = -1.0
        #TODO
        #own geoms/bodies touching each other (not implemented?)
        self.w_self_contact = -0.1

    def reset(self, seed=None, options=None):
        """
        Starts a new episode by placing the robot at a random point in the
        reference motion clip. By starting at random phases, the policy gets 
        exposed to all parts of the motion during training, not just the beginning.
        (Inspired by DeepMimic)
        """
        super().reset(seed=seed)

        # initialize phase at random points from the motion clip
        #TODO
        self.phase = np.random.randint(0, len(self.motion)) # will following any other dist help for long horizon? idk

        # set the mujoco model to ref pose from the motion clip at sampled phase 
        self.data.qpos[:] = self.motion.get_qpos(self.phase) 
        self.data.qvel[:] = self.motion.get_qvel(self.phase)

        # random perturbation for robustness
        #np.random.normal(mean, std, size)
        #TODO
        self.data.qpos[7:] += np.random.normal(0, 0.01, self.model.nu)
        self.data.qvel[6:] += np.random.normal(0, 0.01, self.model.nu)

        # update all body positions
        #calculates the cartesian positions, orientation, velocities, and angular velocities of all the bodies in the model
        mujoco.mj_forward(self.model, self.data)

        # reset no. of steps and last action 
        self.n_steps = 0
        self.last_action = np.zeros(self.model.nu)


        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        """
        Advances the simulation by one policy step. Applies the action via PD
        control, steps the physics n_substeps times, then returns the standard
        Gymnasium tuple. The phase advances by one frame per step, looping
        back to the start when the clip ends.
        """
        self._apply_pd_control(action)
        
        #
        for i in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        self.n_steps += 1
        self.phase = (self.phase + 1) % len(self.motion)

        #returns 
        obs = self._get_obs()
        reward = self._compute_reward(action)
        self.last_action = action.copy()
        terminated = self._is_terminated()
        truncated = self.n_steps >= self.max_steps
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Builds the observation vector by combining the robot's current state
        with the reference state at the current phase. The policy uses this
        to see both where the robot is and where it should be, so it can
        compute the error and correct itself.
        """

        # get robot state from mujoco
        #qpos[0:2] - x,y position of the root ignored to make policy invariant to translation
        root_height = self.data.qpos[2:3]
        root_quat   = self.data.qpos[3:7]
        joint_pos   = self.data.qpos[7:]
        root_vel    = self.data.qvel[0:3]
        root_angvel = self.data.qvel[3:6]
        joint_vel   = self.data.qvel[6:]

        # get ref pose from motion clip
        ref_qpos = self.motion.get_qpos(self.phase)
        ref_qvel = self.motion.get_qvel(self.phase)

        ref_root_height = ref_qpos[2:3]
        ref_root_quat   = ref_qpos[3:7]
        ref_joint_pos   = ref_qpos[7:]
        ref_root_vel    = ref_qvel[0:3]
        ref_root_angvel = ref_qvel[3:6]
        ref_joint_vel   = ref_qvel[6:]

        phase_norm = np.array([self.phase / len(self.motion)])

        obs = np.concatenate([
            root_height,
            root_quat,
            joint_pos,
            joint_vel,
            root_vel,
            root_angvel,
            ref_root_height,
            ref_root_quat,
            ref_joint_pos,
            ref_joint_vel,
            ref_root_vel,
            ref_root_angvel,
            phase_norm
        ]).astype(np.float32)

        return obs

    def _compute_reward(self, action):

        # current body positions (from live sim, no mutation)
        current_pos = {}
        current_rot = {}
        current_vel = {}
        current_ang_vel = {}

        for body_id in self.target_bodies:
            current_pos[body_id] = self.data.xpos[body_id].copy()
            current_rot[body_id] = self.data.xmat[body_id].reshape(3, 3).copy()
            current_vel[body_id] = self.data.cvel[body_id, 3:].copy()
            current_ang_vel[body_id] = self.data.cvel[body_id, 0:3].copy()

        # reference body positions (computed on separate MjData)
        self.ref_data.qpos[:] = self.motion.get_qpos(self.phase)
        self.ref_data.qvel[:] = self.motion.get_qvel(self.phase)
        mujoco.mj_forward(self.model, self.ref_data)

        ref_pos  = {}
        ref_rot  = {}
        ref_linv = {}
        ref_angv = {}
        for body_id in self.target_bodies:
            ref_pos[body_id]  = self.ref_data.xpos[body_id].copy()
            ref_rot[body_id]  = self.ref_data.xmat[body_id].reshape(3, 3).copy()
            ref_linv[body_id] = self.ref_data.cvel[body_id, 3:].copy()
            ref_angv[body_id] = self.ref_data.cvel[body_id, :3].copy()

        #body position tracking reward
        p_b_errors = []
        for body_id in self.target_bodies:
            p_b_errors.append(np.linalg.norm(current_pos[body_id] - ref_pos[body_id])**2)
        #Gaussian Kernel/RBF - normalized version of mean squared error with std dev of 0.3
        # reward is 1 when error is 0 and decreases as error increases
        r_pos = np.exp(-np.mean(p_b_errors)/0.3**2)

        #body orientation tracking reward
        o_b_errors = []
        for body_id in self.target_bodies:
            R_curr = current_rot[body_id]
            R_ref = ref_rot[body_id]
            R_rel = R_ref @ R_curr.T
            cos_angle = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
            angle = np.arccos(cos_angle)
            o_b_errors.append(angle**2)
        r_ori = np.exp(-np.mean(o_b_errors)/0.4**2)

        #body linear velocity tracking reward
        v_b_errors = []
        for body_id in self.target_bodies:
            v_b_errors.append(np.linalg.norm(current_vel[body_id] - ref_linv[body_id])**2)
        r_vel = np.exp(-np.mean(v_b_errors)/1.0**2)

        #body angular velocity tracking reward
        av_b_errors = []
        for body_id in self.target_bodies:
            av_b_errors.append(np.linalg.norm(current_ang_vel[body_id] - ref_angv[body_id])**2)
        r_angv = np.exp(-np.mean(av_b_errors)/3.14**2)

        #action penalty
        r_action = np.linalg.norm(action - self.last_action)**2
        
        #joint pos limits penalty
        #first joint is ignored as it is a free joint of the robot/root
        l_limit = self.model.jnt_range[1:, 0]
        u_limit = self.model.jnt_range[1:, 1]
        
        theta_i = self.data.qpos[7:]

        limit_error = []
        for i in range(self.model.nu):
            limit_error.append(max(0, theta_i[i] - u_limit[i]) + max(0, l_limit[i] - theta_i[i]))
        
        r_limit = np.sum(limit_error)

        #self contact reward is not being included for now.
        r_self_contact = 0

        #combine all rewards with weights
        reward = (self.w_pos * r_pos + 
                  self.w_ori * r_ori + 
                  self.w_vel * r_vel + 
                  self.w_angv * r_angv + 
                  self.w_action * r_action + 
                  self.w_limit * r_limit + 
                  self.w_self_contact * r_self_contact)
        
        return reward
            
    

    def _is_terminated(self):
        """
        Checks if the episode should end early. Terminates if the robot's
        height or orientation deviates too far from the reference, meaning
        the robot has effectively fallen or lost balance beyond recovery.
        Early termination gives the policy a strong signal that these states
        are bad, which speeds up learning. Room for tuning here.
        """

        ref_qpos = self.motion.get_qpos(self.phase)

        # condition 1: height deviation from reference
        height_err = abs(self.data.qpos[2] - ref_qpos[2])
        if height_err > 0.25:
            return True

        # condition 2: root orientation too far from reference
        ref_quat   = ref_qpos[3:7]
        robot_quat = self.data.qpos[3:7]
        #dot product of quaternions gives the cosine of half the angle between the two quaternions
        #we take absolute value because quaternions q and -q represent the same rotation
        dot = np.abs(np.dot(robot_quat, ref_quat))
        if dot < np.cos(0.8 / 2):
            return True

        return False
  

    def _apply_pd_control(self, action):
        """
        Converts the policy's action (normalized joint position targets) into
        actual torques sent to the robot.
        """

        target_pos = self.default_joint_pos + action * self.action_scale 

        self.data.ctrl[:] =  target_pos #mujoco position actuators will handle the torques

    def close(self):
        """
        Cleans up viewer and renderer resources.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def render(self):
        """
        Renders the current simulation state.
        - 'human'     : opens an interactive viewer window
        - 'rgb_array' : returns a (H, W, 3) numpy array for video recording
        """
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # follow camera — tracks robot root position
            self.viewer.cam.lookat[0] = self.data.qpos[0]  # x
            self.viewer.cam.lookat[1] = self.data.qpos[1]  # y
            self.viewer.cam.lookat[2] = self.data.qpos[2]  # z
            self.viewer.cam.distance  = 3.0
            #Horizontal angle of the camera
            self.viewer.cam.azimuth   = 90
            #Vertical angle of the camera
            self.viewer.cam.elevation = -20
            
            self.viewer.sync()
            time.sleep(self.dt)

        elif self.render_mode == 'rgb_array':
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera='side')
            return self.renderer.render()