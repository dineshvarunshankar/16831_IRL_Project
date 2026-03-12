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
        self.kp = np.full(self.model.nu, 500.0)
        self.kd = 2.0 * np.sqrt(self.kp)

        self.action_scale = 0.5

        # action space

        self.action_space = gym.spaces.Box(
            low = -1.0, high = 1.0,
            shape = (self.model.nu,),
            dtype = np.float32 
        )

        # observation space
        """
        TODO We need to figure out the observation space: based on existing work on G1. 
        But for now we use:

        robot joint pos  (29,)
        robot joint vel  (29,)
        robot root vel   (3,)
        ref joint pos    (29,)
        ref joint vel    (29,)
        ref root vel     (3,)
        phase            (1,)
        ─────────────────────
        total            (123,)
        """

        obs_dim = 123
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

    def reset(self, seed=None, options=None):
        """
        Starts a new episode by placing the robot at a random point in the
        reference motion clip. By starting at random phases, the policy gets 
        exposed to all parts of the motion during training, not just the beginning.
        (Inspired by DeepMimic)
        """
        super().reset(seed=seed)

        # initialize phase at random points from the motion clip
        self.phase = np.random.randint(0, len(self.motion)) # will following any other dist help for long horizon? idk

        # set the mujoco model to ref pose from the motion clip at sampled phase 
        self.data.qpos[:] = self.motion.get_qpos(self.phase) 
        self.data.qvel[:] = self.motion.get_qvel(self.phase)

        # random perturbation for robustness
        self.data.qpos[7:] += np.random.normal(0, 0.01, self.model.nu)
        self.data.qvel[6:] += np.random.normal(0, 0.01, self.model.nu)

        # update all body positions
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
           
        for i in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        self.n_steps += 1
        self.phase = (self.phase + 1) % len(self.motion)
        self.last_action = action.copy()

        #returns 
        obs = self._get_obs()
        reward = self._compute_reward()
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
        joint_pos = self.data.qpos[7:]       
        joint_vel = self.data.qvel[6:]        
        root_vel  = self.data.qvel[0:3] 

        # get ref pose from motion clip
        ref_qpos = self.motion.get_qpos(self.phase)
        ref_qvel = self.motion.get_qvel(self.phase)

        # extract joint pos, vel and root vel
        ref_joint_pos = ref_qpos[7:]         
        ref_joint_vel = ref_qvel[6:]          
        ref_root_vel  = ref_qvel[0:3] 
        
        phase_norm = np.array([self.phase / len(self.motion)]) #normalized phase for obs

        obs = np.concatenate([
            joint_pos,
            joint_vel,
            root_vel,
            ref_joint_pos,
            ref_joint_vel,
            ref_root_vel,
            phase_norm
        ]).astype(np.float32)

        return obs

    def _compute_reward(self):
        return 0

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
        dot = np.abs(np.dot(robot_quat, ref_quat))
        if dot < np.cos(0.8 / 2):
            return True

        return False
  

    def _apply_pd_control(self, action):
        """
        Converts the policy's action (normalized joint position targets) into
        actual torques sent to the robot. The PD controller acts like a spring-
        damper on each joint — pulling it toward the target position while
        resisting fast movements. The policy never commands torques directly,
        it only says where it wants each joint to be.
        """

        target_pos = self.default_joint_pos + action * self.action_scale 

        pos_error = target_pos - self.data.qpos[7:]
        vel_error = -self.data.qvel[6:] # no ref error since we are not tracking vels  

        torques   = self.kp * pos_error + self.kd * vel_error

        ctrl_range = self.model.actuator_ctrlrange
        torques    = np.clip(torques, ctrl_range[:, 0], ctrl_range[:, 1])

        self.data.ctrl[:] =  torques

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
            self.viewer.cam.azimuth   = 90
            self.viewer.cam.elevation = -20
            
            self.viewer.sync()
            time.sleep(self.dt)

        elif self.render_mode == 'rgb_array':
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera='side')
            return self.renderer.render()