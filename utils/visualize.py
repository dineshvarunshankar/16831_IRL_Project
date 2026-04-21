"""
Visualize G1 walking motion from LAFAN1 retargeted CSV data.

Usage:
    python visualize.py
    python visualize.py --csv data/lafan1_retargeted/g1/walk1_subject1.csv
    python visualize.py --csv data/lafan1_retargeted/g1/walk1_subject1.csv --speed 0.5
"""

import argparse
import os
import numpy as np
import mujoco
import mujoco.viewer
import time
import glob

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
G1_XML = "unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml"
DATA_DIR = "./data/lafan1_retargeted"
FPS = 30

# G1 joint order from dataset documentation
# root_joint(x,y,z,qx,qy,qz,qw) + 29 joints
G1_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# ─────────────────────────────────────────────────────────────
# Find CSV file
# ─────────────────────────────────────────────────────────────
def find_csv(data_dir):
    """Search for walking CSV files in the dataset."""
    patterns = [
        os.path.join(data_dir, "**", "*walk1*subject1*.csv"),
        os.path.join(data_dir, "**", "*walk1*.csv"),
        os.path.join(data_dir, "**", "*walk*.csv"),
        os.path.join(data_dir, "**", "*.csv"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        # Filter for g1 files if possible
        g1_files = [f for f in files if "g1" in f.lower()]
        if g1_files:
            return sorted(g1_files)[0]
        if files:
            return sorted(files)[0]
    return None

# ─────────────────────────────────────────────────────────────
# Load CSV motion
# ─────────────────────────────────────────────────────────────
def load_csv_motion(csv_path):
    """
    Load G1 motion from CSV.
    
    CSV columns per row:
        0:2   = root xyz
        3:6   = root quaternion (qx, qy, qz, qw)
        7:35  = 29 joint angles
    
    Returns dict with:
        root_pos:   (N, 3)
        root_quat:  (N, 4)  in MuJoCo convention (w, x, y, z)
        joint_pos:  (N, 29)
        n_frames:   int
        fps:        int
    """
    print(f"Loading: {csv_path}")
    
    # Try loading with header, then without
    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except Exception:
        data = np.loadtxt(csv_path, delimiter=",")
    
    print(f"  Raw shape: {data.shape}")
    
    # Expect (N, 36) = 7 root + 29 joints
    if data.shape[1] < 36:
        raise ValueError(f"Expected 36 columns, got {data.shape[1]}")
    
    root_pos  = data[:, 0:3]          # xyz
    root_quat_xyzw = data[:, 3:7]    # qx qy qz qw (dataset convention)
    joint_pos = data[:, 7:36]         # 29 joints
    
    # Convert quaternion from (qx,qy,qz,qw) to MuJoCo (qw,qx,qy,qz)
    root_quat_wxyz = np.concatenate([
        root_quat_xyzw[:, 3:4],   # w
        root_quat_xyzw[:, 0:3]    # x y z
    ], axis=1)
    
    n_frames = data.shape[0]
    print(f"  Frames: {n_frames} ({n_frames/FPS:.1f} seconds)")
    print(f"  Root height range: {root_pos[:,2].min():.3f} - {root_pos[:,2].max():.3f}")
    print(f"  Joint pos range:   {joint_pos.min():.3f} - {joint_pos.max():.3f}")
    
    return {
        "root_pos":  root_pos,
        "root_quat": root_quat_wxyz,
        "joint_pos": joint_pos,
        "n_frames":  n_frames,
        "fps":       FPS,
    }

# ─────────────────────────────────────────────────────────────
# Build qpos vector for MuJoCo
# ─────────────────────────────────────────────────────────────
def motion_to_qpos(motion, frame_idx):
    """
    Convert motion frame to MuJoCo qpos vector (36,).
    
    MuJoCo qpos layout for G1:
        [0:3]  = root xyz
        [3:7]  = root quaternion (w, x, y, z)
        [7:36] = 29 joint angles
    """
    frame_idx = frame_idx % motion["n_frames"]
    
    qpos = np.zeros(36)
    qpos[0:3] = motion["root_pos"][frame_idx]
    qpos[3:7] = motion["root_quat"][frame_idx]
    qpos[7:36] = motion["joint_pos"][frame_idx]
    
    return qpos

# ─────────────────────────────────────────────────────────────
# Visualize
# ─────────────────────────────────────────────────────────────
def visualize(csv_path, speed=1.0, loop=True):
    """
    Playback motion in MuJoCo viewer.
    
    Args:
        csv_path: path to CSV file
        speed:    playback speed (1.0 = realtime, 0.5 = half speed)
        loop:     loop the motion
    """
    # Load model
    print(f"\nLoading G1 model: {G1_XML}")
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data  = mujoco.MjData(model)
    
    print(f"  Model actuators: {model.nu}")
    print(f"  Model qpos dim:  {model.nq}")
    
    # Load motion
    print()
    motion = load_csv_motion(csv_path)
    
    dt = 1.0 / (motion["fps"] * speed)
    
    print(f"\nStarting visualization...")
    print(f"  Speed: {speed}x")
    print(f"  Loop:  {loop}")
    print(f"  Press Ctrl+C to stop\n")
    
    frame = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set nice camera angle
        viewer.cam.azimuth   = 90
        viewer.cam.elevation = -20
        viewer.cam.distance  = 4.0
        viewer.cam.lookat    = np.array([0.0, 0.0, 0.8])
        
        while viewer.is_running():
            t_start = time.time()
            
            # Set robot to reference pose
            qpos = motion_to_qpos(motion, frame)
            data.qpos[:] = qpos
            data.qvel[:] = 0  # kinematic playback, no velocities
            
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            # Track root position with camera
            viewer.cam.lookat[0] = data.qpos[0]
            viewer.cam.lookat[1] = data.qpos[1]
            
            frame += 1
            
            # Loop or stop
            if frame >= motion["n_frames"]:
                if loop:
                    frame = 0
                    print(f"  Looping... (press Ctrl+C to stop)")
                else:
                    print("  Motion complete.")
                    break
            
            # Maintain correct playback speed
            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print("Viewer closed.")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize G1 walking motion")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV motion file. Auto-detected if not specified."
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed (default: 1.0)"
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="Don't loop the motion"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available motion files and exit"
    )
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("Available motion files:")
        all_csv = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
        for f in sorted(all_csv)[:50]:
            print(f"  {f}")
        if len(all_csv) > 50:
            print(f"  ... and {len(all_csv)-50} more")
        exit(0)
    
    # Find CSV
    csv_path = args.csv
    if csv_path is None:
        csv_path = find_csv(DATA_DIR)
        if csv_path is None:
            print(f"ERROR: No CSV files found in {DATA_DIR}")
            print("Did you download the dataset?")
            print()
            print("Download with:")
            print("  python -c \"")
            print("  from huggingface_hub import snapshot_download")
            print("  snapshot_download(")
            print("      repo_id='lvhaidong/LAFAN1_Retargeting_Dataset',")
            print("      repo_type='dataset',")
            print("      local_dir='./data/lafan1_retargeted'")
            print("  )\"")
            exit(1)
        print(f"Auto-selected: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        exit(1)
    
    if not os.path.exists(G1_XML):
        print(f"ERROR: G1 model not found: {G1_XML}")
        print("Clone with: git clone https://github.com/google-deepmind/mujoco_menagerie")
        exit(1)
    
    visualize(csv_path, speed=args.speed, loop=not args.no_loop)