"""
Batch convert CSV motion files to NPZ format for Unitree RL mjlab.

This script finds all CSV files in the data/lafan1_retargeted directory
and converts them to NPZ format using Unitree RL mjlab's conversion tool.

Usage:
    python scripts/convert_motions.py
    python scripts/convert_motions.py --input-dir data/lafan1_retargeted/g1
    python scripts/convert_motions.py --output-dir data/motions_npz --robot g1
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


def find_csv_files(input_dir: str) -> list[str]:
    """Find all CSV files in directory recursively."""
    pattern = os.path.join(input_dir, "**", "*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    return sorted(csv_files)


def convert_single_motion(
    csv_path: str,
    output_dir: str,
    robot: str = "g1",
    input_fps: int = 30,
    output_fps: int = 50
) -> bool:
    """
    Convert a single CSV file to NPZ using Unitree's tool.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save NPZ file
        robot: Robot type (g1, g1_23dof, etc.)
        input_fps: Input CSV frame rate
        output_fps: Output NPZ frame rate
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    # Get motion name from filename
    motion_name = Path(csv_path).stem
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        "unitree_rl_mjlab/scripts/csv_to_npz.py",
        "--input-file", csv_path,
        "--output-name", motion_name,
        "--robot", robot,
        "--input-fps", str(input_fps),
        "--output-fps", str(output_fps)
    ]
    
    print(f"\nConverting: {csv_path}")
    print(f"  → {motion_name}.npz")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert CSV motion files to NPZ format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/lafan1_retargeted",
        help="Directory containing CSV files (default: data/lafan1_retargeted)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/motions_npz",
        help="Directory to save NPZ files (default: data/motions_npz)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        choices=["g1", "g1_23dof", "go2", "h1_2"],
        help="Robot type (default: g1)"
    )
    parser.add_argument(
        "--input-fps",
        type=int,
        default=30,
        help="Input CSV frame rate (default: 30)"
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=50,
        help="Output NPZ frame rate (default: 50)"
    )
    args = parser.parse_args()
    
    # Check if Unitree RL mjlab exists
    if not os.path.exists("unitree_rl_mjlab/scripts/csv_to_npz.py"):
        print("Error: unitree_rl_mjlab not found!")
        print("Please add it as a git submodule:")
        print("  git submodule add https://github.com/unitreerobotics/unitree_rl_mjlab.git")
        sys.exit(1)
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = find_csv_files(args.input_dir)
    
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Convert each file
    success_count = 0
    fail_count = 0
    
    for csv_file in csv_files:
        success = convert_single_motion(
            csv_file,
            args.output_dir,
            robot=args.robot,
            input_fps=args.input_fps,
            output_fps=args.output_fps
        )
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(csv_files)}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
