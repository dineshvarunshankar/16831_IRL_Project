# Learning Locomotion 

This repository contains tools for visualizing retargeted human motion data on the Unitree G1 humanoid robot using MuJoCo.

## Prerequisites

Before running the visualization, you need to set up a conda environment and install the required dependencies.

### 1. Create a Conda Environment

Creating a conda environment:

```bash
conda create -n roblearn python=3.10
conda activate roblearn
```

### 2. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file. Make sure you are using at least `mujoco>=3.5.0`, as older versions do not support the `dampratio` attribute used in the G1 robot XML.

```bash
pip install -r requirements.txt
```

### 3. Clone MuJoCo Menagerie (If not present)

The visualization requires the MuJoCo Menagerie model for the Unitree G1. If you don't already have it in the repository:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

## Running the Visualization

The main script is `visualize.py`. It can automatically find and play back the LAFAN1 motion data from the dataset. 

By default, the script looks for `.csv` files inside `./data/lafan1_retargeted`.

### Usage

**Run with auto-detection:**
If you just run the script, it will automatically find the first available `walk*` CSV file and visualize it in an interactive window.
```bash
python visualize.py
```

**Run a specific CSV file:**
```bash
python visualize.py --csv data/lafan1_retargeted/walk1_subject1.csv
```

**Run at half speed:**
```bash
python visualize.py --speed 0.5
```

**List all available motions:**
```bash
python visualize.py --list
```

## Dataset

If you need the LAFAN1 retargeted dataset, you can download it from Hugging Face using the command suggested by the script:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lvhaidong/LAFAN1_Retargeting_Dataset',
    repo_type='dataset',
    local_dir='./data/lafan1_retargeted'
)"
```
