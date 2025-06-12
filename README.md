# Egocentric Event-Based Vision for Ping Pong Ball Trajectory Prediction

This repository contains the official code for the paper **Egocentric Event-Based Vision for Ping Pong Ball Trajectory Prediction**  ([paper](https://rpg.ifi.uzh.ch/docs/Alberico_cvprw25.pdf)). This paper has been accepted for publication at the IEEE Computer Vision and Pattern Recognition Workshop (CVPRW), Nashville, 2025. ©IEEE

---

## 📝 Project Overview

In this work, we present a real-time egocentric trajectory prediction system for table tennis using event cameras. Unlike standard cameras, which suffer from high latency and motion blur at fast ball speeds, event cameras provide higher temporal resolution, enabling more frequent state updates, greater robustness to outliers, and accurate trajectory predictions using just a short time window after the opponent’s impact. This is the first framework for egocentric table-tennis ball trajectory prediction using event cameras.

<p align="center">
  <img src="media/pipeline_examples.png" alt="Pipeline Example" width="600"/>
</p>

---

## 🛠️ Setup Instructions

Follow these steps to set up the environment and run the project.

## 📦 Requirements

- Python 3.8 installed  
- Git (optional, for cloning the project)

### 1. Clone the Repository

```bash
git clone https://github.com/uzh-rpg/event_based_ping_pong_ball_trajectory_prediction.git
cd event_based_ping_pong_ball_trajectory_prediction
```

Or download the ZIP and extract it.

### 2. Create a Virtual Environment

```bash
python3.8 -m venv venv
```

### 3. Activate the Virtual Environment

- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```

### 4. Install Dependencies

Make sure you're in the same directory as `requirements.txt`:

```bash
pip install -r requirements.txt
```
### 5. Run the Project

#### Input Folder Structure

Before running the pipeline, ensure your input data is organized as follows:

```
dataset_folder/
├── aria_recording.vrs
├── eye_gaze/
│   └── ... (eye gaze data files)
├── slam/
│   └── ... (SLAM/odometry files)
├── your_sequence_folder/
│   ├── config.yml
│   └── ... (other sequence files)
```

- The folder you specify for evaluation (e.g., `./data/game_sequence_test_1`) **must be inside a dataset folder** that also contains:
  - `aria_recording.vrs`
  - `eye_gaze/`
  - `slam/`
- This structure is required to load all necessary Aria information.
- Example:  
  If you run  
  ```bash
  python3.8 ./perception_pipeline_ball.py ./data/game_sequence_test_1
  ```
  then `./data/` should contain `aria_recording.vrs`, `eye_gaze/`, and `slam/`, and `game_sequence_test_1/` should contain `config.yml`.

- You can download the `dataset_folder` from: [this link](https://www.swisstransfer.com/d/6edf5a5b-55e5-4d73-83bb-5bee140e47cb)
- The `game_sequence_test_1` example can be unzipped from `data/game_sequence_test_1.zip`.

#### Model Path Configuration

- The pipeline uses a pre-trained DCGM model for trajectory prediction.
- **Before running**, extract `trained_DCGM_model.zip` (provided separately) into your `/data` directory or another location of your choice.
- In your `config.yml` (inside the sequence folder), set the `path_to_DCGM_model` parameter under the `io` section to the extracted model path. 

#### Running the Pipeline

For every sequence, there is a `config.yml` file inside the sequence folder. You can change parameters and settings for your experiments by editing this config file.