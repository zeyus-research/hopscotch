import logging
import matplotlib
from matplotlib import pyplot as plt
from requests import head
matplotlib.use('QtAgg')
plt.ion()

from pathlib import Path
import kineticstoolkit.lab as ktk
from kineticstoolkit.timeseries import TimeSeries
import sys

import numpy as np
import pandas as pd
import yaml

from processing import mocap_loader

# Add src to path
sys.path.append('src')





def load_raw_data() -> pd.DataFrame:
    data_path = Path("data")
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create configuration
    config = {
        'marker_labels_file': Path('marker_labels_no_floor.txt'),
        'sampling_rate': 300.0,  # Adjust if needed
        'default_body_mass': 25.0,
        'cache_file': 'hopscotch_data_no_floor2.feather',
        'log_level': 'INFO',
        'test_type': 'independent',
    }
    
    loader = mocap_loader.MocapDataLoader(config.get('marker_labels_file', Path('marker_labels_no_floor.txt')),)
    
    cache_file = config.get('cache_file', 'hopscotch_data_no_floor2.feather')
        
    raw_data = loader.load_dataset(
        data_path=data_path,
        file_pattern = "*.tsv",
        cache_file=cache_file
    )
    
    logging.info(f"Loaded {len(raw_data)} records from {len(raw_data.groupby('filename'))} files")

    return raw_data

def raw_data_to_ktk_points(raw_data: pd.DataFrame) -> dict[str, TimeSeries]:
    

    # read marker labels
    marker_labels_file = Path('marker_labels_no_floor.txt')
    with open(marker_labels_file, 'r') as f:
        marker_labels = [line.strip() for line in f.readlines()]
    
    output = {}

    point_factor = 0.001  # Convert mm to meters
    subjects = raw_data['filename'].unique()
    for subject in subjects:
        subject_data = raw_data[raw_data['filename'] == subject]
        logging.info(f"Processing subject {subject} with {len(subject_data)} samples")
        points = TimeSeries()
        times = subject_data['Time'].values
        # print duplicate times
        if len(times) != len(set(times)):
            logging.warning(f"Duplicate time values found for subject {subject}.")
            # find duplicate times
            duplicates = subject_data['Time'][subject_data['Time'].duplicated()].unique()
            logging.warning(f"Duplicate times: {duplicates}")
            # remove duplicate times
            subject_data = subject_data.drop_duplicates(subset=['Time'])
            times = subject_data['Time'].values
            logging.info(f"After removing duplicates, {len(subject_data)} samples remain for subject {subject}.")

        n_samples = len(subject_data)

        for marker in marker_labels:
            x_col = f"{marker}.X"
            y_col = f"{marker}.Y"
            z_col = f"{marker}.Z"
            if x_col in subject_data.columns and y_col in subject_data.columns and z_col in subject_data.columns:
                data_array = np.ndarray([
                    n_samples,
                    4
                ], dtype=np.float32)
                data_array[:, 0] = subject_data[x_col].values * point_factor
                data_array[:, 1] = subject_data[y_col].values * point_factor
                data_array[:, 2] = subject_data[z_col].values * point_factor
                data_array[:, 3] = 1.0
                points.data[marker] = data_array
            else:
                logging.warning(f"Marker {marker} not found in data columns.")
        
        points.time = times
        output[f"Subject_{subject}"] = points

    return output


data = load_raw_data()
ktk_points = raw_data_to_ktk_points(data)
subjects = list(ktk_points.keys())

# add "bones" to the player for better visualization
# available markers:
# head
# foot_front_r
# foot_back_r
# knee_under_r
# knee_over_r
# wrist_r
# elbow_r
# shoulder_r
# hip_front_r
# hip_back_r
# foot_back_l
# foot_front_l
# knee_under_l
# knee_over_l
# hip_front_l
# hip_back_l
# wrist_l
# elbow_l
# shoulder_l

interconnections = {
    "ForearmL": {
        "Color": [1, 0, 0],
        "Links":  [
            ["wrist_l", "elbow_l"],
        ],
    },
    "UpperArmL": {
        "Color": [0, 1, 0],
        "Links":  [
            ["elbow_l", "shoulder_l"],
        ],
    },
    "ForearmR": {
        "Color": [1, 0, 0],
        "Links":  [
            ["wrist_r", "elbow_r"],
        ],
    },
    "UpperArmR": {
        "Color": [0, 1, 0],
        "Links":  [
            ["elbow_r", "shoulder_r"],
        ],
    },
    "ThighL": {
        "Color": [0, 0, 1],
        "Links":  [
            ["knee_over_l", "hip_front_l"],
            ["knee_over_l", "hip_back_l"],

        ],
    },
    "ShinL": {
        "Color": [1, 1, 0],
        "Links":  [
            ["knee_under_l", "foot_front_l"],
            ["knee_under_l", "foot_back_l"],
        ],
    },
    "ShinR": {
        "Color": [1, 1, 0],
        "Links":  [
            ["knee_under_r", "foot_front_r"],
            ["knee_under_r", "foot_back_r"],
        ],
    },
    "ThighR": {
        "Color": [0, 0, 1],
        "Links":  [
            ["knee_over_r", "hip_front_r"],
            ["knee_over_r", "hip_back_r"],
        ],
    },
    "FootL": {
        "Color": [1, 0, 1],
        "Links":  [
            ["foot_front_l", "foot_back_l"],
        ],
    },
    "FootR": {
        "Color": [1, 0, 1],
        "Links":  [
            ["foot_front_r", "foot_back_r"],
        ],
    },
    "Head": {
        "Color": [0, 1, 1],
        "Links":  [
            ["head", "shoulder_l"],
            ["head", "shoulder_r"],
        ],
    },
    "SideL": {
        "Color": [0.5, 0.5, 0.5],
        "Links":  [
            ["shoulder_l", "hip_front_l"],
            ["shoulder_l", "hip_back_l"],
        ],
    }, 
    "SideR": {
        "Color": [0.5, 0.5, 0.5],
        "Links":  [
            ["shoulder_r", "hip_front_r"],
            ["shoulder_r", "hip_back_r"],
        ],
    },
}
def create_segment_frames(points: TimeSeries) -> TimeSeries:
    """
    Create local coordinate frames for each body segment.

    Args:
        points: TimeSeries with marker positions (homogeneous coordinates)

    Returns:
        TimeSeries with coordinate frames for each segment
    """
    frames = ktk.TimeSeries(time=points.time)

    # Helper function to extract 3D coordinates (remove homogeneous coordinate)
    def get_3d(marker_name):
        return points.data[marker_name][:, :3]

    # === UPPER BODY ===

    # Left Upper Arm: origin at shoulder, z-axis toward elbow
    shoulder_l = get_3d('shoulder_l')
    elbow_l = get_3d('elbow_l')
    wrist_l = get_3d('wrist_l')

    upper_arm_l_z = elbow_l - shoulder_l
    # Use wrist to define the plane (helps define rotation around long axis)
    forearm_direction = wrist_l - elbow_l
    upper_arm_l_yz = np.cross(forearm_direction, upper_arm_l_z, axis=1)
    frames.data["UpperArmL"] = ktk.geometry.create_frames(
        origin=shoulder_l,
        z=upper_arm_l_z,
        yz=upper_arm_l_yz
    )

    # Left Forearm: origin at elbow, z-axis toward wrist
    forearm_l_z = wrist_l - elbow_l
    # Cross with upper arm to maintain consistency
    forearm_l_yz = np.cross(upper_arm_l_z, forearm_l_z, axis=1)
    frames.data["ForearmL"] = ktk.geometry.create_frames(
        origin=elbow_l,
        z=forearm_l_z,
        yz=forearm_l_yz
    )

    # Right Upper Arm
    shoulder_r = get_3d('shoulder_r')
    elbow_r = get_3d('elbow_r')
    wrist_r = get_3d('wrist_r')

    upper_arm_r_z = elbow_r - shoulder_r
    forearm_r_direction = wrist_r - elbow_r
    upper_arm_r_yz = np.cross(forearm_r_direction, upper_arm_r_z, axis=1)
    frames.data["UpperArmR"] = ktk.geometry.create_frames(
        origin=shoulder_r,
        z=upper_arm_r_z,
        yz=upper_arm_r_yz
    )

    # Right Forearm
    forearm_r_z = wrist_r - elbow_r
    forearm_r_yz = np.cross(upper_arm_r_z, forearm_r_z, axis=1)
    frames.data["ForearmR"] = ktk.geometry.create_frames(
        origin=elbow_r,
        z=forearm_r_z,
        yz=forearm_r_yz
    )

    # === LOWER BODY ===

    # Pelvis reference frame (average of hip markers)
    hip_front_l = get_3d('hip_front_l')
    hip_back_l = get_3d('hip_back_l')
    hip_front_r = get_3d('hip_front_r')
    hip_back_r = get_3d('hip_back_r')

    # Pelvis center and orientation
    pelvis_center = (hip_front_l + hip_back_l + hip_front_r + hip_back_r) / 4.0
    # X-axis: left to right
    pelvis_x = (hip_front_r + hip_back_r) / 2.0 - (hip_front_l + hip_back_l) / 2.0
    # Y-axis: back to front
    pelvis_y = (hip_front_l + hip_front_r) / 2.0 - (hip_back_l + hip_back_r) / 2.0
    frames.data["Pelvis"] = ktk.geometry.create_frames(
        origin=pelvis_center,
        x=pelvis_x,
        xy=pelvis_y
    )

    # Left Thigh: origin at hip, z-axis toward knee
    hip_l_center = (hip_front_l + hip_back_l) / 2.0
    knee_over_l = get_3d('knee_over_l')
    knee_under_l = get_3d('knee_under_l')

    thigh_l_z = knee_over_l - hip_l_center
    # Use pelvis x-axis to help define medial-lateral direction
    thigh_l_yz = np.cross(pelvis_x, thigh_l_z, axis=1)
    frames.data["ThighL"] = ktk.geometry.create_frames(
        origin=hip_l_center,
        z=thigh_l_z,
        yz=thigh_l_yz
    )

    # Left Shin: origin at knee, z-axis toward ankle
    foot_front_l = get_3d('foot_front_l')
    foot_back_l = get_3d('foot_back_l')
    ankle_l = (foot_front_l + foot_back_l) / 2.0

    shin_l_z = knee_under_l - ankle_l
    shin_l_yz = np.cross(thigh_l_z, shin_l_z, axis=1)
    frames.data["ShinL"] = ktk.geometry.create_frames(
        origin=knee_under_l,
        z=shin_l_z,
        yz=shin_l_yz
    )

    # Left Foot: origin at ankle, x-axis from heel to toe
    foot_l_x = foot_front_l - foot_back_l
    foot_l_xy = shin_l_z
    frames.data["FootL"] = ktk.geometry.create_frames(
        origin=ankle_l,
        x=foot_l_x,
        xy=foot_l_xy
    )

    # Right Thigh
    hip_r_center = (hip_front_r + hip_back_r) / 2.0
    knee_over_r = get_3d('knee_over_r')
    knee_under_r = get_3d('knee_under_r')

    thigh_r_z = knee_over_r - hip_r_center
    thigh_r_yz = np.cross(pelvis_x, thigh_r_z, axis=1)
    frames.data["ThighR"] = ktk.geometry.create_frames(
        origin=hip_r_center,
        z=thigh_r_z,
        yz=thigh_r_yz
    )

    # Right Shin
    foot_front_r = get_3d('foot_front_r')
    foot_back_r = get_3d('foot_back_r')
    ankle_r = (foot_front_r + foot_back_r) / 2.0

    shin_r_z = knee_under_r - ankle_r
    shin_r_yz = np.cross(thigh_r_z, shin_r_z, axis=1)
    frames.data["ShinR"] = ktk.geometry.create_frames(
        origin=knee_under_r,
        z=shin_r_z,
        yz=shin_r_yz
    )

    # Right Foot
    foot_r_x = foot_front_r - foot_back_r
    foot_r_xy = shin_r_z
    frames.data["FootR"] = ktk.geometry.create_frames(
        origin=ankle_r,
        x=foot_r_x,
        xy=foot_r_xy
    )

    # Trunk: from pelvis to shoulders
    shoulder_center = (shoulder_l + shoulder_r) / 2.0
    trunk_z = shoulder_center - pelvis_center
    trunk_x = shoulder_r - shoulder_l
    # Use pelvis forward direction to define the plane
    frames.data["Trunk"] = ktk.geometry.create_frames(
        origin=pelvis_center,
        z=trunk_z,
        x=trunk_x,
        xy=pelvis_y  # Use pelvis y-axis to define the xz plane
    )

    return frames


def calculate_joint_angles(points: TimeSeries) -> TimeSeries:
    """
    Calculate joint angles from marker positions.

    Uses simple geometric angles between body segments.
    For DTW analysis, these relative angles are much better than
    absolute marker positions.

    Args:
        points: TimeSeries with marker positions (homogeneous coordinates)

    Returns:
        TimeSeries with joint angles (in radians, with 3D components)
    """
    angles = ktk.TimeSeries(time=points.time)

    # Helper function to extract 3D coordinates
    def get_3d(marker_name):
        return points.data[marker_name][:, :3]

    # Helper function to calculate angle between two vectors
    def vector_angle(v1, v2):
        """Calculate angle between vectors v1 and v2 (n_frames, 3)"""
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        # Dot product gives cos(angle)
        cos_angle = np.sum(v1_norm * v2_norm, axis=1)
        # Clamp to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # Return angle in radians
        return np.arccos(cos_angle)

    # === ELBOW ANGLES ===
    # Left elbow: angle between upper arm (shoulder->elbow) and forearm (elbow->wrist)
    upper_arm_l = get_3d('elbow_l') - get_3d('shoulder_l')
    forearm_l = get_3d('wrist_l') - get_3d('elbow_l')
    angles.data["ElbowL"] = vector_angle(upper_arm_l, forearm_l).reshape(-1, 1)

    # Right elbow
    upper_arm_r = get_3d('elbow_r') - get_3d('shoulder_r')
    forearm_r = get_3d('wrist_r') - get_3d('elbow_r')
    angles.data["ElbowR"] = vector_angle(upper_arm_r, forearm_r).reshape(-1, 1)

    # === KNEE ANGLES ===
    # Left knee: angle between thigh and shin
    hip_l = (get_3d('hip_front_l') + get_3d('hip_back_l')) / 2.0
    knee_l = get_3d('knee_over_l')
    ankle_l = (get_3d('foot_front_l') + get_3d('foot_back_l')) / 2.0

    thigh_l = knee_l - hip_l
    shin_l = ankle_l - get_3d('knee_under_l')
    angles.data["KneeL"] = vector_angle(thigh_l, shin_l).reshape(-1, 1)

    # Right knee
    hip_r = (get_3d('hip_front_r') + get_3d('hip_back_r')) / 2.0
    knee_r = get_3d('knee_over_r')
    ankle_r = (get_3d('foot_front_r') + get_3d('foot_back_r')) / 2.0

    thigh_r = knee_r - hip_r
    shin_r = ankle_r - get_3d('knee_under_r')
    angles.data["KneeR"] = vector_angle(thigh_r, shin_r).reshape(-1, 1)

    # === HIP ANGLES ===
    # Hip angle: angle between pelvis and thigh
    pelvis_center = (get_3d('hip_front_l') + get_3d('hip_back_l') +
                     get_3d('hip_front_r') + get_3d('hip_back_r')) / 4.0
    shoulder_center = (get_3d('shoulder_l') + get_3d('shoulder_r')) / 2.0
    trunk = shoulder_center - pelvis_center

    angles.data["HipL"] = vector_angle(trunk, thigh_l).reshape(-1, 1)
    angles.data["HipR"] = vector_angle(trunk, thigh_r).reshape(-1, 1)

    # === ANKLE ANGLES ===
    # Ankle angle: angle between shin and foot
    foot_l = get_3d('foot_front_l') - get_3d('foot_back_l')
    foot_r = get_3d('foot_front_r') - get_3d('foot_back_r')

    shin_l_vec = get_3d('knee_under_l') - ankle_l
    shin_r_vec = get_3d('knee_under_r') - ankle_r

    angles.data["AnkleL"] = vector_angle(shin_l_vec, foot_l).reshape(-1, 1)
    angles.data["AnkleR"] = vector_angle(shin_r_vec, foot_r).reshape(-1, 1)

    # === SHOULDER ANGLES ===
    # Shoulder angle: angle between trunk and upper arm
    angles.data["ShoulderL"] = vector_angle(trunk, upper_arm_l).reshape(-1, 1)
    angles.data["ShoulderR"] = vector_angle(trunk, upper_arm_r).reshape(-1, 1)

    # === ADDITIONAL USEFUL ANGLES ===
    # Trunk lean angle (relative to vertical)
    vertical = np.tile([0, 0, 1], (trunk.shape[0], 1))
    angles.data["TrunkLean"] = vector_angle(trunk, vertical).reshape(-1, 1)

    # Arm extension (angle between shoulders and wrists - indicates arm position)
    shoulder_width = get_3d('shoulder_r') - get_3d('shoulder_l')
    arm_span_l = get_3d('wrist_l') - get_3d('shoulder_l')
    arm_span_r = get_3d('wrist_r') - get_3d('shoulder_r')
    angles.data["ArmExtensionL"] = vector_angle(shoulder_width, arm_span_l).reshape(-1, 1)
    angles.data["ArmExtensionR"] = vector_angle(shoulder_width, arm_span_r).reshape(-1, 1)

    return angles


def flatten_angles_for_dtw(angles: TimeSeries) -> np.ndarray:
    """
    Flatten joint angles into a 2D array suitable for DTW.

    Args:
        angles: TimeSeries with joint angle data

    Returns:
        Array of shape (n_frames, n_features) where each row is a time point
        and columns are flattened angle features
    """
    features = []

    for joint_name in sorted(angles.data.keys()):
        joint_angles = angles.data[joint_name]  # Shape: (n_frames, 3) for xyz Euler angles
        features.append(joint_angles)

    # Stack all features horizontally
    return np.hstack(features)


# Test the implementation
print("Creating segment frames...")
frames = create_segment_frames(ktk_points[subjects[0]])
print(f"Created frames for segments: {list(frames.data.keys())}")

print("\nCalculating joint angles...")
angles = calculate_joint_angles(ktk_points[subjects[0]])
print(f"Calculated angles for joints: {list(angles.data.keys())}")

# Show example angle data
print("\nExample - Left Knee angles (first 5 frames):")
print(angles.data["KneeL"][:5])

print("\nFlattened features for DTW (shape):")
dtw_features = flatten_angles_for_dtw(angles)
print(f"Shape: {dtw_features.shape} (n_frames × n_features)")
print(f"First row (frame 0): {dtw_features[0]}")

# Visualize with the player - shows markers, bones, and local coordinate frames
print("\n=== VISUALIZATION ===")
print("Opening 3D player with:")
print("- Markers (points)")
print("- Bone interconnections (colored lines)")
print("- Local coordinate frames (RGB = XYZ axes)")
print("\nControls:")
print("- Play/Pause: spacebar")
print("- Seek: drag timeline")
print("- Rotate: left mouse drag")
print("- Zoom: scroll wheel")
print("- Close window or Ctrl+C to exit")
print("\nStarting visualization...")

p = ktk.Player(
    ktk_points[subjects[0]],
    up="z",
    interconnections=interconnections
)
# Merge the coordinate frames with the point data so they're visualized together
p.set_contents(ktk_points[subjects[0]].merge(frames))

# Also create a live plot of some key angles
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle(f'Joint Angles Over Time - {subjects[0]}')

time = angles.time

# Convert radians to degrees for easier interpretation
def rad_to_deg(rad_array):
    return np.degrees(rad_array.flatten())

# Plot key angles
axes[0, 0].plot(time, rad_to_deg(angles.data["KneeL"]), label='Left', color='blue')
axes[0, 0].plot(time, rad_to_deg(angles.data["KneeR"]), label='Right', color='red')
axes[0, 0].set_ylabel('Angle (degrees)')
axes[0, 0].set_title('Knee Angles')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(time, rad_to_deg(angles.data["ElbowL"]), label='Left', color='blue')
axes[0, 1].plot(time, rad_to_deg(angles.data["ElbowR"]), label='Right', color='red')
axes[0, 1].set_ylabel('Angle (degrees)')
axes[0, 1].set_title('Elbow Angles')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(time, rad_to_deg(angles.data["HipL"]), label='Left', color='blue')
axes[1, 0].plot(time, rad_to_deg(angles.data["HipR"]), label='Right', color='red')
axes[1, 0].set_ylabel('Angle (degrees)')
axes[1, 0].set_title('Hip Angles')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(time, rad_to_deg(angles.data["AnkleL"]), label='Left', color='blue')
axes[1, 1].plot(time, rad_to_deg(angles.data["AnkleR"]), label='Right', color='red')
axes[1, 1].set_ylabel('Angle (degrees)')
axes[1, 1].set_title('Ankle Angles')
axes[1, 1].legend()
axes[1, 1].grid(True)

axes[2, 0].plot(time, rad_to_deg(angles.data["ShoulderL"]), label='Left', color='blue')
axes[2, 0].plot(time, rad_to_deg(angles.data["ShoulderR"]), label='Right', color='red')
axes[2, 0].set_ylabel('Angle (degrees)')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_title('Shoulder Angles')
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].plot(time, rad_to_deg(angles.data["TrunkLean"]), color='green')
axes[2, 1].set_ylabel('Angle (degrees)')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_title('Trunk Lean')
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()

# wait for ctrl+c
try:
    while True:
        # sleep to reduce CPU usage
        plt.pause(0.1)
        pass
except KeyboardInterrupt:
    print("\nExiting...")

print("\nTest completed successfully!")
