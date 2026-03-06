"""
Joint angle calculator for motion capture data.
Computes pose-invariant joint angles from marker positions.

These angles are intrinsic to body configuration and independent of 
global body position/orientation - ideal for DTW comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging


def get_marker_position(df: pd.DataFrame, marker: str, frame: Optional[int] = None) -> np.ndarray:
    """
    Extract 3D position for a marker.
    
    Args:
        df: DataFrame with marker columns (marker.X, marker.Y, marker.Z)
        marker: Marker name (e.g., 'elbow_r')
        frame: Optional specific frame index. If None, returns all frames.
        
    Returns:
        Array of shape (3,) for single frame or (n_frames, 3) for all frames
    """
    try:
        x = df[f'{marker}.X'].values
        y = df[f'{marker}.Y'].values
        z = df[f'{marker}.Z'].values
        
        if frame is not None:
            return np.array([x[frame], y[frame], z[frame]])
        return np.column_stack([x, y, z])
    except KeyError:
        logging.warning(f"Marker {marker} not found in data")
        return np.array([])


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s) to unit length, handling zero vectors."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v
    else:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1, norms)  # Avoid division by zero
        return v / norms


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate angle between two vectors (or arrays of vectors).
    
    Args:
        v1, v2: Vectors of shape (3,) or (n_frames, 3)
        
    Returns:
        Angle in radians (scalar or array of shape (n_frames,))
    """
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    if v1.ndim == 1:
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    else:
        dot = np.clip(np.sum(v1_norm * v2_norm, axis=1), -1.0, 1.0)
    
    return np.arccos(dot)


def build_torso_coordinate_frame(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a local coordinate frame for the torso using hip markers.
    
    The torso frame is defined by:
    - Origin: midpoint of front hip markers
    - X-axis (lateral): right hip front → left hip front
    - Z-axis (forward): perpendicular to the plane formed by hip markers
    - Y-axis (up): cross product of Z and X
    
    Args:
        df: DataFrame with marker data
        
    Returns:
        Tuple of (origin, x_axis, y_axis, z_axis), each of shape (n_frames, 3)
    """
    # Get hip marker positions
    hip_front_l = get_marker_position(df, 'hip_front_l')
    hip_front_r = get_marker_position(df, 'hip_front_r')
    hip_back_l = get_marker_position(df, 'hip_back_l')
    hip_back_r = get_marker_position(df, 'hip_back_r')
    
    n_frames = len(hip_front_l)
    
    # Origin: midpoint of front hips
    origin = (hip_front_l + hip_front_r) / 2
    
    # X-axis (lateral): right → left (so positive X is left)
    x_axis = normalize_vector(hip_front_l - hip_front_r)
    
    # Vector pointing backward (front midpoint → back midpoint)
    back_midpoint = (hip_back_l + hip_back_r) / 2
    backward_vec = back_midpoint - origin
    
    # Z-axis (forward): perpendicular to X, in the horizontal plane
    # Use cross product to get a vector perpendicular to both X and the backward vector
    # Then project to get the forward direction
    y_temp = np.cross(backward_vec, x_axis)  # Roughly upward
    z_axis = normalize_vector(np.cross(x_axis, y_temp))  # Forward
    
    # Y-axis (up): cross product of Z and X
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    
    return origin, x_axis, y_axis, z_axis


def transform_to_local_frame(
    points: np.ndarray,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray
) -> np.ndarray:
    """
    Transform points from global coordinates to local coordinate frame.
    
    Args:
        points: Points to transform, shape (n_frames, 3)
        origin, x_axis, y_axis, z_axis: Local frame definition, each (n_frames, 3)
        
    Returns:
        Points in local coordinates, shape (n_frames, 3)
    """
    # Translate to origin
    centered = points - origin
    
    # Project onto local axes
    local_x = np.sum(centered * x_axis, axis=1)
    local_y = np.sum(centered * y_axis, axis=1)
    local_z = np.sum(centered * z_axis, axis=1)
    
    return np.column_stack([local_x, local_y, local_z])


def spherical_angles_from_vector(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 3D vector to spherical angles (elevation and azimuth).
    
    Args:
        v: Vector(s) of shape (3,) or (n_frames, 3)
        
    Returns:
        Tuple of (elevation, azimuth) in radians
        - elevation: angle from horizontal plane (positive = up)
        - azimuth: angle in horizontal plane from forward direction
    """
    v_norm = normalize_vector(v)
    
    if v.ndim == 1:
        # Single vector
        elevation = np.arcsin(np.clip(v_norm[1], -1.0, 1.0))  # Y component
        azimuth = np.arctan2(v_norm[0], v_norm[2])  # X/Z in horizontal plane
    else:
        # Array of vectors
        elevation = np.arcsin(np.clip(v_norm[:, 1], -1.0, 1.0))
        azimuth = np.arctan2(v_norm[:, 0], v_norm[:, 2])
    
    return elevation, azimuth


# =============================================================================
# INTRINSIC JOINT ANGLES (no external reference needed)
# =============================================================================

def calculate_elbow_angle(df: pd.DataFrame, side: str = 'r') -> np.ndarray:
    """
    Calculate elbow flexion angle (intrinsic - independent of body orientation).
    
    The elbow angle is defined by shoulder → elbow → wrist.
    0° = fully extended, increases with flexion.
    
    Args:
        df: DataFrame with marker data
        side: 'r' for right, 'l' for left
        
    Returns:
        Array of elbow angles in radians, shape (n_frames,)
    """
    shoulder = get_marker_position(df, f'shoulder_{side}')
    elbow = get_marker_position(df, f'elbow_{side}')
    wrist = get_marker_position(df, f'wrist_{side}')
    
    # Vectors forming the angle
    upper_arm = shoulder - elbow  # elbow → shoulder
    forearm = wrist - elbow       # elbow → wrist
    
    # Angle between them (π - this gives flexion angle)
    angle = np.pi - angle_between_vectors(upper_arm, forearm)
    
    return angle


def calculate_knee_angle(df: pd.DataFrame, side: str = 'r') -> np.ndarray:
    """
    Calculate knee flexion angle (intrinsic).
    
    Uses the midpoint of knee_over and knee_under as the knee joint center.
    The angle is defined by hip → knee → ankle (foot midpoint).
    0° = fully extended, increases with flexion.
    
    Args:
        df: DataFrame with marker data
        side: 'r' for right, 'l' for left
        
    Returns:
        Array of knee angles in radians, shape (n_frames,)
    """
    # Hip center (midpoint of front and back hip markers)
    hip_front = get_marker_position(df, f'hip_front_{side}')
    hip_back = get_marker_position(df, f'hip_back_{side}')
    hip = (hip_front + hip_back) / 2
    
    # Knee center (midpoint of over and under markers)
    knee_over = get_marker_position(df, f'knee_over_{side}')
    knee_under = get_marker_position(df, f'knee_under_{side}')
    knee = (knee_over + knee_under) / 2
    
    # Ankle/foot center (midpoint of foot markers)
    foot_front = get_marker_position(df, f'foot_front_{side}')
    foot_back = get_marker_position(df, f'foot_back_{side}')
    ankle = (foot_front + foot_back) / 2
    
    # Vectors forming the angle
    thigh = hip - knee    # knee → hip
    shank = ankle - knee  # knee → ankle
    
    # Angle (π - this gives flexion angle)
    angle = np.pi - angle_between_vectors(thigh, shank)
    
    return angle


def calculate_ankle_angle(df: pd.DataFrame, side: str = 'r') -> np.ndarray:
    """
    Calculate ankle angle (foot relative to shank).
    
    Approximated using knee center, ankle, and foot orientation.
    ~90° = neutral standing, <90° = dorsiflexion, >90° = plantarflexion.
    
    Args:
        df: DataFrame with marker data
        side: 'r' for right, 'l' for left
        
    Returns:
        Array of ankle angles in radians, shape (n_frames,)
    """
    # Knee center
    knee_over = get_marker_position(df, f'knee_over_{side}')
    knee_under = get_marker_position(df, f'knee_under_{side}')
    knee = (knee_over + knee_under) / 2
    
    # Ankle (midpoint of foot markers as proxy)
    foot_front = get_marker_position(df, f'foot_front_{side}')
    foot_back = get_marker_position(df, f'foot_back_{side}')
    ankle = (foot_front + foot_back) / 2
    
    # Foot direction vector
    foot_vec = foot_front - foot_back  # Points forward along foot
    
    # Shank vector (pointing down from knee to ankle)
    shank = ankle - knee
    
    # Angle between shank and foot
    angle = angle_between_vectors(-shank, foot_vec)  # Negate shank to point down
    
    return angle


# =============================================================================
# JOINT ANGLES IN LOCAL COORDINATE FRAME
# =============================================================================

def calculate_shoulder_angles(df: pd.DataFrame, side: str = 'r') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate shoulder angles (upper arm relative to torso).
    
    Returns two angles:
    - elevation: how high the arm is raised (0 = down, π/2 = horizontal, π = up)
    - azimuth: rotation in horizontal plane (0 = forward, π/2 = lateral)
    
    Args:
        df: DataFrame with marker data
        side: 'r' for right, 'l' for left
        
    Returns:
        Tuple of (elevation, azimuth) arrays in radians, each shape (n_frames,)
    """
    # Build torso coordinate frame
    origin, x_axis, y_axis, z_axis = build_torso_coordinate_frame(df)
    
    # Get shoulder and elbow positions
    shoulder = get_marker_position(df, f'shoulder_{side}')
    elbow = get_marker_position(df, f'elbow_{side}')
    
    # Upper arm vector in global coordinates
    upper_arm_global = elbow - shoulder
    
    # Transform to local torso frame
    upper_arm_local = transform_to_local_frame(
        shoulder + upper_arm_global,  # End point of upper arm
        shoulder,  # Use shoulder as origin for this transform
        x_axis, y_axis, z_axis
    )
    
    # Actually, let's reconsider - we want the direction relative to torso
    # Transform just the direction vector
    upper_arm_local = np.column_stack([
        np.sum(upper_arm_global * x_axis, axis=1),
        np.sum(upper_arm_global * y_axis, axis=1),
        np.sum(upper_arm_global * z_axis, axis=1)
    ])
    
    # Convert to spherical angles
    elevation, azimuth = spherical_angles_from_vector(upper_arm_local)
    
    # Adjust elevation so 0 = arm down, positive = arm raised
    elevation = -elevation  # Flip because Y is up but arm down is negative Y
    
    return elevation, azimuth


def calculate_hip_angles(df: pd.DataFrame, side: str = 'r') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate hip angles (thigh relative to torso).
    
    Returns two angles:
    - flexion: forward/backward leg swing (0 = straight down, positive = forward)
    - abduction: lateral leg spread (0 = together, positive = apart)
    
    Args:
        df: DataFrame with marker data
        side: 'r' for right, 'l' for left
        
    Returns:
        Tuple of (flexion, abduction) arrays in radians, each shape (n_frames,)
    """
    # Build torso coordinate frame
    origin, x_axis, y_axis, z_axis = build_torso_coordinate_frame(df)
    
    # Get hip and knee positions
    hip_front = get_marker_position(df, f'hip_front_{side}')
    hip_back = get_marker_position(df, f'hip_back_{side}')
    hip = (hip_front + hip_back) / 2
    
    knee_over = get_marker_position(df, f'knee_over_{side}')
    knee_under = get_marker_position(df, f'knee_under_{side}')
    knee = (knee_over + knee_under) / 2
    
    # Thigh vector in global coordinates (pointing down from hip to knee)
    thigh_global = knee - hip
    
    # Transform to local torso frame
    thigh_local = np.column_stack([
        np.sum(thigh_global * x_axis, axis=1),
        np.sum(thigh_global * y_axis, axis=1),
        np.sum(thigh_global * z_axis, axis=1)
    ])
    
    # Normalize thigh vector for angle calculations
    thigh_norm = normalize_vector(thigh_local)
    
    # Flexion: angle from vertical in the sagittal plane
    # Use the angle between thigh and the negative Y axis (down)
    # Project onto Y-Z plane first
    thigh_sagittal = thigh_norm.copy()
    thigh_sagittal[:, 0] = 0  # Remove lateral component
    thigh_sagittal = normalize_vector(thigh_sagittal)
    
    # Flexion is positive when leg is forward (positive Z in local frame)
    flexion = np.arcsin(np.clip(thigh_sagittal[:, 2], -1.0, 1.0))
    
    # Abduction: angle from vertical in the frontal plane
    # Project onto X-Y plane
    thigh_frontal = thigh_norm.copy()
    thigh_frontal[:, 2] = 0  # Remove forward component
    thigh_frontal = normalize_vector(thigh_frontal)
    
    # Abduction is positive when leg is spread outward
    if side == 'r':
        abduction = np.arcsin(np.clip(-thigh_frontal[:, 0], -1.0, 1.0))
    else:
        abduction = np.arcsin(np.clip(thigh_frontal[:, 0], -1.0, 1.0))
    
    return flexion, abduction


def calculate_trunk_inclination(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate trunk inclination angles.
    
    Returns:
    - forward_lean: positive = leaning forward
    - lateral_lean: positive = leaning right
    
    Args:
        df: DataFrame with marker data
        
    Returns:
        Tuple of (forward_lean, lateral_lean) in radians
    """
    # Use shoulder midpoint and hip midpoint to define trunk axis
    shoulder_l = get_marker_position(df, 'shoulder_l')
    shoulder_r = get_marker_position(df, 'shoulder_r')
    shoulder_mid = (shoulder_l + shoulder_r) / 2
    
    hip_front_l = get_marker_position(df, 'hip_front_l')
    hip_front_r = get_marker_position(df, 'hip_front_r')
    hip_mid = (hip_front_l + hip_front_r) / 2
    
    # Trunk vector (pointing up from hips to shoulders)
    trunk = shoulder_mid - hip_mid
    trunk_norm = normalize_vector(trunk)
    
    # Vertical reference (assuming Y is up in Qualisys)
    vertical = np.array([0, 1, 0])
    
    # Forward lean: angle in sagittal plane
    # Project trunk onto Y-Z plane (sagittal), measure angle from vertical
    sagittal_proj = trunk_norm.copy()
    sagittal_proj[:, 0] = 0  # Zero out X component
    sagittal_proj = normalize_vector(sagittal_proj)
    
    # Forward lean is positive when Z component is positive (leaning forward)
    forward_lean = np.arctan2(trunk_norm[:, 2], trunk_norm[:, 1])
    
    # Lateral lean: angle in frontal plane
    # Project trunk onto X-Y plane, measure angle from vertical
    lateral_lean = np.arctan2(trunk_norm[:, 0], trunk_norm[:, 1])
    
    return forward_lean, lateral_lean


def calculate_head_angle(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate head position relative to shoulders.
    
    Since we only have one head marker, we measure its position relative
    to the shoulder midpoint.
    
    Returns:
    - head_forward: forward/backward tilt (positive = forward)
    - head_lateral: left/right tilt (positive = right)
    
    Args:
        df: DataFrame with marker data
        
    Returns:
        Tuple of (head_forward, head_lateral) in radians (as angles)
    """
    # Build torso frame for reference
    origin, x_axis, y_axis, z_axis = build_torso_coordinate_frame(df)
    
    head = get_marker_position(df, 'head')
    shoulder_l = get_marker_position(df, 'shoulder_l')
    shoulder_r = get_marker_position(df, 'shoulder_r')
    shoulder_mid = (shoulder_l + shoulder_r) / 2
    
    # Vector from shoulders to head
    head_vec = head - shoulder_mid
    
    # Transform to local frame
    head_local = np.column_stack([
        np.sum(head_vec * x_axis, axis=1),
        np.sum(head_vec * y_axis, axis=1),
        np.sum(head_vec * z_axis, axis=1)
    ])
    
    # Normalize for angle calculations
    head_norm = normalize_vector(head_local)
    
    # Forward tilt: use arcsin of Z component (bounded, no wrapping)
    head_forward = np.arcsin(np.clip(head_norm[:, 2], -1.0, 1.0))
    
    # Lateral tilt: use arcsin of X component
    head_lateral = np.arcsin(np.clip(head_norm[:, 0], -1.0, 1.0))
    
    return head_forward, head_lateral


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def calculate_all_joint_angles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all joint angles for a trial.
    
    Args:
        df: DataFrame with marker data
        
    Returns:
        DataFrame with joint angle time series
    """
    n_frames = len(df)
    
    angles = {
        'frame': np.arange(n_frames),
    }
    
    # Add time if available
    if 'Time' in df.columns:
        angles['time'] = df['Time'].values
    
    # Intrinsic angles (single scalar per frame)
    try:
        angles['elbow_r'] = calculate_elbow_angle(df, 'r')
        angles['elbow_l'] = calculate_elbow_angle(df, 'l')
    except Exception as e:
        logging.warning(f"Could not calculate elbow angles: {e}")
    
    try:
        angles['knee_r'] = calculate_knee_angle(df, 'r')
        angles['knee_l'] = calculate_knee_angle(df, 'l')
    except Exception as e:
        logging.warning(f"Could not calculate knee angles: {e}")
    
    try:
        angles['ankle_r'] = calculate_ankle_angle(df, 'r')
        angles['ankle_l'] = calculate_ankle_angle(df, 'l')
    except Exception as e:
        logging.warning(f"Could not calculate ankle angles: {e}")
    
    # Angles in local coordinate frame (two components each)
    try:
        shoulder_r_elev, shoulder_r_az = calculate_shoulder_angles(df, 'r')
        angles['shoulder_r_elevation'] = shoulder_r_elev
        angles['shoulder_r_azimuth'] = shoulder_r_az
        
        shoulder_l_elev, shoulder_l_az = calculate_shoulder_angles(df, 'l')
        angles['shoulder_l_elevation'] = shoulder_l_elev
        angles['shoulder_l_azimuth'] = shoulder_l_az
    except Exception as e:
        logging.warning(f"Could not calculate shoulder angles: {e}")
    
    try:
        hip_r_flex, hip_r_abd = calculate_hip_angles(df, 'r')
        angles['hip_r_flexion'] = hip_r_flex
        angles['hip_r_abduction'] = hip_r_abd
        
        hip_l_flex, hip_l_abd = calculate_hip_angles(df, 'l')
        angles['hip_l_flexion'] = hip_l_flex
        angles['hip_l_abduction'] = hip_l_abd
    except Exception as e:
        logging.warning(f"Could not calculate hip angles: {e}")
    
    try:
        forward_lean, lateral_lean = calculate_trunk_inclination(df)
        angles['trunk_forward_lean'] = forward_lean
        angles['trunk_lateral_lean'] = lateral_lean
    except Exception as e:
        logging.warning(f"Could not calculate trunk angles: {e}")
    
    try:
        head_forward, head_lateral = calculate_head_angle(df)
        angles['head_forward'] = head_forward
        angles['head_lateral'] = head_lateral
    except Exception as e:
        logging.warning(f"Could not calculate head angles: {e}")
    
    return pd.DataFrame(angles)


def convert_angles_to_ndarray(angles_df: pd.DataFrame, 
                              exclude_cols: List[str] = ['frame', 'time']) -> np.ndarray:
    """
    Convert joint angles DataFrame to ndarray for DTW.
    
    Args:
        angles_df: DataFrame from calculate_all_joint_angles
        exclude_cols: Columns to exclude from the array
        
    Returns:
        Array of shape (n_frames, n_angles) suitable for DTW
    """
    angle_cols = [col for col in angles_df.columns if col not in exclude_cols]
    return angles_df[angle_cols].values


def get_angle_labels() -> List[str]:
    """Get the list of angle labels in the order they appear in the array."""
    return [
        'elbow_r', 'elbow_l',
        'knee_r', 'knee_l', 
        'ankle_r', 'ankle_l',
        'shoulder_r_elevation', 'shoulder_r_azimuth',
        'shoulder_l_elevation', 'shoulder_l_azimuth',
        'hip_r_flexion', 'hip_r_abduction',
        'hip_l_flexion', 'hip_l_abduction',
        'trunk_forward_lean', 'trunk_lateral_lean',
        'head_forward', 'head_lateral'
    ]


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from processing.mocap_loader import MocapDataLoader
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    loader = MocapDataLoader(Path('marker_labels_no_floor.txt'))
    data = loader.load_dataset(
        Path('data'),
        file_pattern='*.tsv',
        cache_file='hopscotch_data_no_floor2.feather'
    )
    
    # Get one trial
    trial = data[data['filename'] == data['filename'].iloc[0]].copy()
    print(f"Trial: {trial['filename'].iloc[0]}, frames: {len(trial)}")
    
    # Calculate joint angles
    angles_df = calculate_all_joint_angles(trial)
    print(f"\nCalculated {len(angles_df.columns) - 1} angle time series")
    print(f"Angle columns: {[c for c in angles_df.columns if c != 'frame']}")
    
    # Summary statistics
    print("\nAngle statistics (in degrees):")
    for col in angles_df.columns:
        if col not in ['frame', 'time']:
            vals = np.degrees(angles_df[col].values)
            print(f"  {col}: mean={vals.mean():.1f}°, std={vals.std():.1f}°, "
                  f"range=[{vals.min():.1f}°, {vals.max():.1f}°]")
    
    # Convert to array for DTW
    angles_array = convert_angles_to_ndarray(angles_df)
    print(f"\nArray shape for DTW: {angles_array.shape}")
