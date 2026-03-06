"""
Test script for joint angle calculations.
Verifies the joint_angles module works with your existing data pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# For testing, create mock data that matches your structure
def create_mock_trial_data(n_frames: int = 100) -> pd.DataFrame:
    """
    Create mock motion capture data matching your marker structure.
    Simulates a simple jumping motion.
    """
    np.random.seed(42)
    
    # Time vector
    t = np.linspace(0, n_frames/300, n_frames)  # 300 Hz sampling
    
    # Base positions (standing pose, in mm, Y is up)
    base_positions = {
        'head': [0, 1700, 0],
        'shoulder_r': [-200, 1400, 0],
        'shoulder_l': [200, 1400, 0],
        'elbow_r': [-300, 1150, 0],
        'elbow_l': [300, 1150, 0],
        'wrist_r': [-350, 900, 50],
        'wrist_l': [350, 900, 50],
        'hip_front_r': [-100, 900, 50],
        'hip_front_l': [100, 900, 50],
        'hip_back_r': [-100, 900, -50],
        'hip_back_l': [100, 900, -50],
        'knee_over_r': [-100, 500, 30],
        'knee_over_l': [100, 500, 30],
        'knee_under_r': [-100, 450, 30],
        'knee_under_l': [100, 450, 30],
        'foot_front_r': [-100, 50, 100],
        'foot_front_l': [100, 50, 100],
        'foot_back_r': [-100, 50, -50],
        'foot_back_l': [100, 50, -50],
    }
    
    data = {'Frame': np.arange(n_frames), 'Time': t}
    
    # Add movement: simulate a jump with arm swing
    # Vertical oscillation
    vertical_motion = 100 * np.sin(2 * np.pi * 2 * t)  # Jump up and down
    
    # Arm swing
    arm_swing = 50 * np.sin(2 * np.pi * 2 * t + np.pi/4)
    
    # Knee bend (opposite phase to jump)
    knee_bend = 50 * np.sin(2 * np.pi * 2 * t + np.pi)
    
    for marker, base_pos in base_positions.items():
        # Add base position + movement + noise
        x = base_pos[0] + np.random.randn(n_frames) * 5
        y = base_pos[1] + vertical_motion + np.random.randn(n_frames) * 5
        z = base_pos[2] + np.random.randn(n_frames) * 5
        
        # Add specific movements
        if 'wrist' in marker or 'elbow' in marker:
            z += arm_swing if '_r' in marker else -arm_swing
        if 'knee' in marker:
            y += knee_bend
            z += knee_bend * 0.5
        
        data[f'{marker}.X'] = x
        data[f'{marker}.Y'] = y
        data[f'{marker}.Z'] = z
    
    return pd.DataFrame(data)


def test_basic_calculations():
    """Test basic joint angle calculations."""
    print("=" * 60)
    print("Testing Joint Angle Calculations")
    print("=" * 60)
    
    # Import the module
    from joint_angles import (
        calculate_all_joint_angles,
        convert_angles_to_ndarray,
        get_angle_labels,
        calculate_elbow_angle,
        calculate_knee_angle,
        build_torso_coordinate_frame
    )
    
    # Create mock data
    df = create_mock_trial_data(100)
    print(f"\nCreated mock trial data: {len(df)} frames")
    print(f"Markers: {[c.replace('.X', '') for c in df.columns if c.endswith('.X')]}")
    
    # Test torso frame construction
    print("\n--- Testing Torso Coordinate Frame ---")
    origin, x_axis, y_axis, z_axis = build_torso_coordinate_frame(df)
    print(f"Origin shape: {origin.shape}")
    print(f"First frame origin: {origin[0]}")
    print(f"X-axis (lateral) sample: {x_axis[0]}")
    print(f"Y-axis (up) sample: {y_axis[0]}")
    print(f"Z-axis (forward) sample: {z_axis[0]}")
    
    # Verify orthogonality
    dot_xy = np.abs(np.sum(x_axis[0] * y_axis[0]))
    dot_xz = np.abs(np.sum(x_axis[0] * z_axis[0]))
    dot_yz = np.abs(np.sum(y_axis[0] * z_axis[0]))
    print(f"Orthogonality check (should be ~0): XY={dot_xy:.4f}, XZ={dot_xz:.4f}, YZ={dot_yz:.4f}")
    
    # Test individual angles
    print("\n--- Testing Individual Angles ---")
    
    elbow_r = calculate_elbow_angle(df, 'r')
    print(f"Right elbow: mean={np.degrees(np.mean(elbow_r)):.1f}°, "
          f"range=[{np.degrees(np.min(elbow_r)):.1f}°, {np.degrees(np.max(elbow_r)):.1f}°]")
    
    knee_r = calculate_knee_angle(df, 'r')
    print(f"Right knee: mean={np.degrees(np.mean(knee_r)):.1f}°, "
          f"range=[{np.degrees(np.min(knee_r)):.1f}°, {np.degrees(np.max(knee_r)):.1f}°]")
    
    # Test full calculation
    print("\n--- Testing Full Joint Angle Calculation ---")
    angles_df = calculate_all_joint_angles(df)
    print(f"Output DataFrame shape: {angles_df.shape}")
    print(f"Columns: {list(angles_df.columns)}")
    
    # Convert to array for DTW
    angles_array = convert_angles_to_ndarray(angles_df)
    print(f"\nArray for DTW: shape={angles_array.shape}")
    print(f"Expected labels: {get_angle_labels()}")
    
    # Summary statistics
    print("\n--- Angle Summary Statistics (degrees) ---")
    for col in angles_df.columns:
        if col not in ['frame', 'time']:
            vals = np.degrees(angles_df[col].values)
            print(f"  {col:25s}: mean={vals.mean():7.1f}°, std={vals.std():5.1f}°")
    
    return angles_df


def test_visualization(angles_df: pd.DataFrame):
    """Create visualization of joint angles over time."""
    print("\n--- Creating Visualization ---")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Joint Angles Over Time (Mock Data)', fontsize=14, fontweight='bold')
    
    frames = angles_df['frame'].values
    
    # Plot groups of related angles
    angle_groups = [
        (['elbow_r', 'elbow_l'], 'Elbow Flexion'),
        (['knee_r', 'knee_l'], 'Knee Flexion'),
        (['shoulder_r_elevation', 'shoulder_l_elevation'], 'Shoulder Elevation'),
        (['hip_r_flexion', 'hip_l_flexion'], 'Hip Flexion'),
        (['trunk_forward_lean', 'trunk_lateral_lean'], 'Trunk Inclination'),
        (['hip_r_abduction', 'hip_l_abduction'], 'Hip Abduction'),
    ]
    
    for ax, (cols, title) in zip(axes.flat, angle_groups):
        for col in cols:
            if col in angles_df.columns:
                ax.plot(frames, np.degrees(angles_df[col]), label=col, alpha=0.8)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/joint_angles_test.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to /home/claude/joint_angles_test.png")
    
    return fig


def test_dtw_integration():
    """Test that the output is suitable for DTW analysis."""
    print("\n" + "=" * 60)
    print("Testing DTW Integration")
    print("=" * 60)
    
    from joint_angles import calculate_all_joint_angles, convert_angles_to_ndarray
    
    # Create two mock trials with slightly different movements
    np.random.seed(42)
    trial1 = create_mock_trial_data(100)
    
    np.random.seed(123)
    trial2 = create_mock_trial_data(120)  # Different length
    
    # Calculate angles for both
    angles1 = calculate_all_joint_angles(trial1)
    angles2 = calculate_all_joint_angles(trial2)
    
    # Convert to arrays
    array1 = convert_angles_to_ndarray(angles1)
    array2 = convert_angles_to_ndarray(angles2)
    
    print(f"Trial 1 angles shape: {array1.shape}")
    print(f"Trial 2 angles shape: {array2.shape}")
    
    # Check for NaN/Inf values
    nan_count1 = np.sum(np.isnan(array1))
    nan_count2 = np.sum(np.isnan(array2))
    inf_count1 = np.sum(np.isinf(array1))
    inf_count2 = np.sum(np.isinf(array2))
    
    print(f"NaN values: trial1={nan_count1}, trial2={nan_count2}")
    print(f"Inf values: trial1={inf_count1}, trial2={inf_count2}")
    
    # Verify dtype is suitable for DTW
    print(f"Dtype: {array1.dtype}")
    
    # Simple DTW distance calculation (if dtaidistance is available)
    try:
        from dtaidistance import dtw_ndim
        
        distance = dtw_ndim.distance(array1, array2)
        print(f"\nDTW distance between trials: {distance:.2f}")
        print("✓ DTW integration successful!")
        
    except ImportError:
        print("\n(dtaidistance not available for full DTW test)")
        print("But array format is correct for DTW!")
    
    return array1, array2


if __name__ == "__main__":
    # Run tests
    angles_df = test_basic_calculations()
    test_visualization(angles_df)
    test_dtw_integration()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
