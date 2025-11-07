#!/usr/bin/env python3
"""
Script to calculate motion energy from motion capture data.
This can analyze one or multiple body points over time from the hopscotch dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import pickle
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate motion energy from mocap data")
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/mocap_data.feather",
        help="Path to mocap data file (feather format)"
    )
    parser.add_argument(
        "--subjects", 
        type=int, 
        nargs="+",
        help="Subject IDs to analyze (can specify multiple)"
    )
    parser.add_argument(
        "--conditions", 
        type=str, 
        nargs="+",
        help="Conditions to analyze (e.g., 'k' or 'h', can specify multiple)"
    )
    parser.add_argument(
        "--obstacles", 
        type=int, 
        nargs="+",
        help="Obstacle counts to analyze (can specify multiple)"
    )
    parser.add_argument(
        "--points", 
        type=str, 
        nargs="+",
        default=["head", "foot_front_r", "foot_front_l"],
        help="Body points to analyze (without .X/.Y/.Z suffix)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["velocity", "jerk", "composite"],
        default="velocity",
        help="Energy calculation method to use"
    )
    parser.add_argument(
        "--sampling-rate", 
        type=int, 
        default=300,
        help="Sampling rate in Hz"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="energy_results.pkl",
        help="Output file path for saving results (pickle format)"
    )
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate plots of energy over time"
    )
    parser.add_argument(
        "--plot-dir", 
        type=str, 
        default="energy_plots",
        help="Directory for saving energy plots"
    )
    parser.add_argument(
        "--by-time", 
        action="store_true",
        help="Calculate energy per second in addition to total energy"
    )
    parser.add_argument(
        "--normalize", 
        action="store_true",
        help="Normalize energy results by trial duration"
    )
    return parser.parse_args()


def load_data(filepath, subjects=None, conditions=None, obstacles=None):
    """Load mocap data with optional filtering."""
    try:
        data = pd.read_feather(filepath)
        logging.info(f"Loaded data with shape: {data.shape}")
        
        # Apply filters if specified
        if subjects is not None:
            data = data[data['subject'].isin(subjects)]
        if conditions is not None:
            data = data[data['condition'].isin(conditions)]
        if obstacles is not None:
            data = data[data['obstacles'].isin(obstacles)]
            
        if len(data) == 0:
            raise ValueError("No data matches the specified filters")
        
        # Log how many trials we're working with
        trial_count = data.groupby(['subject', 'condition', 'obstacles']).ngroups
        logging.info(f"Found {trial_count} unique trials matching filters")
            
        logging.info(f"Filtered data shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found: {filepath}")
        raise


def prepare_points_data(data, points):
    """Extract coordinates for specified body points."""
    
    # Group data by subject, condition, and obstacles
    subject_condition_groups = data.groupby(['subject', 'condition', 'obstacles'])
    
    coords_data = {}
    
    for (subject_id, condition, obstacles), group_data in subject_condition_groups:
        for point in points:
            x_col = f"{point}.X"
            y_col = f"{point}.Y"
            z_col = f"{point}.Z"
            
            if not all(col in group_data.columns for col in [x_col, y_col, z_col]):
                logging.warning(f"Point {point} not found in data for subject {subject_id}, condition {condition}")
                continue
                
            # Extract coordinates
            x = group_data[x_col].values
            y = group_data[y_col].values
            z = group_data[z_col].values
            
            # Check for empty data or NaN/Inf values
            if len(x) == 0 or np.all(np.isnan(x)) or np.all(np.isnan(y)) or np.all(np.isnan(z)) or \
               np.all(np.isinf(x)) or np.all(np.isinf(y)) or np.all(np.isinf(z)):
                logging.warning(f"Skipping {point} for subject {subject_id}, condition {condition}, obstacles {obstacles} due to invalid data")
                continue
                
            # Replace any remaining NaN/Inf values with interpolation
            mask = np.logical_or.reduce([
                np.isnan(x), np.isnan(y), np.isnan(z),
                np.isinf(x), np.isinf(y), np.isinf(z)
            ])
            
            if np.any(mask):
                logging.warning(f"Interpolating {np.sum(mask)} invalid values for {point}, subject {subject_id}, condition {condition}")
                # Create indices array for interpolation
                indices = np.arange(len(x))
                # Perform interpolation for each coordinate
                valid_indices = indices[~mask]
                
                if len(valid_indices) > 0:
                    x = np.interp(indices, valid_indices, x[~mask])
                    y = np.interp(indices, valid_indices, y[~mask])
                    z = np.interp(indices, valid_indices, z[~mask])
                else:
                    logging.warning(f"No valid data points for interpolation, skipping {point}")
                    continue
                
            # Create a unique identifier for this point-subject-condition combination
            point_key = f"{point}_S{subject_id}_C{condition}_O{obstacles}"
            
            coords_data[point_key] = {
                'x': x,
                'y': y,
                'z': z,
                'point': point,
                'subject': subject_id,
                'condition': condition,
                'obstacles': obstacles
            }
    
    return coords_data


def calculate_motion_energy_velocity(marker_data, sampling_rate=300, mass=1.0):
    """
    Calculate kinetic energy from marker position data based on velocity
    
    Parameters:
    - marker_data: numpy array of shape [frames, 3] (x,y,z) - assumed in mm
    - sampling_rate: Hz (frames per second)
    - mass: arbitrary constant (can be set to 1.0 for relative comparisons)
    
    Returns:
    - energy_per_frame: array of energy values for each frame
    - total_energy: sum of energy across all frames
    """
    # Safety check for data length
    if len(marker_data) < 2:
        return np.array([0]), 0
    
    # Convert from mm to m (common mocap unit conversion)
    marker_data_m = marker_data / 1000.0
    
    # Calculate velocity (first derivative of position)
    velocity = np.diff(marker_data_m, axis=0) * sampling_rate
    
    # Calculate speed (magnitude of velocity vector)
    speed = np.linalg.norm(velocity, axis=1)
    
    # Calculate kinetic energy (½mv²) - now in proper units (J)
    energy_per_frame = 0.5 * mass * speed**2
    
    # Total energy for the trial
    total_energy = np.sum(energy_per_frame)
    
    # Return energy per frame and total
    return energy_per_frame, total_energy


def calculate_motion_energy_jerk(marker_data, sampling_rate=300):
    """
    Calculate energy based on jerk (third derivative of position)
    High jerk values indicate more abrupt, high-energy movements
    
    Parameters:
    - marker_data: numpy array of shape [frames, 3] (x,y,z)
    - sampling_rate: Hz (frames per second)
    
    Returns:
    - jerk_energy: array of jerk-based energy values for each frame
    - total_jerk_energy: sum of jerk energy across all frames
    """
    # Safety check for data length
    if len(marker_data) < 4:  # Need at least 4 points for jerk calculation
        return np.array([0]), 0
    
    # First derivative (velocity)
    velocity = np.diff(marker_data, axis=0) * sampling_rate
    
    # Second derivative (acceleration)
    acceleration = np.diff(velocity, axis=0) * sampling_rate
    
    # Third derivative (jerk)
    jerk = np.diff(acceleration, axis=0) * sampling_rate
    
    # Jerk magnitude
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    
    # Energy metric based on jerk
    jerk_energy = jerk_magnitude**2
    
    # Total jerk energy
    total_jerk_energy = np.sum(jerk_energy)
    
    return jerk_energy, total_jerk_energy


def calculate_composite_energy(coords_data, point_keys, sampling_rate=300):
    """
    Calculate a composite energy measure across multiple markers
    
    Parameters:
    - coords_data: dictionary of marker data
    - point_keys: list of point keys to include in composite calculation
    - sampling_rate: Hz (frames per second)
    
    Returns:
    - combined_energy: array of composite energy values
    - total_energy: total composite energy
    """
    # Marker weights - can be adjusted based on biomechanical considerations
    marker_types = {k.split('_')[0] for k in point_keys}
    weight_map = {
        'head': 0.2,
        'foot': 0.15,
        'knee': 0.1,
        'hip': 0.2,
        'wrist': 0.05,
        'elbow': 0.05,
        'shoulder': 0.15,
        'floor': 0.000001,  # Less important for energy calculations
    }
    
    combined_energy = None
    min_frames = float('inf')
    
    # First pass to find shortest sequence
    for key in point_keys:
        data = coords_data[key]
        marker_positions = np.column_stack([data['x'], data['y'], data['z']])
        min_frames = min(min_frames, len(marker_positions) - 1)  # -1 for diff
    
    # Second pass to actually combine energies
    for key in point_keys:
        data = coords_data[key]
        point_type = data['point'].split('_')[0]  # Extract base type (head, foot, etc.)
        
        # Get weight for this marker type
        weight = weight_map.get(point_type, 0.1)
        
        # Combine x, y, z into array
        marker_positions = np.column_stack([data['x'], data['y'], data['z']])
        
        # Calculate energy for this marker using velocity method
        energy, _ = calculate_motion_energy_velocity(marker_positions, sampling_rate)
        
        # Trim to minimum length
        energy = energy[:min_frames]
        
        # Weight the energy by marker importance
        weighted_energy = energy * weight
        
        # Add to combined energy
        if combined_energy is None:
            combined_energy = weighted_energy
        else:
            combined_energy += weighted_energy
    
    # Handle edge case of no valid data
    if combined_energy is None or len(combined_energy) == 0:
        return np.array([0]), 0
    
    # Calculate total energy
    total_energy = np.sum(combined_energy)
    
    return combined_energy, total_energy


def analyze_energy(coords_data, method="velocity", sampling_rate=300, by_time=False, normalize=False):
    """
    Analyze motion energy for all subjects and conditions
    
    Parameters:
    - coords_data: dictionary of marker data from prepare_points_data
    - method: energy calculation method ('velocity', 'jerk', or 'composite')
    - sampling_rate: frames per second
    - by_time: whether to calculate energy per second
    - normalize: whether to normalize by trial duration
    
    Returns:
    - results: dictionary of energy analysis results
    """
    results = {}
    
    # For composite method, group by trial
    if method == "composite":
        # Group points by trial
        trial_points = {}
        for key, data in coords_data.items():
            trial_id = f"S{data['subject']}_C{data['condition']}_O{data['obstacles']}"
            if trial_id not in trial_points:
                trial_points[trial_id] = []
            trial_points[trial_id].append(key)
        
        # Calculate composite energy for each trial
        for trial_id, point_keys in trial_points.items():
            subject, condition, obstacles = trial_id.split('_')
            subject = int(subject[1:])
            condition = condition[1:]
            obstacles = int(obstacles[1:])
            
            # Calculate composite energy
            energy_profile, total_energy = calculate_composite_energy(
                coords_data, point_keys, sampling_rate
            )
            
            # Calculate trial duration
            frames = len(energy_profile)
            duration = frames / sampling_rate
            
            # Calculate energy per second if requested
            if by_time:
                # Reshape array to get energy in 1-second chunks
                seconds = int(np.ceil(duration))
                frames_per_second = sampling_rate
                
                energy_per_second = []
                for i in range(seconds):
                    start_idx = i * frames_per_second
                    end_idx = min((i + 1) * frames_per_second, len(energy_profile))
                    if start_idx >= len(energy_profile):
                        break
                    energy_per_second.append(np.sum(energy_profile[start_idx:end_idx]))
            else:
                energy_per_second = None
            
            # Normalize by duration if requested
            if normalize and duration > 0:
                normalized_energy = total_energy / duration
            else:
                normalized_energy = total_energy
            
            # Store results
            results[trial_id] = {
                'subject': subject,
                'condition': condition,
                'obstacles': obstacles,
                'total_energy': float(total_energy),
                'normalized_energy': float(normalized_energy),
                'duration': float(duration),
                'energy_per_second': energy_per_second,
                'energy_profile': energy_profile.tolist(),
                'method': method,
                'points': [coords_data[key]['point'] for key in point_keys]
            }
    else:
        # Process each marker independently for velocity and jerk methods
        for point_key, data in tqdm(coords_data.items(), desc="Calculating energy"):
            # Extract metadata
            subject = data['subject']
            condition = data['condition']
            obstacles = data['obstacles']
            point = data['point']
            
            # Combine coordinates
            positions = np.column_stack([data['x'], data['y'], data['z']])
            
            # Calculate energy based on selected method
            if method == "velocity":
                energy_profile, total_energy = calculate_motion_energy_velocity(positions, sampling_rate)
            elif method == "jerk":
                energy_profile, total_energy = calculate_motion_energy_jerk(positions, sampling_rate)
            else:
                # Should never reach here due to argparse choices
                raise ValueError(f"Unknown method: {method}")
            
            # Calculate trial duration
            frames = len(positions)
            duration = frames / sampling_rate
            
            # Calculate energy per second if requested
            if by_time and frames > 0:
                # Reshape array to get energy in 1-second chunks
                seconds = int(np.ceil(duration))
                frames_per_second = sampling_rate
                
                energy_per_second = []
                for i in range(seconds):
                    start_idx = i * frames_per_second
                    end_idx = min((i + 1) * frames_per_second, len(energy_profile))
                    if start_idx >= len(energy_profile):
                        break
                    energy_per_second.append(np.sum(energy_profile[start_idx:end_idx]))
            else:
                energy_per_second = None
            
            # Normalize by duration if requested
            if normalize and duration > 0:
                normalized_energy = total_energy / duration
            else:
                normalized_energy = total_energy
            
            # Store results
            result_key = f"{point}_S{subject}_C{condition}_O{obstacles}"
            results[result_key] = {
                'subject': subject,
                'condition': condition,
                'obstacles': obstacles,
                'point': point,
                'total_energy': float(total_energy),
                'normalized_energy': float(normalized_energy),
                'duration': float(duration),
                'energy_per_second': energy_per_second if energy_per_second is None else [float(e) for e in energy_per_second],
                'energy_profile': energy_profile.tolist(),
                'method': method
            }
    
    return results


def plot_energy_results(results, output_dir=None, method="velocity"):
    """Generate plots for energy analysis results."""
    # Create output directory if it doesn't exist
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Create different plot types
    create_total_energy_plot(results, output_dir, method)
    create_energy_time_series_plot(results, output_dir, method)
    if any('energy_per_second' in result and result['energy_per_second'] is not None for result in results.values()):
        create_energy_per_second_plot(results, output_dir, method)


def create_total_energy_plot(results, output_dir, method):
    """Create bar plot of total energy by condition and subject."""
    # Extract data for plotting
    plot_data = []
    for key, result in results.items():
        if 'point' in result:  # Skip composite results for point-specific plots
            plot_data.append({
                'subject': result['subject'],
                'condition': result['condition'],
                'obstacles': result['obstacles'],
                'point': result.get('point', 'composite'),
                'total_energy': result['total_energy'],
                'normalized_energy': result['normalized_energy']
            })
    
    if not plot_data:
        return
        
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(plot_data)
    
    # Create bar plot by condition
    plt.figure(figsize=(12, 8))
    
    # Use subject and condition for grouping
    by_condition = df.groupby(['condition', 'obstacles', 'point'])['normalized_energy'].mean().reset_index()
    by_condition['condition_label'] = by_condition.apply(
        lambda x: f"{x['condition']}{x['obstacles']}", axis=1
    )
    
    pivot_data = by_condition.pivot(index='condition_label', columns='point', values='normalized_energy')
    pivot_data.plot(kind='bar', ax=plt.gca())
    
    plt.title(f'Average Energy by Condition and Marker ({method.capitalize()} Method)')
    plt.ylabel('Normalized Energy (J/s)')
    plt.xlabel('Condition')
    plt.legend(title='Body Point')
    plt.tight_layout()
    
    if output_dir:
        output_file = Path(output_dir) / f"energy_by_condition_{method}.png"
        plt.savefig(output_file)
        logging.info(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    # Create bar plot by subject
    plt.figure(figsize=(12, 8))
    
    by_subject = df.groupby(['subject', 'point'])['normalized_energy'].mean().reset_index()
    pivot_data = by_subject.pivot(index='subject', columns='point', values='normalized_energy')
    pivot_data.plot(kind='bar', ax=plt.gca())
    
    plt.title(f'Average Energy by Subject and Marker ({method.capitalize()} Method)')
    plt.ylabel('Normalized Energy (J/s)')
    plt.xlabel('Subject ID')
    plt.legend(title='Body Point')
    plt.tight_layout()
    
    if output_dir:
        output_file = Path(output_dir) / f"energy_by_subject_{method}.png"
        plt.savefig(output_file)
        logging.info(f"Saved plot to {output_file}")
    else:
        plt.show()


def create_energy_time_series_plot(results, output_dir, method):
    """Create time series plots of energy profiles."""
    # Group results by subject
    subjects = set(result['subject'] for result in results.values())
    
    for subject in subjects:
        plt.figure(figsize=(14, 10))
        
        # Filter results for this subject
        subject_results = {k: v for k, v in results.items() if v['subject'] == subject}
        
        # Plot each point's energy profile
        for key, result in subject_results.items():
            if 'energy_profile' not in result or not result['energy_profile']:
                continue
                
            # Get condition label
            if 'point' in result:
                label = f"{result['point']} - C{result['condition']}O{result['obstacles']}"
            else:
                label = f"Composite - C{result['condition']}O{result['obstacles']}"
            
            # Plot energy profile
            plt.plot(result['energy_profile'], label=label)
        
        plt.title(f'Energy Profile for Subject {subject} ({method.capitalize()} Method)')
        plt.xlabel('Time (frame number)')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / f"energy_profile_S{subject}_{method}.png"
            plt.savefig(output_file)
            logging.info(f"Saved plot to {output_file}")
        else:
            plt.show()


def create_energy_per_second_plot(results, output_dir, method):
    """Create plots of energy per second."""
    # Group results by subject
    subjects = set(result['subject'] for result in results.values())
    
    for subject in subjects:
        plt.figure(figsize=(14, 10))
        
        # Filter results for this subject
        subject_results = {k: v for k, v in results.items() if v['subject'] == subject}
        
        # Plot each point's energy per second
        for key, result in subject_results.items():
            if 'energy_per_second' not in result or not result['energy_per_second']:
                continue
                
            # Get condition label
            if 'point' in result:
                label = f"{result['point']} - C{result['condition']}O{result['obstacles']}"
            else:
                label = f"Composite - C{result['condition']}O{result['obstacles']}"
            
            # Plot energy per second
            plt.plot(result['energy_per_second'], marker='o', label=label)
        
        plt.title(f'Energy per Second for Subject {subject} ({method.capitalize()} Method)')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy per Second (J/s)')
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            output_file = Path(output_dir) / f"energy_per_second_S{subject}_{method}.png"
            plt.savefig(output_file)
            logging.info(f"Saved plot to {output_file}")
        else:
            plt.show()


def main():
    args = parse_args()
    
    # Load data
    data = load_data(
        args.data, 
        subjects=args.subjects, 
        conditions=args.conditions, 
        obstacles=args.obstacles
    )
    
    # Prepare coordinate data
    coords_data = prepare_points_data(data, args.points)
    
    if not coords_data:
        logging.error("No valid points found in the data")
        return
    
    # Analyze motion energy
    results = analyze_energy(
        coords_data,
        method=args.method,
        sampling_rate=args.sampling_rate,
        by_time=args.by_time,
        normalize=args.normalize
    )
    
    # Save results to file
    with open(args.output, 'w+b') as f:
        pickle.dump(results, f)
    logging.info(f"Saved results to {args.output}")
    
    # Generate plots if requested
    if args.plot:
        plot_energy_results(
            results,
            output_dir=args.plot_dir,
            method=args.method
        )
    
    # Print summary
    print("\nEnergy Analysis Summary:")
    print(f"Method: {args.method}")
    print(f"Total trials analyzed: {len(results)}")
    
    # Calculate average energy by condition
    if args.method == "composite":
        # For composite method, data is already grouped by trial
        condition_energy = {}
        for key, result in results.items():
            condition = f"{result['condition']}{result['obstacles']}"
            if condition not in condition_energy:
                condition_energy[condition] = []
            condition_energy[condition].append(result['normalized_energy'])
    else:
        # For point-based methods, group by condition and average across points
        point_types = set(data['point'] for data in results.values() if 'point' in data)
        for point in point_types:
            condition_energy = {}
            for key, result in results.items():
                if 'point' in result and result['point'] == point:
                    condition = f"{result['condition']}{result['obstacles']}"
                    if condition not in condition_energy:
                        condition_energy[condition] = []
                    condition_energy[condition].append(result['normalized_energy'])
            
            if condition_energy:
                print(f"\nPoint: {point}")
                for condition, energies in sorted(condition_energy.items()):
                    avg_energy = np.mean(energies)
                    print(f"  Condition {condition}: {avg_energy:.2f} average energy")
    
    # Also print the composite summary if available
    if args.method == "composite":
        print("\nComposite Energy:")
        for condition, energies in sorted(condition_energy.items()):
            avg_energy = np.mean(energies)
            print(f"  Condition {condition}: {avg_energy:.2f} average energy")
    
    print(f"\nDetailed results saved to {args.output}")
    if args.plot:
        print(f"Plots saved to {args.plot_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
