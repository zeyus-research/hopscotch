"""
DTW analysis using joint angles instead of raw marker positions.

This provides pose-invariant comparisons - two children doing the same
movement but in different global positions/orientations will have 
similar DTW distances.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from dtaidistance import dtw_ndim
from tqdm import tqdm
from itertools import combinations
import multiprocessing as mp
import threading

# Add src to path
sys.path.insert(0, 'src')
from processing import mocap_loader

# Import joint angle calculator
sys.path.insert(0, '.')
from joint_angles import calculate_all_joint_angles, convert_angles_to_ndarray


def load_raw_data() -> pd.DataFrame:
    """Load the motion capture data."""
    data_path = Path("data")
    output_dir = Path("hopscotch_results")
    output_dir.mkdir(exist_ok=True)
    
    config = {
        'marker_labels_file': Path('marker_labels_no_floor.txt'),
        'sampling_rate': 300.0,
        'default_body_mass': 25.0,
        'cache_file': 'hopscotch_data_no_floor2.feather',
        'log_level': 'INFO',
    }
    
    loader = mocap_loader.MocapDataLoader(config.get('marker_labels_file'))
    
    # Save config for reference
    config_file = output_dir / 'analysis_config_joint_angles.yaml'
    with open(config_file, 'w') as f:
        yaml.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, 
                  f, default_flow_style=False)
    
    raw_data = loader.load_dataset(
        data_path=data_path,
        file_pattern="*.tsv",
        cache_file=config.get('cache_file')
    )
    
    logging.info(f"Loaded {len(raw_data)} records from {len(raw_data.groupby('filename'))} files")
    return raw_data


def compute_joint_angles_for_trial(trial_data: pd.DataFrame) -> np.ndarray:
    """
    Compute joint angles for a single trial.
    
    Args:
        trial_data: DataFrame with marker data for one trial
        
    Returns:
        Array of shape (n_frames, n_angles) with joint angles
    """
    angles_df = calculate_all_joint_angles(trial_data)
    return convert_angles_to_ndarray(angles_df)


def compute_dtw_distance_joint_angles(data1: pd.DataFrame, data2: pd.DataFrame) -> tuple[float, np.ndarray]:
    """
    Compute DTW distance between two trials using joint angles.
    
    Args:
        data1, data2: DataFrames with marker data
        
    Returns:
        Tuple of (distance, distance_matrix)
    """
    # Convert to joint angles
    angles1 = compute_joint_angles_for_trial(data1)
    angles2 = compute_joint_angles_for_trial(data2)
    
    if angles1.size > 0 and angles2.size > 0:
        # Ensure float64 dtype for dtaidistance
        angles1 = angles1.astype(np.float64)
        angles2 = angles2.astype(np.float64)
        
        distance, paths = dtw_ndim.warping_paths(angles1, angles2)
        return distance, paths
    else:
        logging.warning("One or both angle arrays are empty")
        return np.nan, np.array([])


# Thread lock for file writing
_file_lock = threading.Lock()

def append_result_to_tsv(result_row: dict, output_file: Path):
    """Thread-safe append of a single result to TSV file."""
    with _file_lock:
        df = pd.DataFrame([result_row])
        if output_file.exists() and output_file.stat().st_size > 0:
            df.to_csv(output_file, mode='a', header=False, sep='\t', index=False)
        else:
            df.to_csv(output_file, mode='w', header=True, sep='\t', index=False)


def save_distance_matrix(distance_matrix: np.ndarray, unique_key: str, index: int, 
                         output_dir: Path) -> None:
    """Thread-safe save DTW distance matrix to a file."""
    with _file_lock:
        filename = output_dir / f'dtw_joint_angles_matrix_{unique_key}_{index}.npy'
        np.save(filename, distance_matrix)


def process_comparison(comparison_data: tuple) -> dict:
    """Worker function to process a single DTW comparison using joint angles."""
    (data1, data2, comparison_key, result_index, dtw_type,
     subject_1, subject_2, condition_1, condition_2, obstacles_1, obstacles_2) = comparison_data
    
    if len(data1) > 0 and len(data2) > 0:
        try:
            distance, distance_matrix = compute_dtw_distance_joint_angles(data1, data2)
            
            result_row = {
                'index': result_index,
                'dtw_type': dtw_type,
                'subject_1': subject_1,
                'subject_2': subject_2,
                'condition_1': condition_1,
                'condition_2': condition_2,
                'obstacles_1': obstacles_1,
                'obstacles_2': obstacles_2,
                'dtw_distance': distance,
                'distance_matrix_key': comparison_key,
            }
            
            return {
                'result_row': result_row,
                'distance_matrix': distance_matrix,
                'comparison_key': comparison_key,
                'result_index': result_index
            }
        except Exception as e:
            logging.error(f"Error processing {comparison_key}: {e}")
            return None
    
    return None


def run_dtw_joint_angles_analysis(raw_data: pd.DataFrame, n_cores: int = None,
                                   save_matrices: bool = False) -> None:
    """
    Compute DTW analysis using joint angles for all subject/condition/obstacle combinations.
    
    Args:
        raw_data: DataFrame with motion capture data
        n_cores: Number of cores to use. Defaults to (total cores - 1)
        save_matrices: Whether to save full distance matrices (large files!)
    """
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
    
    logging.info(f"Using {n_cores} cores for parallel processing")
    logging.info("Computing DTW distances using JOINT ANGLES (pose-invariant)")
    
    output_dir = Path("hopscotch_results")
    output_dir.mkdir(exist_ok=True)
    
    date = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'dtw_joint_angles_results_{date}.tsv'
    
    subjects = sorted(raw_data['subject'].unique())
    conditions = sorted(raw_data['condition'].unique())
    obstacles = sorted(raw_data['obstacles'].unique())
    
    logging.info(f"Processing {len(subjects)} subjects, {len(conditions)} conditions, "
                 f"{len(obstacles)} obstacle levels")
    
    # Prepare all comparison tasks
    comparison_tasks = []
    result_index = 0
    
    # Within-subject comparisons
    logging.info("Generating within-subject comparison tasks...")
    for subject in subjects:
        subject_data = raw_data[raw_data['subject'] == subject]
        
        # Condition comparisons within each obstacle level
        for obstacle in obstacles:
            obstacle_data = subject_data[subject_data['obstacles'] == obstacle]
            
            for cond1, cond2 in combinations(conditions, 2):
                data1 = obstacle_data[obstacle_data['condition'] == cond1].copy()
                data2 = obstacle_data[obstacle_data['condition'] == cond2].copy()
                
                comparison_key = f"s{subject}-s{subject}_c{cond1}-c{cond2}_o{obstacle}-o{obstacle}"
                
                task = (data1, data2, comparison_key, result_index, 'within',
                        subject, subject, cond1, cond2, obstacle, obstacle)
                comparison_tasks.append(task)
                result_index += 1
        
        # Obstacle comparisons within each condition
        for condition in conditions:
            condition_data = subject_data[subject_data['condition'] == condition]
            
            for obs1, obs2 in combinations(obstacles, 2):
                data1 = condition_data[condition_data['obstacles'] == obs1].copy()
                data2 = condition_data[condition_data['obstacles'] == obs2].copy()
                
                comparison_key = f"s{subject}-s{subject}_c{condition}-c{condition}_o{obs1}-o{obs2}"
                
                task = (data1, data2, comparison_key, result_index, 'within',
                        subject, subject, condition, condition, obs1, obs2)
                comparison_tasks.append(task)
                result_index += 1
    
    # Between-subject comparisons
    logging.info("Generating between-subject comparison tasks...")
    for subj1, subj2 in combinations(subjects, 2):
        for condition in conditions:
            for obstacle in obstacles:
                data1 = raw_data[(raw_data['subject'] == subj1) &
                                 (raw_data['condition'] == condition) &
                                 (raw_data['obstacles'] == obstacle)].copy()
                data2 = raw_data[(raw_data['subject'] == subj2) &
                                 (raw_data['condition'] == condition) &
                                 (raw_data['obstacles'] == obstacle)].copy()
                
                comparison_key = f"s{subj1}-s{subj2}_c{condition}-c{condition}_o{obstacle}-o{obstacle}"
                
                task = (data1, data2, comparison_key, result_index, 'between',
                        subj1, subj2, condition, condition, obstacle, obstacle)
                comparison_tasks.append(task)
                result_index += 1
    
    logging.info(f"Generated {len(comparison_tasks)} DTW comparison tasks")
    
    # Process tasks in parallel
    with mp.Pool(processes=n_cores) as pool:
        with tqdm(total=len(comparison_tasks), desc="Computing DTW (joint angles)") as pbar:
            for result in pool.imap(process_comparison, comparison_tasks):
                if result is not None:
                    append_result_to_tsv(result['result_row'], output_file)
                    if save_matrices:
                        save_distance_matrix(result['distance_matrix'],
                                             result['comparison_key'],
                                             result['result_index'],
                                             output_dir)
                pbar.update(1)
    
    logging.info(f"DTW joint angle analysis completed. Results saved to {output_file}")
    logging.info(f"Total comparisons processed: {len(comparison_tasks)}")


def compare_single_pair(raw_data: pd.DataFrame, 
                        subject1: str, condition1: str, obstacles1: str,
                        subject2: str, condition2: str, obstacles2: str) -> dict:
    """
    Compute DTW for a single pair of trials (useful for testing).
    
    Returns dict with distance and metadata.
    """
    data1 = raw_data[(raw_data['subject'] == subject1) &
                     (raw_data['condition'] == condition1) &
                     (raw_data['obstacles'] == obstacles1)]
    
    data2 = raw_data[(raw_data['subject'] == subject2) &
                     (raw_data['condition'] == condition2) &
                     (raw_data['obstacles'] == obstacles2)]
    
    if len(data1) == 0 or len(data2) == 0:
        logging.warning(f"No data found for one or both trials")
        return None
    
    distance, matrix = compute_dtw_distance_joint_angles(data1, data2)
    
    return {
        'subject_1': subject1, 'condition_1': condition1, 'obstacles_1': obstacles1,
        'subject_2': subject2, 'condition_2': condition2, 'obstacles_2': obstacles2,
        'dtw_distance': distance,
        'frames_1': len(data1),
        'frames_2': len(data2)
    }


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    raw_data = load_raw_data()
    
    logging.info("Columns in raw_data: " + str(raw_data.columns.tolist()))
    logging.info("Conditions: " + str(raw_data['condition'].unique()))
    logging.info("Subjects: " + str(raw_data['subject'].unique()))
    logging.info("Obstacles: " + str(raw_data['obstacles'].unique()))
    
    # Get number of available cores
    total_cores = mp.cpu_count()
    n_cores = max(1, total_cores - 1)
    logging.info(f"System has {total_cores} cores, using {n_cores} for processing")
    
    # Run the analysis
    run_dtw_joint_angles_analysis(raw_data, n_cores=n_cores, save_matrices=False)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
