from processing import mocap_loader
import sys
from pathlib import Path
import pandas as pd
import yaml
import logging
from dtaidistance import dtw_ndim
import numpy as np
from tqdm import tqdm
from itertools import combinations
import multiprocessing as mp
import threading

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
    # Save config for reference
    config_file = output_dir / 'analysis_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    cache_file = config.get('cache_file', 'hopscotch_data_no_floor2.feather')
        
    raw_data = loader.load_dataset(
        data_path=data_path,
        file_pattern = "*.tsv",
        cache_file=cache_file
    )
    
    logging.info(f"Loaded {len(raw_data)} records from {len(raw_data.groupby('filename'))} files")

    return raw_data

def convert_to_ndarray(data: pd.DataFrame, markers: list) -> np.ndarray:
    """Convert DataFrame to ndarray with shape (timesteps, n_markers, 3)"""
    n_frames = len(data)
    n_markers = len(markers)
    
    # Initialize array
    result = np.zeros((n_frames, n_markers, 3), dtype=np.double)
    
    # Fill array with marker positions
    for i, marker in enumerate(markers):
        if f'{marker}.X' in data.columns:
            result[:, i, 0] = data[f'{marker}.X'].values  # X coordinate
            result[:, i, 1] = data[f'{marker}.Y'].values  # Y coordinate  
            result[:, i, 2] = data[f'{marker}.Z'].values  # Z coordinate
        else:
            logging.warning(f"Marker {marker} not found in data")
    
    return result

def compute_dtw_distance(data1: pd.DataFrame, data2: pd.DataFrame, marker_labels: list) -> tuple[float, np.ndarray]:
    """Compute DTW distance between two datasets"""
    array_1 = convert_to_ndarray(data1, marker_labels)
    array_2 = convert_to_ndarray(data2, marker_labels)
    
    if array_1.size > 0 and array_2.size > 0:
        distance, distance_matrix = dtw_ndim.warping_paths(array_1, array_2)
        return distance, distance_matrix
    else:
        logging.warning("One or both arrays are empty")
        return np.nan

# Thread lock for file writing
_file_lock = threading.Lock()

def append_result_to_tsv(result_row: dict, output_file: Path):
    """Thread-safe append of a single result to TSV file"""
    with _file_lock:
        df = pd.DataFrame([result_row])
        
        # Check if file exists and has content
        if output_file.exists() and output_file.stat().st_size > 0:
            df.to_csv(output_file, mode='a', header=False, sep='\t', index=False)
        else:
            df.to_csv(output_file, mode='w', header=True, sep='\t', index=False)

def save_distance_matrix(distance_matrix: np.ndarray, unique_key: str, index: int) -> None:
    """Thread-safe save DTW distance matrix to a file"""
    with _file_lock:
        output_dir = Path("analysis")
        filename = output_dir / f'dtw_distance_matrix_{unique_key}_{index}.npy'
        
        # Ensure directory exists
        output_dir.mkdir(exist_ok=True)
        
        np.save(filename, distance_matrix)

def process_comparison(comparison_data: tuple) -> dict:
    """Worker function to process a single DTW comparison"""
    (data1, data2, marker_labels, comparison_key, result_index, dtw_type, 
     subject_1, subject_2, condition_1, condition_2, obstacles_1, obstacles_2) = comparison_data
    
    if len(data1) > 0 and len(data2) > 0:
        distance, distance_matrix = compute_dtw_distance(data1, data2, marker_labels)
        
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
    
    return None

def run_dtw_dataset_pairs(raw_data: pd.DataFrame, n_cores: int = None) -> None:
    """
    Compute comprehensive DTW analysis for all subject/condition/obstacle combinations.
    Uses multiprocessing to parallelize DTW computations.
    
    Within-subject comparisons:
    - For each subject: h0-k0, h0-s0, k0-s0, h1-k1, h1-s1, k1-s1, h0-h1, k0-k1, s0-s1
    
    Between-subject comparisons:
    - For each subject pair: same condition+obstacle combinations
    
    Args:
        raw_data: DataFrame with motion capture data
        n_cores: Number of cores to use. Defaults to (total cores - 1)
    """
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)
    
    logging.info(f"Using {n_cores} cores for parallel processing")
    
    marker_labels_file = Path('marker_labels_no_floor.txt')
    marker_labels = marker_labels_file.read_text().splitlines()
    marker_labels = [label.strip() for label in marker_labels if label.strip()]
    
    output_dir = Path("analysis")
    date = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'dtw_results_{date}.tsv'
    
    subjects = sorted(raw_data['subject'].unique())
    conditions = sorted(raw_data['condition'].unique())  # ['h', 'k', 's']
    obstacles = sorted(raw_data['obstacles'].unique())   # ['0', '1']
    
    logging.info(f"Processing {len(subjects)} subjects, {len(conditions)} conditions, {len(obstacles)} obstacle levels")
    
    # Prepare all comparison tasks
    comparison_tasks = []
    result_index = 0
    
    # Within-subject comparisons - generate tasks
    for subject in subjects:
        subject_data = raw_data[raw_data['subject'] == subject]
        
        # Condition comparisons within each obstacle level
        for obstacle in obstacles:
            obstacle_data = subject_data[subject_data['obstacles'] == obstacle]
            
            for cond1, cond2 in combinations(conditions, 2):
                data1 = obstacle_data[obstacle_data['condition'] == cond1]
                data2 = obstacle_data[obstacle_data['condition'] == cond2]
                
                comparison_key = f"s{subject}-s{subject}_c{cond1}-c{cond2}_o{obstacle}-o{obstacle}"
                
                task = (data1, data2, marker_labels, comparison_key, result_index, 'within',
                       subject, subject, cond1, cond2, obstacle, obstacle)
                comparison_tasks.append(task)
                result_index += 1
        
        # Obstacle comparisons within each condition
        for condition in conditions:
            condition_data = subject_data[subject_data['condition'] == condition]
            
            for obs1, obs2 in combinations(obstacles, 2):
                data1 = condition_data[condition_data['obstacles'] == obs1]
                data2 = condition_data[condition_data['obstacles'] == obs2]
                
                comparison_key = f"s{subject}-s{subject}_c{condition}-c{condition}_o{obs1}-o{obs2}"
                
                task = (data1, data2, marker_labels, comparison_key, result_index, 'within',
                       subject, subject, condition, condition, obs1, obs2)
                comparison_tasks.append(task)
                result_index += 1
    
    # Between-subject comparisons - generate tasks
    for subj1, subj2 in combinations(subjects, 2):
        for condition in conditions:
            for obstacle in obstacles:
                data1 = raw_data[(raw_data['subject'] == subj1) & 
                               (raw_data['condition'] == condition) & 
                               (raw_data['obstacles'] == obstacle)]
                data2 = raw_data[(raw_data['subject'] == subj2) & 
                               (raw_data['condition'] == condition) & 
                               (raw_data['obstacles'] == obstacle)]
                
                comparison_key = f"s{subj1}-s{subj2}_c{condition}-c{condition}_o{obstacle}-o{obstacle}"
                
                task = (data1, data2, marker_labels, comparison_key, result_index, 'between',
                       subj1, subj2, condition, condition, obstacle, obstacle)
                comparison_tasks.append(task)
                result_index += 1
    
    logging.info(f"Generated {len(comparison_tasks)} DTW comparison tasks")
    
    # Process tasks in parallel
    with mp.Pool(processes=n_cores) as pool:
        # Use tqdm to track progress
        results = []
        with tqdm(total=len(comparison_tasks), desc="Computing DTW distances") as pbar:
            for result in pool.imap(process_comparison, comparison_tasks):
                if result is not None:
                    # Save results immediately as they complete
                    append_result_to_tsv(result['result_row'], output_file)
                    save_distance_matrix(result['distance_matrix'], 
                                       result['comparison_key'], 
                                       result['result_index'])
                pbar.update(1)
    
    logging.info(f"DTW analysis completed. Results saved to {output_file}")
    logging.info(f"Total comparisons processed: {len(comparison_tasks)}")
    

def main():
    log_level = 'INFO'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    raw_data = load_raw_data()
    logging.info("Columns in raw_data:" + str(raw_data.columns.tolist()))
    logging.info("conditions:" + str(raw_data['condition'].unique()))
    logging.info("subjects:" + str(raw_data['subject'].unique()))
    logging.info("obstacles:"+  str(raw_data['obstacles'].unique()))
    
    # Get number of available cores
    total_cores = mp.cpu_count()
    n_cores = max(1, total_cores - 1)  # Use all cores except 1
    logging.info(f"System has {total_cores} cores, using {n_cores} for processing")
    
    # Compute comprehensive DTW analysis with multiprocessing
    run_dtw_dataset_pairs(raw_data, n_cores=n_cores)


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.set_start_method('spawn', force=True)
    main()
