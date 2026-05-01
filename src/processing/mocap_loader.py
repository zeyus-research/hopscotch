"""
Motion capture data loader with support for TSV and C3D files.
Handles subject/condition grouping and data preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import c3d
import logging
from tqdm import tqdm

class MocapDataLoader:
    """
    Load and preprocess motion capture data from TSV or C3D files.
    """

    def __init__(self, marker_labels_file: Optional[Path] = None):
        """
        Initialize the MocapDataLoader.
        
        Args:
            marker_labels_file: Path to file containing marker labels
        """
        self.marker_labels: List[str] = self._load_marker_labels(marker_labels_file)
        self.sampling_rate: float = 300.0
        self.total_removed_rows: int = 0
        
    def _load_marker_labels(self, labels_file: Optional[Path]) -> List[str]:
        """Load marker labels from file."""
        if labels_file and labels_file.exists():
            with open(labels_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []
    
    def load_c3d_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single C3D file.
        
        Args:
            file_path: Path to C3D file
            
        Returns:
            DataFrame with marker data
        """
        try:
            with open(file_path, 'rb') as f:
                reader = c3d.Reader(f)
                
                # Get sampling rate
                self.sampling_rate = reader.header.frame_rate
                
                # Extract marker data
                frames = []
                for i, (points, analog) in enumerate(reader.read_frames()):
                    frame_data = {'Frame': i, 'Time': i / self.sampling_rate}
                    
                    # Add marker positions
                    for j, point in enumerate(points):
                        if j < len(self.marker_labels):
                            marker_name = self.marker_labels[j]
                            frame_data[f'{marker_name}.X'] = point[0]
                            frame_data[f'{marker_name}.Y'] = point[1] 
                            frame_data[f'{marker_name}.Z'] = point[2]
                    
                    frames.append(frame_data)
                
                return pd.DataFrame(frames)
                
        except Exception as e:
            logging.error(f"Error loading C3D file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_tsv_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single TSV file.
        
        Args:
            file_path: Path to TSV file
            
        Returns:
            DataFrame with marker data
        """
        try:
            # Preprocess file header (similar to existing loader)
            # with open(file_path, "r+") as f:
            #     lines = f.readlines()
                
            #     lines[0].strip().replace(" ", ".") + "\n"
                
            #     # Now we need to carefully handle a cleanup issue
            #     # in the TSV files, the frame number is sequential
            #     # but most files have some trailing errors e.g.
            #     # 1234
            #     # 1235
            #     # 1236
            #     # \t0.3
            #     # 1235
            #     # 1236
            #     # in the above case, the frame (empty)\t0.3 and all below should be removed
            #     # these are repeat frames and an odd row without a frame number
            #     last_frame_index = 0
            #     end_row_index = 1
            #     for i in range(1, len(lines)):
            #         first_col = lines[i].split("\t")[0].strip()
            #         if first_col.isdigit():
            #             if int(first_col) > last_frame_index:
            #                 last_frame_index = int(first_col)
            #                 end_row_index = i + 1
            #         else:
            #             break
            #     if end_row_index < len(lines):
            #         logging.warning(f"Trimming {len(lines) - end_row_index} rows from {file_path} due to non-sequential frame numbers")
            #     lines = lines[:end_row_index]
            #     # go to start of file
            #     f.seek(0)
            #     # truncate file to remove old content
            #     f.buffer.truncate(0)
            #     # write cleaned lines back to file
            #     f.writelines(lines)
            
            # Read the TSV file
            df = pd.read_csv(file_path, sep="\t", low_memory=False, skip_blank_lines=True)
            
            # Remove extra columns if present
            if "X" in df.columns:
                df.drop(columns=["X"], inplace=True)
            
            # count total NaN values before cleaning
            total_nans_before = df.isnull().sum().sum()
            if total_nans_before > 10_000:
                logging.warning(f"File {file_path} has {total_nans_before} NaN values - EXCLUDING")
                return pd.DataFrame()

            # Clean data - remove rows where all coordinates are NaN
            df = self.clean_mocap_data(df, file_path.name)
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading TSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def clean_mocap_data(self, df: pd.DataFrame, filename: str|None) -> pd.DataFrame:
        """
        Clean motion capture data by handling missing values.
        
        Args:
            df: Raw DataFrame with potential NaN values
            
        Returns:
            Cleaned DataFrame
        """
        # Check for rows with excessive NaN values
        marker_cols = [col for col in df.columns if any(coord in col for coord in ['.X', '.Y', '.Z'])]
        
        if len(marker_cols) > 0:
            # Count NaN values per row for marker columns
            # nan_per_row = df[marker_cols].isnull().sum(axis=1)
            
            # # Remove rows where more than 50% of marker data is missing
            # threshold = len(marker_cols) * 0.5
            # clean_df = df[nan_per_row <= threshold].copy()
            
            # removed_rows = len(df) - len(clean_df)
            # if removed_rows > 0:
            #     self.total_removed_rows += removed_rows
            #     logging.info(f"Removed {removed_rows} rows with excessive missing data from {filename or 'data'}")
            
            # For remaining NaN values, use forward fill then backward fill
            df[marker_cols] = df[marker_cols].ffill().bfill()
            
            return df
        
        return df
    
    def extract_subject_condition(self, filename: str) -> Tuple[str, str, str]:
        """
        Extract subject, condition, and obstacles info from filename.
        Format: {SUBJECTID}_{CONDITION}{OBSTACLES}.tsv
        - CONDITION: h (extrinsic/speed), s (intrinsic/fun), k (control)
        - OBSTACLES: 0 (no obstacles), 1 (with obstacles)
        
        Args:
            filename: File name to parse
            
        Returns:
            Tuple of (subject, condition, obstacles)
        """
        stem = Path(filename).stem
        
        # Parse format like "23_h0.tsv" -> subject=23, condition=h, obstacles=0
        parts = stem.split("_")
        if len(parts) >= 2:
            subject = parts[0]  # e.g., "23"
            condition_info = parts[1]  # e.g., "h0"
            if len(condition_info) >= 2:
                condition = condition_info[0]  # e.g., "h"
                obstacles = condition_info[1:]  # e.g., "0"
            else:
                condition = condition_info
                obstacles = "0"
        else:
            subject = stem
            condition = "unknown"
            obstacles = "0"
        
        return subject, condition, obstacles
    
    def load_dataset(self, 
                    data_path: Path,
                    file_pattern: str = "*.tsv",
                    subject_config: Optional[Dict] = None,
                    cache_file: Optional[str] = "mocap_data.feather") -> pd.DataFrame:
        """
        Load entire dataset from directory.
        
        Args:
            data_path: Path to directory containing data files
            file_pattern: File pattern to match (e.g., "*.tsv", "*.c3d")
            subject_config: Optional dictionary mapping files to subjects/conditions
            cache_file: Optional cache file name
            
        Returns:
            Combined DataFrame with all data
        """
        # Try loading from cache first
        if cache_file:
            cache_path = data_path / cache_file
            if cache_path.exists():
                try:
                    dataset = pd.read_feather(cache_path)
                    logging.info("Loaded cached mocap data.")
                    return dataset
                except Exception as e:
                    logging.warning(f"Could not load cache: {e}")
            else:
                logging.warning(f"Cache file {cache_path} not found, loading from raw data.")
        
        # Load files
        files = list(data_path.glob(file_pattern))
        files.sort()
        
        dataset_frames = []
        
        for file_path in tqdm(files, desc="Loading mocap files", unit=" files"):
            # Load file based on extension
            if file_path.suffix.lower() == '.c3d':
                df = self.load_c3d_file(file_path)
            else:
                df = self.load_tsv_file(file_path)
            
            if df.empty:
                continue
            
            # Add metadata
            filename = file_path.name
            df['filename'] = filename
            
            # Extract subject/condition info
            if subject_config and filename in subject_config:
                config = subject_config[filename]
                df['subject'] = config.get('subject', 'unknown')
                df['condition'] = config.get('condition', 'unknown')
                df['obstacles'] = config.get('obstacles', '0')
            else:
                subject, condition, obstacles = self.extract_subject_condition(filename)
                df['subject'] = subject
                df['condition'] = condition
                df['obstacles'] = obstacles
            
            dataset_frames.append(df)
        
        if not dataset_frames:
            logging.warning("No valid data files found.")
            return pd.DataFrame()
        
        # Combine all data
        dataset = pd.concat(dataset_frames, ignore_index=True)

        # # report the number of NaN values per subject, trial, obstacles
        # nan_report = dataset.groupby(['subject', 'condition', 'obstacles'])[[col for col in dataset.columns if any(coord in col for coord in ['.X', '.Y', '.Z'])]].apply(lambda x: x.isnull().sum().sum())
        # # order by number of NaN values
        # nan_report = nan_report.sort_values(ascending=False)
        # logging.info("NaN values report (sorted by total NaNs):")
        # logging.info(nan_report)
        # Save cache if requested
        if cache_file:
            cache_path = data_path / cache_file
            try:
                dataset.to_feather(cache_path)
                logging.info(f"Saved cached data to {cache_path}")
            except Exception as e:
                logging.warning(f"Could not save cache: {e}")
        
        # Show total removed rows info
        if self.total_removed_rows > 0:
            logging.info(f"Total removed rows across dataset: {self.total_removed_rows}")
            # Include a time based on sampling rate
            if self.sampling_rate:
                total_time_removed = self.total_removed_rows / self.sampling_rate
                logging.info(f"Total time removed due to missing data: {total_time_removed:.2f} seconds")

        return dataset
    
    def get_marker_positions(self, df: pd.DataFrame, marker_name: str) -> np.ndarray:
        """
        Extract 3D positions for specific marker.
        
        Args:
            df: DataFrame with marker data
            marker_name: Name of marker
            
        Returns:
            Array of shape (n_frames, 3) with XYZ coordinates
        """
        try:
            x = df[f'{marker_name}.X'].values
            y = df[f'{marker_name}.Y'].values  
            z = df[f'{marker_name}.Z'].values
            return np.column_stack([x, y, z])
        except KeyError:
            logging.error(f"Marker {marker_name} not found in data")
            return np.array([])
    
    def get_available_markers(self, df: pd.DataFrame) -> List[str]:
        """Get list of available markers in dataset."""
        markers = []
        for col in df.columns:
            if col.endswith('.X'):
                markers.append(col[:-2])
        return markers
