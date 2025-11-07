import os
import numpy as np
import pandas as pd
import ezc3d

def c3d_to_trc(c3d_file, output_trc=None, coordinate_transform='opensim'):
    """
    Convert C3D file to OpenSim TRC format
    
    Parameters:
    -----------
    c3d_file : str
        Path to the C3D file
    output_trc : str, optional
        Path for output TRC file. If None, creates file with same name but .trc extension
    coordinate_transform : str, optional
        Type of coordinate transform to apply:
        - 'opensim': Maps Qualisys to OpenSim coordinate system [X→X, Z→Y, Y→-Z]
        - 'none': Keep original coordinate system
    """
    # Set output filename if not provided
    if output_trc is None:
        output_trc = os.path.splitext(c3d_file)[0] + '.trc'
    
    # Load C3D file
    c3d = ezc3d.c3d(c3d_file)
    
    # Extract point data and metadata
    point_data = c3d['data']['points']
    labels = c3d['parameters']['POINT']['LABELS']['value']
    frame_rate = float(c3d['parameters']['POINT']['RATE']['value'][0] 
                      if isinstance(c3d['parameters']['POINT']['RATE']['value'], np.ndarray) 
                      else c3d['parameters']['POINT']['RATE']['value'])
    
    # Get dimensions
    n_frames = point_data.shape[2]
    n_markers = len(labels)
    
    # Create time vector (fix for scalar conversion)
    start_time = 0.0
    time_step = 1.0 / frame_rate
    time_vector = np.linspace(start_time, start_time + (n_frames - 1) * time_step, n_frames)
    
    # Create data frame for TRC
    column_labels = ['Frame#', 'Time']
    for label in labels:
        column_labels.extend([f'{label}_X', f'{label}_Y', f'{label}_Z'])
    
    # Initialize the data array
    data = np.zeros((n_frames, 2 + n_markers * 3))
    data[:, 0] = np.arange(1, n_frames + 1)  # Frame numbers (1-based)
    data[:, 1] = time_vector  # Time column
    
    # Fill in the marker data with coordinate transformation
    # Qualisys: X=forward, Y=lateral(right), Z=vertical(up)
    # OpenSim:  X=forward, Y=vertical(up), Z=lateral(right)
    for i, label in enumerate(labels):
        if coordinate_transform.lower() == 'opensim':
            # Convert from Qualisys to OpenSim coordinate system and from mm to m
            # X: keep the same (forward)
            data[:, 2 + i*3] = point_data[0, i, :] / 1000.0  # X → X
            # Y: use Z from C3D (vertical)
            data[:, 3 + i*3] = point_data[2, i, :] / 1000.0  # Z → Y
            # Z: use -Y from C3D (lateral, with sign flip)
            data[:, 4 + i*3] = -point_data[1, i, :] / 1000.0  # -Y → Z
        else:
            # No transformation, just convert from mm to m
            data[:, 2 + i*3] = point_data[0, i, :] / 1000.0  # X
            data[:, 3 + i*3] = point_data[1, i, :] / 1000.0  # Y
            data[:, 4 + i*3] = point_data[2, i, :] / 1000.0  # Z
    
    # Create dataframe
    df = pd.DataFrame(data, columns=column_labels)
    
    # Write TRC file header
    with open(output_trc, 'w') as f:
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(output_trc))
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            frame_rate, frame_rate, n_frames, n_markers, 
            "m", frame_rate, 1, n_frames
        ))
        
        # Write column labels (marker names)
        f.write("Frame#\tTime\t")
        marker_names = '\t'.join([name for name in labels])
        f.write(marker_names + '\n')
        
        f.write('\t\t')
        coord_names = '\t'.join(['X\tY\tZ' for _ in range(n_markers)])
        f.write(coord_names + '\n')
        
    # Append numerical data
    with open(output_trc, 'a') as f:
        df.to_csv(f, sep='\t', index=False, header=False, float_format='%.6f')
    
    print(f"TRC file successfully created: {output_trc}")
    return output_trc

def check_has_analog_data(c3d_file):
    """
    Check if a C3D file contains analog data
    
    Parameters:
    -----------
    c3d_file : str
        Path to the C3D file
        
    Returns:
    --------
    bool
        True if analog data exists, False otherwise
    """
    c3d = ezc3d.c3d(c3d_file)
    
    if ('analogs' not in c3d['data'] or 
        c3d['data']['analogs'].size == 0 or 
        'LABELS' not in c3d['parameters']['ANALOG']):
        return False
    
    # Check if there's actual data in the analog channels
    analog_data = c3d['data']['analogs']
    if analog_data.size == 0:
        return False
    
    # Extract analog metadata
    labels = c3d['parameters']['ANALOG']['LABELS']['value']
    
    # Check if we have any channels
    if not labels or len(labels) == 0:
        return False
    
    # Check if we have a valid frame rate
    if 'RATE' not in c3d['parameters']['ANALOG']:
        return False
    
    # Get the frame rate and check if it's valid
    frame_rate_value = c3d['parameters']['ANALOG']['RATE']['value']
    frame_rate = float(frame_rate_value[0] if isinstance(frame_rate_value, np.ndarray) else frame_rate_value)
    
    if frame_rate <= 0:
        return False
    
    return True

def process_c3d_file(c3d_file, output_dir=None, file_prefix=None, coordinate_transform='opensim'):
    """
    Process a C3D file to create OpenSim compatible files
    
    Parameters:
    -----------
    c3d_file : str
        Path to the C3D file
    output_dir : str, optional
        Directory where output files will be saved
    file_prefix : str, optional
        Prefix for output filenames
    coordinate_transform : str, optional
        Type of coordinate transform to apply
    
    Returns:
    --------
    dict
        Dictionary containing paths to created files
    """
    # Setup output paths
    if output_dir is None:
        output_dir = os.path.dirname(c3d_file)
    
    if file_prefix is None:
        file_prefix = os.path.splitext(os.path.basename(c3d_file))[0]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    trc_file = os.path.join(output_dir, file_prefix + '.trc')
    
    # Process marker data
    result = {'trc_file': c3d_to_trc(c3d_file, trc_file, coordinate_transform)}
    
    # Check if the file has analog data
    if check_has_analog_data(c3d_file):
        print("Analog data detected in C3D file. Adding MOT file creation functionality.")
        # The MOT file creation would go here if needed in the future
    else:
        print("No analog data found in C3D file. Skipping MOT file creation.")
    
    return result

# Process multiple files in a directory
def process_directory(directory, pattern="*.c3d", output_dir=None, coordinate_transform='opensim'):
    """
    Process all C3D files in a directory
    
    Parameters:
    -----------
    directory : str
        Directory containing C3D files
    pattern : str, optional
        Glob pattern to match C3D files
    output_dir : str, optional
        Directory where output files will be saved
    coordinate_transform : str, optional
        Type of coordinate transform to apply
    """
    import glob
    
    # Get all C3D files in the directory
    c3d_files = glob.glob(os.path.join(directory, pattern))
    
    if not c3d_files:
        print(f"No C3D files found in {directory} matching pattern {pattern}")
        return
    
    # Process each file
    for c3d_file in c3d_files:
        file_name = os.path.basename(c3d_file)
        print(f"Processing {file_name}...")
        process_c3d_file(c3d_file, output_dir, coordinate_transform=coordinate_transform)
    
    print(f"Processed {len(c3d_files)} C3D files")

# Example usage
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert C3D files to OpenSim compatible formats')
    parser.add_argument('input', help='Input C3D file or directory containing C3D files')
    parser.add_argument('--output-dir', '-o', help='Output directory for converted files')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all C3D files in the input directory')
    parser.add_argument('--coordinate-transform', '-c', choices=['opensim', 'none'], default='opensim',
                        help='Type of coordinate transform to apply (default: opensim)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Process all C3D files in the directory
        process_directory(args.input, output_dir=args.output_dir, coordinate_transform=args.coordinate_transform)
    else:
        # Process a single file
        if not os.path.isfile(args.input):
            print(f"Error: File {args.input} not found")
            sys.exit(1)
        
        process_c3d_file(args.input, args.output_dir, coordinate_transform=args.coordinate_transform)
