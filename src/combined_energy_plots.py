"""
Combined energy analysis plots for hopscotch motion capture data.
Creates comprehensive visualizations showing energy usage by condition, time, and trial duration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

def create_combined_energy_plot(energy_metrics: pd.DataFrame, 
                               raw_data: Optional[pd.DataFrame] = None,
                               output_file: Optional[str] = None,
                               figsize: tuple = (20, 12)) -> plt.Figure:
    """
    Create a combined plot showing:
    1) Total energy usage by condition
    2) Energy usage by time (per second)
    3) Trial duration by condition
    
    Args:
        energy_metrics: DataFrame with calculated energy metrics per trial
        raw_data: Optional raw motion capture data for time-series analysis
        output_file: Optional path to save the plot
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Hopscotch Energy Analysis: Combined View', fontsize=16, fontweight='bold')
    
    # Define consistent colors for conditions
    conditions = sorted(energy_metrics['condition'].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))
    condition_colors = dict(zip(conditions, colors))
    
    # 1. Total Energy Usage by Condition (Top Left)
    
    ax1 = axes[0]
    if 'composite_hopscotch_energy' in energy_metrics.columns:
        energy_metric = 'composite_hopscotch_energy'
        energy_label = 'Composite Hopscotch Energy (J)'
    elif 'total_metabolic_energy_pandolf' in energy_metrics.columns:
        energy_metric = 'total_metabolic_energy_pandolf'
        energy_label = 'Total Metabolic Energy (J)'
    
    else:
        # Use the first available energy metric
        energy_cols = [col for col in energy_metrics.columns if 'energy' in col.lower()]
        energy_metric = energy_cols[0] if energy_cols else 'metabolic_rate_pandolf'
        energy_label = energy_metric.replace('_', ' ').title()
    
    # Create box plot for energy by condition
    box_data = []
    box_labels = []
    for condition in conditions:
        data = energy_metrics[energy_metrics['condition'] == condition][energy_metric].dropna()
        if not data.empty:
            box_data.append(data)
            box_labels.append(condition.upper())
    
    if box_data:
        box_plot = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, condition in zip(box_plot['boxes'], conditions):
            patch.set_facecolor(condition_colors[condition])
            patch.set_alpha(0.7)
        
        # Add individual data points
        for i, (condition, data) in enumerate(zip(conditions, box_data)):
            y = data.values
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax1.scatter(x, y, alpha=0.6, color=condition_colors[condition], s=30)
    
    ax1.set_title('Total Energy Usage by Condition')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel(energy_label)
    ax1.grid(True, alpha=0.3)
    

    # 5. Energy Efficiency Comparison (Bottom Middle)
    ax2 = axes[1]
    if 'trial_duration' in energy_metrics.columns and energy_metric in energy_metrics.columns:
        # Calculate energy per second (efficiency metric)
        energy_metrics_copy = energy_metrics.copy()
        energy_metrics_copy['energy_per_second'] = (
            energy_metrics_copy[energy_metric] / energy_metrics_copy['trial_duration']
        )
        
        efficiency_data = []
        efficiency_labels = []
        for condition in conditions:
            data = energy_metrics_copy[energy_metrics_copy['condition'] == condition]['energy_per_second'].dropna()
            if not data.empty:
                efficiency_data.append(data)
                efficiency_labels.append(condition.upper())
        
        if efficiency_data:
            box_plot = ax2.boxplot(efficiency_data, labels=efficiency_labels, patch_artist=True)
            for patch, condition in zip(box_plot['boxes'], conditions):
                patch.set_facecolor(condition_colors[condition])
                patch.set_alpha(0.7)
    
    ax2.set_title('Energy Efficiency (Energy/Time)')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Energy per Second (J/s)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Trial Duration by Condition (Top Right)
    ax3 = axes[2]
    if 'trial_duration' in energy_metrics.columns:
        duration_data = []
        duration_labels = []
        for condition in conditions:
            data = energy_metrics[energy_metrics['condition'] == condition]['trial_duration'].dropna()
            if not data.empty:
                duration_data.append(data)
                duration_labels.append(condition.upper())
        
        if duration_data:
            # Create bar plot with error bars
            means = [data.mean() for data in duration_data]
            stds = [data.std() for data in duration_data]
            x_pos = np.arange(len(duration_labels))
            
            bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            for bar, condition in zip(bars, conditions):
                bar.set_color(condition_colors[condition])
            
            # Add individual data points
            for i, data in enumerate(duration_data):
                y = data.values
                x = np.random.normal(i, 0.1, size=len(y))
                ax3.scatter(x, y, alpha=0.6, color='black', s=20)
    
    ax3.set_title('Trial Duration by Condition')
    ax3.set_xlabel('Condition')
    ax3.set_ylabel('Duration (seconds)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(duration_labels)
    ax3.grid(True, alpha=0.3)
    
    
    # Adjust layout and save
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Combined energy plot saved to {output_file}")
    
    return fig


def load_and_plot_energy_data(data_dir: str, 
                             output_file: str = "combined_energy_analysis.png") -> plt.Figure:
    """
    Convenience function to load data and create combined energy plot.
    
    Args:
        data_dir: Directory containing the analysis results
        output_file: Output filename for the plot
        
    Returns:
        Matplotlib figure object
    """
    data_path = Path(data_dir)
    
    # Try to load energy metrics
    energy_metrics = None
    for file_ext in ['.feather', '.csv']:
        energy_file = data_path / f'energy_metrics{file_ext}'
        if energy_file.exists():
            if file_ext == '.feather':
                energy_metrics = pd.read_feather(energy_file)
            else:
                energy_metrics = pd.read_csv(energy_file)
            break
    
    if energy_metrics is None:
        raise FileNotFoundError("No energy metrics file found. Run the energy analysis pipeline first.")
    
    # Try to load raw data for time series
    raw_data = None
    for file_ext in ['.feather', '.csv']:
        raw_file = data_path / f'raw_data{file_ext}'
        if raw_file.exists():
            if file_ext == '.feather':
                raw_data = pd.read_feather(raw_file)
            else:
                raw_data = pd.read_csv(raw_file)
            break
    
    # Create the combined plot
    fig = create_combined_energy_plot(
        energy_metrics=energy_metrics,
        raw_data=raw_data,
        output_file=output_file
    )
    
    return fig


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Create combined energy analysis plots')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing energy metrics and raw data files')
    parser.add_argument('--output', type=str, default='combined_energy_analysis.png',
                       help='Output filename for the plot')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        fig = load_and_plot_energy_data(args.data_dir, args.output)
        plt.show()
        logging.info("Combined energy analysis plot created successfully!")
        
    except Exception as e:
        logging.error(f"Error creating plot: {e}")
        raise
