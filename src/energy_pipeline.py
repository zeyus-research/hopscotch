"""
Main energy analysis pipeline for motion capture data.
Integrates data loading, energy calculations, and group analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
import logging
from tqdm import tqdm

from processing.mocap_loader import MocapDataLoader
from processing.energy_calculator import EnergyCalculator
from processing.group_analysis import GroupAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

class EnergyAnalysisPipeline:
    """
    Complete pipeline for motion capture energy analysis.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the energy analysis pipeline.
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self.config = self._load_config(config_file) if config_file else {}
        self.setup_logging()
        
        # Initialize components
        marker_labels_file = Path(self.config.get('marker_labels_file', 'marker_labels.txt'))
        self.loader = MocapDataLoader(marker_labels_file)
        
        sampling_rate = self.config.get('sampling_rate', 100.0)
        self.calculator = EnergyCalculator(sampling_rate)
        
        # Data storage
        self.raw_data = None
        self.energy_metrics = None
        self.group_analyzer = None
        
    def _load_config(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config file {config_file}: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self, data_path: Path, 
                  file_pattern: str = "*.tsv",
                  subject_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Load motion capture data.
        
        Args:
            data_path: Path to data directory
            file_pattern: File pattern to match
            subject_config: Optional subject/condition configuration
            
        Returns:
            Loaded DataFrame
        """
        logging.info(f"Loading data from {data_path}")
        
        # Use subject config from file if not provided
        if subject_config is None:
            subject_config = self.config.get('subject_config', {})
        
        cache_file = self.config.get('cache_file', 'mocap_data.feather')
        
        self.raw_data = self.loader.load_dataset(
            data_path=data_path,
            file_pattern=file_pattern,
            subject_config=subject_config,
            cache_file=cache_file
        )
        
        logging.info(f"Loaded {len(self.raw_data)} records from {len(self.raw_data.groupby('filename'))} files")
        return self.raw_data
    
    def calculate_energy_metrics(self, body_mass_config: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Calculate energy metrics for all trials.
        
        Args:
            body_mass_config: Dictionary mapping subjects to body masses
            
        Returns:
            DataFrame with energy metrics per trial
        """
        if self.raw_data is None:
            raise ValueError("Must load data first using load_data()")
        
        logging.info("Calculating energy metrics")
        
        # Default body mass if not specified
        default_body_mass = self.config.get('default_body_mass', 70.0)
        
        energy_results = []
        
        # Group by trial (filename)
        grouped = self.raw_data.groupby('filename')
        
        for filename, trial_data in tqdm(grouped, desc="Processing trials"):
            try:
                # Get subject info
                subject = trial_data['subject'].iloc[0]
                condition = trial_data['condition'].iloc[0]
                
                # Handle both 'obstacles' and 'session' columns for compatibility
                if 'obstacles' in trial_data.columns:
                    obstacles = trial_data['obstacles'].iloc[0]
                elif 'session' in trial_data.columns:
                    obstacles = trial_data['session'].iloc[0]
                else:
                    obstacles = '0'
                
                # Get body mass for this subject
                if body_mass_config and subject in body_mass_config:
                    body_mass = body_mass_config[subject]
                else:
                    body_mass = self.config.get('body_masses', {}).get(str(subject), default_body_mass)
                
                # Calculate energy summary for this trial
                energy_summary = self.calculator.calculate_energy_summary(
                    trial_data, body_mass=body_mass
                )
                
                if energy_summary:
                    energy_summary.update({
                        'filename': filename,
                        'subject': subject,
                        'condition': condition,
                        'obstacles': obstacles,
                        'body_mass': body_mass
                    })
                    energy_results.append(energy_summary)
                
            except Exception as e:
                logging.error(f"Error processing trial {filename}: {e}")
                continue
        
        self.energy_metrics = pd.DataFrame(energy_results)
        logging.info(f"Calculated energy metrics for {len(self.energy_metrics)} trials")
        
        return self.energy_metrics
    
    def run_group_analysis(self, metrics: Optional[List[str]] = None) -> GroupAnalyzer:
        """
        Run group-level statistical analysis.
        
        Args:
            metrics: List of metrics to analyze (default: all energy metrics)
            
        Returns:
            GroupAnalyzer instance with results
        """
        if self.energy_metrics is None:
            raise ValueError("Must calculate energy metrics first")
        
        logging.info("Running group analysis")
        
        self.group_analyzer = GroupAnalyzer(self.raw_data, self.energy_metrics)
        
        # Default metrics to analyze (include hopscotch-specific metrics)
        if metrics is None:
            metrics = [
                'metabolic_rate_pandolf',
                'mean_metabolic_power_efficiency', 
                'mean_speed',
                'total_metabolic_energy_pandolf',
                # Hopscotch-specific metrics
                'composite_hopscotch_energy',
                'composite_hopscotch_power',
                'total_vertical_work',
                'mean_vertical_velocity',
                'speed_variability_cv',
                'mean_acceleration_magnitude',
                'coordination_index'
            ]
        
        # Run comparisons for each metric
        for metric in metrics:
            if metric in self.energy_metrics.columns:
                test_type = self.config.get('test_type', 'independent')
                try:
                    self.group_analyzer.compare_conditions(metric, test_type)
                except Exception as e:
                    logging.error(f"Error analyzing metric {metric}: {e}")
        
        return self.group_analyzer
    
    def save_results(self, output_dir: Path, 
                    save_raw: bool = True,
                    save_metrics: bool = True,
                    save_plots: bool = True,
                    save_report: bool = True):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
            save_raw: Save raw processed data
            save_metrics: Save energy metrics
            save_plots: Save analysis plots
            save_report: Save text report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logging.info(f"Saving results to {output_dir}")
        
        # Save raw data
        if save_raw and self.raw_data is not None:
            self.raw_data.to_csv(output_dir / 'raw_data.csv', index=False)
            self.raw_data.to_feather(output_dir / 'raw_data.feather')
        
        # Save energy metrics
        if save_metrics and self.energy_metrics is not None:
            self.energy_metrics.to_csv(output_dir / 'energy_metrics.csv', index=False)
            self.energy_metrics.to_feather(output_dir / 'energy_metrics.feather')
        
        # Save plots and report
        if self.group_analyzer:
            if save_plots:
                plots_dir = output_dir / 'plots'
                plots_dir.mkdir(exist_ok=True)
                
                # Generate and save plots for key metrics (include hopscotch-specific)
                key_metrics = [
                    'metabolic_rate_pandolf', 
                    'mean_speed', 
                    'total_metabolic_energy_pandolf',
                    'composite_hopscotch_energy',
                    'composite_hopscotch_power',
                    'mean_metabolic_power_efficiency',
                    'total_vertical_work',
                    'speed_variability_cv',
                    'mean_vertical_velocity',
                    'coordination_index',
                    'trial_duration'
                ]
                
                for metric in key_metrics:
                    if metric in self.energy_metrics.columns:
                        try:
                            # Condition comparison plot
                            fig = self.group_analyzer.plot_condition_comparison(metric, 'boxplot')
                            fig.savefig(plots_dir / f'{metric}_conditions.png', dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            
                            # Subject profiles plot
                            fig = self.group_analyzer.plot_subject_profiles(metric)
                            fig.savefig(plots_dir / f'{metric}_subjects.png', dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            
                        except Exception as e:
                            logging.error(f"Error creating plots for {metric}: {e}")
                
                # Create comprehensive power efficiency plots
                try:
                    self.create_power_efficiency_plots(output_dir)
                    logging.info("Power efficiency plots created successfully")
                except Exception as e:
                    logging.error(f"Error creating power efficiency plots: {e}")
            
            if save_report:
                report = self.group_analyzer.generate_report()
                with open(output_dir / 'analysis_report.txt', 'w') as f:
                    f.write(report)
        
        logging.info("Results saved successfully")
    
    def create_power_efficiency_plots(self, output_dir: Path):
        """
        Create comprehensive power efficiency comparison plots.
        
        Args:
            output_dir: Directory to save plots
        """
        if self.energy_metrics is None or self.group_analyzer is None:
            logging.warning("No data available for power efficiency plots")
            return
        
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Power efficiency metrics to compare - verify units exist in data
        efficiency_metrics = []
        
        # Define potential metrics with their expected units
        potential_metrics = [
            ('mean_metabolic_power_efficiency', 'Metabolic Power Efficiency', 'W'),
            ('composite_hopscotch_power', 'Hopscotch Power', 'W'),
            ('metabolic_rate_pandolf', 'Pandolf Metabolic Rate', 'W'),
            ('mean_vertical_power', 'Vertical Power', 'W'),
            ('mean_mechanical_power', 'Mechanical Power', 'W'),
            ('trial_duration', 'Trial Duration', 's'),
            ('total_vertical_work', 'Total Vertical Work', 'J'),
            ('mean_speed', 'Mean Speed', 'm/s'),
            ('speed_variability_cv', 'Speed Variability', ''),
            ('coordination_index', 'Coordination Index', '')
        ]
        
        # Only include metrics that exist in the data
        for metric, title, unit in potential_metrics:
            if metric in self.energy_metrics.columns:
                efficiency_metrics.append((metric, title, unit))
        
        # Create comparison plots for each efficiency metric
        for metric, title, unit in efficiency_metrics:
            if metric in self.energy_metrics.columns:
                try:
                    # Create multi-panel comparison plot
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle(f'{title} Analysis', fontsize=16, fontweight='bold')
                    
                    # Condition comparison (boxplot)
                    ax1 = axes[0, 0]
                    # Create the plot using the existing method and copy to our subplot
                    temp_fig = self.group_analyzer.plot_condition_comparison(metric, 'boxplot')
                    
                    # Copy the plot elements to our subplot
                    sns.boxplot(data=self.energy_metrics, x='condition', y=metric, ax=ax1)
                    sns.stripplot(data=self.energy_metrics, x='condition', y=metric, 
                                 color='red', alpha=0.7, ax=ax1)
                    ax1.set_title(f'{title} by Condition')
                    ax1.set_ylabel(f'{title} ({unit})')
                    plt.close(temp_fig)
                    
                    # Obstacle effect comparison
                    ax2 = axes[0, 1]
                    if 'obstacles' in self.energy_metrics.columns:
                        obstacle_data = []
                        obstacle_labels = []
                        for obs_level in sorted(self.energy_metrics['obstacles'].unique()):
                            data = self.energy_metrics[self.energy_metrics['obstacles'] == obs_level][metric]
                            obstacle_data.append(data.dropna())
                            label = 'No Obstacles' if obs_level == '0' else 'With Obstacles'
                            obstacle_labels.append(label)
                        
                        ax2.boxplot(obstacle_data, labels=obstacle_labels)
                        ax2.set_title(f'{title} by Obstacle Condition')
                        ax2.set_ylabel(f'{title} ({unit})')
                    
                    # Subject profiles
                    ax3 = axes[1, 0]
                    # Create subject profiles manually since the method doesn't accept ax parameter
                    subject_condition = self.energy_metrics.groupby(['subject', 'condition'])[metric].mean().unstack()
                    
                    # Plot each subject as a line
                    for subject in subject_condition.index:
                        ax3.plot(subject_condition.columns, subject_condition.loc[subject], 
                               marker='o', alpha=0.7, label=f'Subject {subject}')
                    
                    # Add mean line
                    if not subject_condition.empty:
                        mean_line = subject_condition.mean(axis=0)
                        ax3.plot(subject_condition.columns, mean_line, 
                               color='black', linewidth=3, marker='s', markersize=8, label='Mean')
                    
                    ax3.set_title(f'{title} by Subject')
                    ax3.set_ylabel(f'{title} ({unit})')
                    ax3.set_xlabel('Condition')
                    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Distribution histogram
                    ax4 = axes[1, 1]
                    data = self.energy_metrics[metric].dropna()
                    ax4.hist(data, bins=20, alpha=0.7, edgecolor='black')
                    ax4.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                               label=f'Mean: {data.mean():.2f} {unit}')
                    ax4.axvline(data.median(), color='blue', linestyle='--', linewidth=2,
                               label=f'Median: {data.median():.2f} {unit}')
                    ax4.set_title(f'{title} Distribution')
                    ax4.set_xlabel(f'{title} ({unit})')
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                    
                    plt.tight_layout()
                    
                    # Save plot
                    plot_filename = f'power_efficiency_{metric}.png'
                    fig.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    logging.error(f"Error creating power efficiency plot for {metric}: {e}")
        
        # Create comprehensive power efficiency comparison plot
        try:
            self._create_comprehensive_power_comparison(plots_dir)
        except Exception as e:
            logging.error(f"Error creating comprehensive power comparison: {e}")
        
        # Create condition+obstacle combination plots
        try:
            self._create_condition_obstacle_plots(plots_dir)
        except Exception as e:
            logging.error(f"Error creating condition+obstacle plots: {e}")
    
    def _create_comprehensive_power_comparison(self, plots_dir: Path):
        """Create comprehensive comparison of all power metrics."""
        
        # Available power metrics
        power_metrics = {
            'metabolic_rate_pandolf': 'Pandolf Rate',
            'composite_hopscotch_power': 'Hopscotch Power', 
            'mean_metabolic_power_efficiency': 'Efficiency Power',
            'mean_mechanical_power': 'Mechanical Power',
            'mean_vertical_power': 'Vertical Power'
        }
        
        available_metrics = {k: v for k, v in power_metrics.items() 
                           if k in self.energy_metrics.columns}
        
        if len(available_metrics) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Power Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Correlation matrix
        ax1 = axes[0, 0]
        power_data = self.energy_metrics[list(available_metrics.keys())].dropna()
        if not power_data.empty:
            correlation_matrix = power_data.corr()
            im = ax1.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax1.set_xticks(range(len(available_metrics)))
            ax1.set_yticks(range(len(available_metrics)))
            ax1.set_xticklabels([available_metrics[k] for k in power_data.columns], rotation=45)
            ax1.set_yticklabels([available_metrics[k] for k in power_data.columns])
            ax1.set_title('Power Metrics Correlation')
            
            # Add correlation values
            for i in range(len(available_metrics)):
                for j in range(len(available_metrics)):
                    ax1.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax1)
        
        # Condition-based comparison
        ax2 = axes[0, 1]
        conditions = self.energy_metrics['condition'].unique()
        x_pos = np.arange(len(conditions))
        width = 0.15
        
        for i, (metric, label) in enumerate(available_metrics.items()):
            means = []
            stds = []
            for condition in conditions:
                data = self.energy_metrics[self.energy_metrics['condition'] == condition][metric]
                means.append(data.mean())
                stds.append(data.std())
            
            ax2.bar(x_pos + i*width, means, width, yerr=stds, 
                   label=label, alpha=0.8, capsize=5)
        
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('Metric Value (mixed units)')
        ax2.set_title('Power Metrics by Condition')
        ax2.set_xticks(x_pos + width * (len(available_metrics)-1) / 2)
        ax2.set_xticklabels([c.upper() for c in conditions])
        ax2.legend()
        
        # Efficiency ratio analysis
        ax3 = axes[1, 0]
        if 'composite_hopscotch_power' in available_metrics and 'metabolic_rate_pandolf' in available_metrics:
            self.energy_metrics['efficiency_ratio'] = (
                self.energy_metrics['composite_hopscotch_power'] / 
                self.energy_metrics['metabolic_rate_pandolf']
            )
            
            efficiency_by_condition = []
            for condition in conditions:
                ratio_data = self.energy_metrics[
                    self.energy_metrics['condition'] == condition
                ]['efficiency_ratio'].dropna()
                if not ratio_data.empty:
                    efficiency_by_condition.append(ratio_data)
            
            if efficiency_by_condition:
                ax3.boxplot(efficiency_by_condition, labels=[c.upper() for c in conditions])
                ax3.set_title('Hopscotch/Pandolf Power Ratio')
                ax3.set_ylabel('Efficiency Ratio')
                ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                           label='Equal Power Line')
                ax3.legend()
        
        # Power vs Speed relationship
        ax4 = axes[1, 1]
        if 'mean_speed' in self.energy_metrics.columns:
            for condition in conditions:
                cond_data = self.energy_metrics[self.energy_metrics['condition'] == condition]
                if 'composite_hopscotch_power' in available_metrics:
                    ax4.scatter(cond_data['mean_speed'], 
                              cond_data['composite_hopscotch_power'],
                              label=f'Condition {condition.upper()}', alpha=0.7)
            
            ax4.set_xlabel('Mean Speed (m/s)')
            ax4.set_ylabel('Hopscotch Power (W)')
            ax4.set_title('Power vs Speed Relationship')
            ax4.legend()
        
        plt.tight_layout()
        fig.savefig(plots_dir / 'comprehensive_power_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_condition_obstacle_plots(self, plots_dir: Path):
        """Create plots showing condition+obstacle combinations (6 total conditions)."""
        
        # Create condition+obstacle combination column
        if 'obstacles' in self.energy_metrics.columns:
            self.energy_metrics['condition_obstacle'] = (
                self.energy_metrics['condition'].astype(str) + 
                self.energy_metrics['obstacles'].astype(str)
            )
        else:
            logging.warning("No obstacles column found, skipping condition+obstacle plots")
            return
        
        # Define condition+obstacle labels
        condition_obstacle_labels = {
            'h0': 'Extrinsic (fast)/No Obstacles',
            'h1': 'Extrinsic(fast)/Obstacles', 
            's0': 'Intrinsic (fun)/No Obstacles',
            's1': 'Intrinsic (fun)/Obstacles',
            'k0': 'Control/No Obstacles', 
            'k1': 'Control/Obstacles'
        }
        
        # Metrics to plot with condition+obstacle combinations
        combo_metrics = [
            ('metabolic_rate_pandolf', 'Pandolf Metabolic Rate', 'W'),
            ('composite_hopscotch_power', 'Hopscotch Power', 'W'),
            ('mean_metabolic_power_efficiency', 'Metabolic Power Efficiency', 'W'),
            ('trial_duration', 'Trial Duration', 's'),
            ('mean_speed', 'Mean Speed', 'm/s'),
            ('total_vertical_work', 'Total Vertical Work', 'J'),
            ('speed_variability_cv', 'Speed Variability CV', ''),
            ('coordination_index', 'Coordination Index', '')
        ]
        
        for metric, title, unit in combo_metrics:
            if metric not in self.energy_metrics.columns:
                continue
                
            try:
                # Create 2x2 subplot figure
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'{title} - Condition+Obstacle Analysis', fontsize=16, fontweight='bold')
                
                # Boxplot comparison
                ax1 = axes[0, 0]
                available_combos = sorted(self.energy_metrics['condition_obstacle'].unique())
                combo_data = []
                combo_labels = []
                
                for combo in available_combos:
                    if combo in condition_obstacle_labels:
                        data = self.energy_metrics[
                            self.energy_metrics['condition_obstacle'] == combo
                        ][metric].dropna()
                        if not data.empty:
                            combo_data.append(data)
                            combo_labels.append(condition_obstacle_labels[combo])
                
                if combo_data:
                    box_plot = ax1.boxplot(combo_data, labels=combo_labels, patch_artist=True)
                    
                    # Color boxes by condition
                    colors = {'h': 'lightblue', 's': 'lightgreen', 'k': 'lightcoral'}
                    for combo, patch in zip(available_combos, box_plot['boxes']):
                        if combo in condition_obstacle_labels:
                            condition = combo[0]  # First character is condition
                            patch.set_facecolor(colors.get(condition, 'lightgray'))
                    
                    ax1.set_title(f'{title} by Condition+Obstacle')
                    ax1.set_ylabel(f'{title} ({unit})' if unit else title)
                    ax1.tick_params(axis='x', rotation=45)
                
                # Bar chart with error bars
                ax2 = axes[0, 1]
                if combo_data:
                    means = [data.mean() for data in combo_data]
                    stds = [data.std() for data in combo_data]
                    x_pos = np.arange(len(combo_labels))
                    
                    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                    
                    # Color bars by condition
                    for combo, bar in zip(available_combos, bars):
                        if combo in condition_obstacle_labels:
                            condition = combo[0]
                            bar.set_color(colors.get(condition, 'gray'))
                    
                    ax2.set_title(f'{title} Means by Condition+Obstacle')
                    ax2.set_ylabel(f'{title} ({unit})' if unit else title)
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(combo_labels, rotation=45)
                
                # Subject profiles across condition+obstacle combinations
                ax3 = axes[1, 0]
                if 'subject' in self.energy_metrics.columns:
                    subject_combo = self.energy_metrics.groupby(['subject', 'condition_obstacle'])[metric].mean().unstack()
                    
                    # Plot each subject as a line
                    if not subject_combo.empty:
                        for subject in subject_combo.index:
                            valid_data = subject_combo.loc[subject].dropna()
                            if not valid_data.empty:
                                x_indices = [i for i, combo in enumerate(available_combos) 
                                           if combo in valid_data.index and combo in condition_obstacle_labels]
                                y_values = [valid_data[combo] for combo in available_combos 
                                          if combo in valid_data.index and combo in condition_obstacle_labels]
                                
                                if x_indices and y_values:
                                    ax3.plot(x_indices, y_values, marker='o', alpha=0.7, 
                                           label=f'Subject {subject}')
                        
                        # Add mean line
                        mean_values = []
                        for combo in available_combos:
                            if combo in condition_obstacle_labels:
                                combo_mean = self.energy_metrics[
                                    self.energy_metrics['condition_obstacle'] == combo
                                ][metric].mean()
                                mean_values.append(combo_mean)
                        
                        if mean_values:
                            ax3.plot(range(len(mean_values)), mean_values, 
                                   color='black', linewidth=3, marker='s', markersize=8, 
                                   label='Mean')
                    
                    ax3.set_title(f'{title} Subject Profiles')
                    ax3.set_ylabel(f'{title} ({unit})' if unit else title)
                    ax3.set_xticks(range(len(combo_labels)))
                    ax3.set_xticklabels(combo_labels, rotation=45)
                    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Distribution histogram
                ax4 = axes[1, 1]
                data = self.energy_metrics[metric].dropna()
                if not data.empty:
                    ax4.hist(data, bins=20, alpha=0.7, edgecolor='black')
                    ax4.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {data.mean():.2f} {unit}')
                    ax4.axvline(data.median(), color='blue', linestyle='--', linewidth=2,
                               label=f'Median: {data.median():.2f} {unit}')
                    ax4.set_title(f'{title} Distribution')
                    ax4.set_xlabel(f'{title} ({unit})' if unit else title)
                    ax4.set_ylabel('Frequency')
                    ax4.legend()
                
                plt.tight_layout()
                
                # Save plot
                plot_filename = f'condition_obstacle_{metric}.png'
                fig.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                logging.error(f"Error creating condition+obstacle plot for {metric}: {e}")
        
        # Create summary comparison plot
        self._create_condition_obstacle_summary(plots_dir, condition_obstacle_labels)
    
    def _create_condition_obstacle_summary(self, plots_dir: Path, condition_obstacle_labels: dict):
        """Create summary comparison across all condition+obstacle combinations."""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Condition+Obstacle Summary Analysis', fontsize=16, fontweight='bold')
            
            # Multi-metric comparison
            ax1 = axes[0, 0]
            key_metrics = ['metabolic_rate_pandolf', 'composite_hopscotch_power', 'trial_duration']
            available_combos = sorted(self.energy_metrics['condition_obstacle'].unique())
            
            x_pos = np.arange(len([combo for combo in available_combos if combo in condition_obstacle_labels]))
            width = 0.25
            
            for i, metric in enumerate(key_metrics):
                if metric in self.energy_metrics.columns:
                    means = []
                    stds = []
                    for combo in available_combos:
                        if combo in condition_obstacle_labels:
                            data = self.energy_metrics[
                                self.energy_metrics['condition_obstacle'] == combo
                            ][metric]
                            means.append(data.mean())
                            stds.append(data.std())
                    
                    if means:
                        ax1.bar(x_pos + i*width, means, width, yerr=stds, 
                               label=metric.replace('_', ' ').title(), alpha=0.8, capsize=3)
            
            ax1.set_xlabel('Condition+Obstacle')
            ax1.set_ylabel('Normalized Values')
            ax1.set_title('Key Metrics by Condition+Obstacle')
            ax1.set_xticks(x_pos + width)
            ax1.set_xticklabels([condition_obstacle_labels[combo] for combo in available_combos 
                               if combo in condition_obstacle_labels], rotation=45)
            ax1.legend()
            
            # Trial duration focus
            ax2 = axes[0, 1]
            if 'trial_duration' in self.energy_metrics.columns:
                duration_data = []
                duration_labels = []
                
                for combo in available_combos:
                    if combo in condition_obstacle_labels:
                        data = self.energy_metrics[
                            self.energy_metrics['condition_obstacle'] == combo
                        ]['trial_duration'].dropna()
                        if not data.empty:
                            duration_data.append(data)
                            duration_labels.append(condition_obstacle_labels[combo])
                
                if duration_data:
                    ax2.boxplot(duration_data, labels=duration_labels)
                    ax2.set_title('Trial Duration by Condition+Obstacle')
                    ax2.set_ylabel('Duration (s)')
                    ax2.tick_params(axis='x', rotation=45)
            
            # Obstacle effect analysis
            ax3 = axes[1, 0]
            if 'metabolic_rate_pandolf' in self.energy_metrics.columns:
                conditions = ['h', 's', 'k']
                condition_names = {'h': 'Extrinsic (fast)', 's': 'Intrinsic (fun)', 'k': 'Control'}
                no_obs_means = []
                with_obs_means = []
                
                for cond in conditions:
                    no_obs = self.energy_metrics[
                        self.energy_metrics['condition_obstacle'] == f'{cond}0'
                    ]['metabolic_rate_pandolf'].mean()
                    with_obs = self.energy_metrics[
                        self.energy_metrics['condition_obstacle'] == f'{cond}1'
                    ]['metabolic_rate_pandolf'].mean()
                    
                    no_obs_means.append(no_obs if not np.isnan(no_obs) else 0)
                    with_obs_means.append(with_obs if not np.isnan(with_obs) else 0)
                
                x_pos = np.arange(len(conditions))
                width = 0.35
                
                ax3.bar(x_pos - width/2, no_obs_means, width, label='No Obstacles', alpha=0.8)
                ax3.bar(x_pos + width/2, with_obs_means, width, label='With Obstacles', alpha=0.8)
                
                ax3.set_xlabel('Movement Condition')
                ax3.set_ylabel('Pandolf Metabolic Rate (W)')
                ax3.set_title('Obstacle Effect on Metabolic Rate')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels([condition_names[c] for c in conditions])
                ax3.legend()
            
            # Correlation between key metrics
            ax4 = axes[1, 1]
            if all(col in self.energy_metrics.columns for col in ['trial_duration', 'metabolic_rate_pandolf']):
                for combo in available_combos:
                    if combo in condition_obstacle_labels:
                        combo_data = self.energy_metrics[
                            self.energy_metrics['condition_obstacle'] == combo
                        ]
                        if not combo_data.empty:
                            ax4.scatter(combo_data['trial_duration'], 
                                      combo_data['metabolic_rate_pandolf'],
                                      label=condition_obstacle_labels[combo], alpha=0.7)
                
                ax4.set_xlabel('Trial Duration (s)')
                ax4.set_ylabel('Metabolic Rate (W)')
                ax4.set_title('Duration vs Metabolic Rate')
                ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            fig.savefig(plots_dir / 'condition_obstacle_summary.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            logging.error(f"Error creating condition+obstacle summary: {e}")

    def run_full_pipeline(self, 
                         data_path: Path,
                         output_dir: Path,
                         file_pattern: str = "*.tsv",
                         subject_config: Optional[Dict] = None,
                         body_mass_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            data_path: Path to motion capture data
            output_dir: Output directory for results  
            file_pattern: File pattern to match
            subject_config: Subject/condition configuration
            body_mass_config: Body mass configuration
            
        Returns:
            Dictionary with pipeline results summary
        """
        logging.info("Starting full energy analysis pipeline")
        
        try:
            # Load data
            self.load_data(data_path, file_pattern, subject_config)
            
            # Calculate energy metrics
            self.calculate_energy_metrics(body_mass_config)
            
            # Run group analysis
            self.run_group_analysis()
            
            # Save results
            self.save_results(output_dir)
            
            # Generate summary
            summary = {
                'total_trials': len(self.energy_metrics),
                'subjects': list(self.energy_metrics['subject'].unique()),
                'conditions': list(self.energy_metrics['condition'].unique()),
                'mean_metabolic_rate': self.energy_metrics['metabolic_rate_pandolf'].mean(),
                'output_directory': str(output_dir)
            }
            
            logging.info("Pipeline completed successfully")
            return summary
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise

def create_example_config() -> Dict[str, Any]:
    """Create example configuration dictionary."""
    return {
        'marker_labels_file': 'marker_labels.txt',
        'sampling_rate': 100.0,
        'default_body_mass': 70.0,
        'cache_file': 'mocap_data.feather',
        'log_level': 'INFO',
        'test_type': 'independent',  # or 'paired'
        'body_masses': {
            '23': 75.0,
            '24': 68.0,
            '25': 72.0
        },
        'subject_config': {
            # Example: map filenames to subject/condition info
            # 'custom_file.tsv': {'subject': 'S01', 'condition': 'A', 'session': '1'}
        }
    }
