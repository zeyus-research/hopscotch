"""
Group-level analysis for motion capture energy data.
Provides statistical analysis and visualization across subjects and conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, f_oneway
import logging

class GroupAnalyzer:
    """
    Perform group-level statistical analysis on energy data.
    """
    
    def __init__(self, data: pd.DataFrame, energy_metrics: pd.DataFrame):
        """
        Initialize group analyzer.
        
        Args:
            data: Raw motion capture data with subject/condition info
            energy_metrics: Calculated energy metrics per trial
        """
        self.data = data
        self.energy_metrics = energy_metrics
        self.results = {}
        
    def get_subject_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics per subject.
        
        Returns:
            DataFrame with subject-level summaries
        """
        # Group by subject and calculate means
        subject_summary = self.energy_metrics.groupby('subject').agg({
            'mean_speed': ['mean', 'std', 'count'],
            'mean_metabolic_power_efficiency': ['mean', 'std'],
            'metabolic_rate_pandolf': ['mean', 'std'],
            'total_metabolic_energy_pandolf': ['mean', 'std'],
            'trial_duration': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        subject_summary.columns = ['_'.join(col).strip() for col in subject_summary.columns]
        
        return subject_summary
    
    def get_condition_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics per condition.
        
        Returns:
            DataFrame with condition-level summaries
        """
        condition_summary = self.energy_metrics.groupby('condition').agg({
            'mean_speed': ['mean', 'std', 'count'],
            'mean_metabolic_power_efficiency': ['mean', 'std'],
            'metabolic_rate_pandolf': ['mean', 'std'],
            'total_metabolic_energy_pandolf': ['mean', 'std'],
            'trial_duration': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        condition_summary.columns = ['_'.join(col).strip() for col in condition_summary.columns]
        
        return condition_summary
    
    def compare_conditions(self, metric: str = 'metabolic_rate_pandolf',
                          test_type: str = 'independent') -> Dict[str, Any]:
        """
        Compare conditions using statistical tests.
        
        Args:
            metric: Energy metric to compare
            test_type: 'independent' for between-subjects, 'paired' for within-subjects
            
        Returns:
            Dictionary with test results
        """
        if metric not in self.energy_metrics.columns:
            raise ValueError(f"Metric {metric} not found in energy metrics")
        
        conditions = self.energy_metrics['condition'].unique()
        
        if len(conditions) < 2:
            logging.warning("Need at least 2 conditions for comparison")
            return {}
        
        results = {
            'metric': metric,
            'test_type': test_type,
            'conditions': list(conditions),
            'comparisons': []
        }
        
        # Pairwise comparisons
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                data1 = self.energy_metrics[self.energy_metrics['condition'] == cond1][metric].dropna()
                data2 = self.energy_metrics[self.energy_metrics['condition'] == cond2][metric].dropna()
                
                if len(data1) == 0 or len(data2) == 0:
                    continue
                
                # Choose appropriate test
                if test_type == 'paired':
                    # Check if we have paired data (same subjects)
                    subjects1 = set(self.energy_metrics[self.energy_metrics['condition'] == cond1]['subject'])
                    subjects2 = set(self.energy_metrics[self.energy_metrics['condition'] == cond2]['subject'])
                    common_subjects = subjects1.intersection(subjects2)
                    
                    if len(common_subjects) < 2:
                        logging.warning(f"Not enough paired data for {cond1} vs {cond2}")
                        continue
                    
                    # Get paired data
                    paired_data1 = []
                    paired_data2 = []
                    for subj in common_subjects:
                        val1 = self.energy_metrics[
                            (self.energy_metrics['subject'] == subj) & 
                            (self.energy_metrics['condition'] == cond1)
                        ][metric].mean()  # Average if multiple trials
                        val2 = self.energy_metrics[
                            (self.energy_metrics['subject'] == subj) & 
                            (self.energy_metrics['condition'] == cond2)
                        ][metric].mean()
                        paired_data1.append(val1)
                        paired_data2.append(val2)
                    
                    stat, p_value = ttest_rel(paired_data1, paired_data2)
                    test_name = 'Paired t-test'
                    
                else:
                    stat, p_value = ttest_ind(data1, data2)
                    test_name = 'Independent t-test'
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                     (len(data2)-1)*np.var(data2, ddof=1)) / 
                                    (len(data1) + len(data2) - 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                
                comparison = {
                    'condition1': cond1,
                    'condition2': cond2,
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2),
                    'std1': np.std(data1),
                    'std2': np.std(data2),
                    'n1': len(data1),
                    'n2': len(data2)
                }
                
                results['comparisons'].append(comparison)
        
        # Overall ANOVA if more than 2 conditions
        if len(conditions) > 2:
            groups = [self.energy_metrics[self.energy_metrics['condition'] == cond][metric].dropna() 
                     for cond in conditions]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups
            
            if len(groups) > 1:
                f_stat, anova_p = f_oneway(*groups)
                results['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': anova_p
                }
        
        self.results[f'comparison_{metric}'] = results
        return results
    
    def plot_condition_comparison(self, metric: str = 'metabolic_rate_pandolf',
                                 plot_type: str = 'boxplot') -> plt.Figure:
        """
        Create plots comparing conditions.
        
        Args:
            metric: Energy metric to plot
            plot_type: 'boxplot', 'violin', or 'bar'
            
        Returns:
            Matplotlib figure
        """
        if metric not in self.energy_metrics.columns:
            raise ValueError(f"Metric {metric} not found in energy metrics")
        
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'boxplot':
            sns.boxplot(data=self.energy_metrics, x='condition', y=metric, ax=ax)
            sns.stripplot(data=self.energy_metrics, x='condition', y=metric, 
                         color='red', alpha=0.7, ax=ax)
        elif plot_type == 'violin':
            sns.violinplot(data=self.energy_metrics, x='condition', y=metric, ax=ax)
        elif plot_type == 'bar':
            condition_means = self.energy_metrics.groupby('condition')[metric].agg(['mean', 'std'])
            ax.bar(range(len(condition_means)), condition_means['mean'], 
                  yerr=condition_means['std'], capsize=5)
            ax.set_xticks(range(len(condition_means)))
            ax.set_xticklabels(condition_means.index)
        
        # Define units for common metrics
        metric_units = {
            'metabolic_rate_pandolf': 'W',
            'composite_hopscotch_power': 'W', 
            'mean_metabolic_power_efficiency': 'W',
            'mean_mechanical_power': 'W',
            'mean_vertical_power': 'W',
            'total_metabolic_energy_pandolf': 'J',
            'total_vertical_work': 'J',
            'mean_kinetic_energy': 'J',
            'mean_potential_energy': 'J',
            'mean_speed': 'm/s',
            'mean_vertical_velocity': 'm/s',
            'trial_duration': 's',
            'speed_variability_cv': '',
            'coordination_index': ''
        }
        
        unit = metric_units.get(metric, '')
        unit_label = f' ({unit})' if unit else ''
        
        ax.set_title(f'{metric.replace("_", " ").title()} by Condition')
        ax.set_xlabel('Condition')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}{unit_label}')
        
        plt.tight_layout()
        return fig
    
    def plot_subject_profiles(self, metric: str = 'metabolic_rate_pandolf') -> plt.Figure:
        """
        Create individual subject profiles across conditions.
        
        Args:
            metric: Energy metric to plot
            
        Returns:
            Matplotlib figure
        """
        if metric not in self.energy_metrics.columns:
            raise ValueError(f"Metric {metric} not found in energy metrics")
        
        # Create subject x condition matrix
        subject_condition = self.energy_metrics.groupby(['subject', 'condition'])[metric].mean().unstack()
        
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each subject as a line
        for subject in subject_condition.index:
            ax.plot(subject_condition.columns, subject_condition.loc[subject], 
                   marker='o', alpha=0.7, label=f'Subject {subject}')
        
        # Add mean line
        mean_line = subject_condition.mean(axis=0)
        ax.plot(subject_condition.columns, mean_line, 
               color='black', linewidth=3, marker='s', markersize=8, label='Mean')
        
        # Use the same metric units dictionary
        metric_units = {
            'metabolic_rate_pandolf': 'W',
            'composite_hopscotch_power': 'W', 
            'mean_metabolic_power_efficiency': 'W',
            'mean_mechanical_power': 'W',
            'mean_vertical_power': 'W',
            'total_metabolic_energy_pandolf': 'J',
            'total_vertical_work': 'J',
            'mean_kinetic_energy': 'J',
            'mean_potential_energy': 'J',
            'mean_speed': 'm/s',
            'mean_vertical_velocity': 'm/s',
            'trial_duration': 's',
            'speed_variability_cv': '',
            'coordination_index': ''
        }
        
        unit = metric_units.get(metric, '')
        unit_label = f' ({unit})' if unit else ''
        
        ax.set_title(f'Individual Subject Profiles: {metric.replace("_", " ").title()}')
        ax.set_xlabel('Condition')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}{unit_label}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("MOTION CAPTURE ENERGY ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 20)
        report_lines.append(f"Total trials: {len(self.energy_metrics)}")
        report_lines.append(f"Subjects: {len(self.energy_metrics['subject'].unique())}")
        report_lines.append(f"Conditions: {list(self.energy_metrics['condition'].unique())}")
        report_lines.append("")
        
        # Subject summary
        subject_summary = self.get_subject_summary()
        report_lines.append("SUBJECT SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(subject_summary.to_string())
        report_lines.append("")
        
        # Condition summary
        condition_summary = self.get_condition_summary()
        report_lines.append("CONDITION SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(condition_summary.to_string())
        report_lines.append("")
        
        # Statistical comparisons
        if 'comparison_metabolic_rate_pandolf' in self.results:
            results = self.results['comparison_metabolic_rate_pandolf']
            report_lines.append("STATISTICAL COMPARISONS")
            report_lines.append("-" * 20)
            
            for comp in results['comparisons']:
                report_lines.append(f"{comp['condition1']} vs {comp['condition2']}:")
                report_lines.append(f"  Test: {comp['test']}")
                report_lines.append(f"  Statistic: {comp['statistic']:.3f}")
                report_lines.append(f"  P-value: {comp['p_value']:.3f}")
                report_lines.append(f"  Effect size (Cohen's d): {comp['cohens_d']:.3f}")
                report_lines.append(f"  Mean difference: {comp['mean1']:.3f} vs {comp['mean2']:.3f}")
                report_lines.append("")
            
            if 'anova' in results:
                report_lines.append("Overall ANOVA:")
                report_lines.append(f"  F-statistic: {results['anova']['f_statistic']:.3f}")
                report_lines.append(f"  P-value: {results['anova']['p_value']:.3f}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logging.info(f"Report saved to {output_file}")
        
        return report_text