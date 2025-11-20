"""
Example usage of the motion capture energy analysis pipeline.
"""

from pathlib import Path
import yaml
from src.energy_pipeline import EnergyAnalysisPipeline, create_example_config

def main():
    """Run example analysis."""
    
    # Setup paths
    data_dir = Path("data")  # Adjust to your data directory
    output_dir = Path("results")
    config_file = Path("config.yaml")
    
    # Create example configuration file
    if not config_file.exists():
        config = create_example_config()
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created example config file: {config_file}")
    
    # Initialize pipeline
    pipeline = EnergyAnalysisPipeline(config_file)
    
    # Run full analysis
    try:
        summary = pipeline.run_full_pipeline(
            data_path=data_dir,
            output_dir=output_dir,
            file_pattern="*.tsv"  # or "*.c3d" for C3D files
        )
        
        print("Analysis completed successfully!")
        print(f"Processed {summary['total_trials']} trials")
        print(f"Subjects: {summary['subjects']}")
        print(f"Conditions: {summary['conditions']}")
        print(f"Mean metabolic rate: {summary['mean_metabolic_rate']:.2f} W")
        print(f"Results saved to: {summary['output_directory']}")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")

def run_step_by_step_example():
    """Example of running pipeline step by step."""
    
    # Initialize pipeline without config file
    pipeline = EnergyAnalysisPipeline()
    
    # Load data
    data_dir = Path("data")
    raw_data = pipeline.load_data(data_dir, "*.tsv")
    print(f"Loaded {len(raw_data)} records")
    
    # Calculate energy metrics
    # You can specify body masses for different subjects
    body_masses = {'23': 75.0, '24': 68.0}  # subject_id: mass_kg
    energy_metrics = pipeline.calculate_energy_metrics(body_masses)
    print(f"Calculated metrics for {len(energy_metrics)} trials")
    
    # Run group analysis
    group_analyzer = pipeline.run_group_analysis()
    
    # Get summaries
    subject_summary = group_analyzer.get_subject_summary()
    condition_summary = group_analyzer.get_condition_summary()
    
    print("\nSubject Summary:")
    print(subject_summary)
    print("\nCondition Summary:")
    print(condition_summary)
    
    # Compare conditions
    comparison_results = group_analyzer.compare_conditions('metabolic_rate_pandolf')
    for comp in comparison_results['comparisons']:
        print(f"{comp['condition1']} vs {comp['condition2']}: p={comp['p_value']:.3f}")
    
    # Save results
    pipeline.save_results(Path("results"))

if __name__ == "__main__":
    main()
    # Uncomment to run step-by-step example instead
    # run_step_by_step_example()