"""
Hopscotch Data Analysis Script

Analyzes all TSV files in the data/ directory using the energy analysis pipeline.
File format: {SUBJECTID}_{CONDITION}{OBSTACLES}.tsv
- CONDITION: h (hop), s (step), k (kick)  
- OBSTACLES: 0 (no obstacles), 1 (with obstacles)
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import logging

# Add src to path
sys.path.append('src')

from energy_pipeline import EnergyAnalysisPipeline
from combined_energy_plots import create_combined_energy_plot

def setup_logging():
    """Setup logging for the analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hopscotch_analysis.log'),
            logging.StreamHandler()
        ]
    )

def create_subject_body_mass_config():
    """Create body mass configuration for all subjects."""
    
    # Get all TSV files to identify subjects
    data_path = Path("data")
    tsv_files = list(data_path.glob("*.tsv"))
    
    subjects = set()
    for file in tsv_files:
        try:
            subject_id = file.stem.split('_')[0]
            subjects.add(subject_id)
        except:
            continue
    
    # Assign realistic body masses (you can customize these)
    body_masses = {}
    for subject in sorted(subjects):
        # Default assignment - you can customize based on actual subject data
        if subject.isdigit():
            # subject_num = int(subject)
            # Vary body mass slightly based on subject number for realism
            base_mass = 25.0  # kg
            variation = 0
            body_masses[subject] = base_mass + variation
        else:
            body_masses[subject] = 25.0
    
    logging.info(f"Created body mass config for {len(subjects)} subjects")
    return body_masses

def analyze_by_condition_and_obstacles():
    """Run comprehensive analysis of hopscotch data."""
    
    print("🏃 HOPSCOTCH MOTION CAPTURE ENERGY ANALYSIS")
    print("=" * 60)
    
    # Setup
    setup_logging()
    data_path = Path("data")
    output_dir = Path("hopscotch_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create configuration
    config = {
        'marker_labels_file': 'marker_labels.txt',
        'sampling_rate': 300.0,  # Adjust if needed
        'default_body_mass': 25.0,
        'cache_file': 'hopscotch_data.feather',
        'log_level': 'INFO',
        'test_type': 'independent',
        'body_masses': create_subject_body_mass_config()
    }
    
    # Save config for reference
    config_file = output_dir / 'analysis_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize pipeline
    pipeline = EnergyAnalysisPipeline()
    pipeline.config = config
    
    print(f"\n📁 Loading data from: {data_path}")
    
    # Load all TSV files
    try:
        raw_data = pipeline.load_data(data_path, "*.tsv")
        print(f"✅ Loaded {len(raw_data)} total frames")
        
        # Show data overview
        files_loaded = raw_data['filename'].nunique()
        subjects = raw_data['subject'].nunique()
        conditions = sorted(raw_data['condition'].unique())
        obstacles = sorted(raw_data['obstacles'].unique())
        
        condition_labels = {
            'h': 'extrinsic/speed', 
            's': 'intrinsic/fun', 
            'k': 'control'
        }
        
        print(f"📊 Dataset Overview:")
        print(f"   Files: {files_loaded}")
        print(f"   Subjects: {subjects}")
        print(f"   Conditions: {conditions}")
        for cond in conditions:
            print(f"      {cond} = {condition_labels.get(cond, 'unknown')}")
        print(f"   Obstacles: {obstacles} (0=none, 1=obstacles)")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"\n⚡ Calculating energy metrics...")
    
    # Calculate energy metrics
    try:
        energy_metrics = pipeline.calculate_energy_metrics(config['body_masses'])
        print(f"✅ Calculated metrics for {len(energy_metrics)} trials")
        
        # Show sample results (include hopscotch-specific metrics)
        print(f"\n📈 Sample Results:")
        for condition in conditions:
            condition_data = energy_metrics[energy_metrics['condition'] == condition]
            if len(condition_data) > 0:
                mean_speed = condition_data['mean_speed'].mean()
                pandolf_energy = condition_data['metabolic_rate_pandolf'].mean()
                hopscotch_power = condition_data['composite_hopscotch_power'].mean() if 'composite_hopscotch_power' in condition_data.columns else 0
                vertical_work = condition_data['total_vertical_work'].mean() if 'total_vertical_work' in condition_data.columns else 0
                print(f"   {condition.upper()}: {mean_speed:.1f} m/s")
                print(f"      Pandolf: {pandolf_energy:.1f} W, Hopscotch: {hopscotch_power:.1f} W")
                print(f"      Vertical work: {vertical_work:.1f} J")
                
    except Exception as e:
        print(f"❌ Error calculating energy metrics: {e}")
        return
    
    print(f"\n📊 Running statistical analysis...")
    
    # Group analysis
    try:
        group_analyzer = pipeline.run_group_analysis([
            'metabolic_rate_pandolf',
            'mean_speed', 
            'total_metabolic_energy_pandolf',
            'trial_duration',
            # Hopscotch-specific metrics
            'composite_hopscotch_power',
            'total_vertical_work',
            'mean_vertical_velocity',
            'speed_variability_cv',
            'coordination_index'
        ])
        
        print(f"✅ Statistical analysis completed")
        
        # Show condition comparisons for multiple metrics
        print(f"\n🔬 Condition Comparisons:")
        
        # Compare both traditional and hopscotch metrics
        comparison_metrics = [
            ('metabolic_rate_pandolf', 'Pandolf Rate'),
            ('composite_hopscotch_power', 'Hopscotch Power'),
            ('total_vertical_work', 'Vertical Work')
        ]
        
        for metric, label in comparison_metrics:
            if metric in energy_metrics.columns:
                print(f"\n   {label}:")
                comparison = group_analyzer.compare_conditions(metric)
                
                for comp in comparison.get('comparisons', []):
                    p_val = comp['p_value']
                    effect_size = comp['cohens_d']
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    
                    print(f"      {comp['condition1'].upper()} vs {comp['condition2'].upper()}: "
                          f"p={p_val:.3f} {significance}, d={effect_size:.2f}")
        
        # Show subject analysis 
        print(f"\n👥 Subject Summary:")
        subject_summary = group_analyzer.get_subject_summary()
        
        # Show both traditional and hopscotch metrics
        metric_cols = {
            'metabolic_rate_pandolf': 'Pandolf',
            'composite_hopscotch_power': 'Hopscotch', 
            'total_vertical_work': 'Vertical Work'
        }
        
        available_cols = {k: v for k, v in metric_cols.items() if f'mean_{k}' in subject_summary.columns}
        
        if available_cols:
            for subject in subject_summary.index[:5]:  # Show first 5 subjects
                values = []
                for col, label in available_cols.items():
                    mean_col = f'mean_{col}'
                    if mean_col in subject_summary.columns:
                        value = subject_summary.loc[subject, mean_col]
                        unit = 'W' if 'power' in col or 'rate' in col else 'J'
                        values.append(f"{label}: {value:.1f} {unit}")
                
                print(f"   Subject {subject}: {', '.join(values)}")
            
            if len(subject_summary) > 5:
                print(f"   ... and {len(subject_summary) - 5} more subjects")
                
    except Exception as e:
        print(f"❌ Error in statistical analysis: {e}")
        logging.error(f"Statistical analysis error: {e}", exc_info=True)
    
    print(f"\n💾 Saving results to: {output_dir}")
    
    # Save results
    try:
        pipeline.save_results(
            output_dir,
            save_raw=True,
            save_metrics=True, 
            save_plots=False,
            save_report=True
        )
        print(f"✅ Results saved successfully")
        
        # Create the combined energy analysis plot
        print(f"📊 Creating combined energy analysis plot...")
        try:
            combined_plot = create_combined_energy_plot(
                energy_metrics=pipeline.energy_metrics,
                raw_data=pipeline.raw_data,
                output_file=str(output_dir / "combined_energy_analysis.png")
            )
            import matplotlib.pyplot as plt
            plt.close(combined_plot)  # Close to free memory
            print(f"✅ Combined energy plot saved to: {output_dir}/combined_energy_analysis.png")
        except Exception as e:
            print(f"⚠️  Warning: Could not create combined energy plot: {e}")
            logging.warning(f"Combined plot creation failed: {e}")
        
        # List output files
        output_files = list(output_dir.glob("*"))
        print(f"\n📄 Generated Files:")
        for file in sorted(output_files):
            size_kb = file.stat().st_size / 1024 if file.is_file() else 0
            icon = "📊" if file.name == "combined_energy_analysis.png" else "📄"
            print(f"   {icon} {file.name} ({size_kb:.1f} KB)")
            
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        logging.error(f"Save error: {e}", exc_info=True)
    
    # Create obstacle comparison analysis
    print(f"\n🚧 Analyzing obstacle effects...")
    
    try:
        # Add obstacle analysis
        obstacle_analysis = analyze_obstacle_effects(energy_metrics, output_dir)
        print(f"✅ Obstacle analysis completed")
        
    except Exception as e:
        print(f"❌ Error in obstacle analysis: {e}")
        logging.error(f"Obstacle analysis error: {e}", exc_info=True)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Check {output_dir} for detailed results")
    print(f"📝 See hopscotch_analysis.log for detailed logs")

def analyze_obstacle_effects(energy_metrics, output_dir):
    """Analyze the effect of obstacles on energy expenditure."""
    
    obstacle_results = []
    
    condition_labels = {
        'h': 'Extrinsic/Speed', 
        's': 'Intrinsic/Fun', 
        'k': 'Control'
    }
    
    # Compare obstacle vs no-obstacle for each condition
    for condition in energy_metrics['condition'].unique():
        condition_data = energy_metrics[energy_metrics['condition'] == condition]
        
        no_obstacles = condition_data[condition_data['obstacles'] == '0']
        with_obstacles = condition_data[condition_data['obstacles'] == '1'] 
        
        if len(no_obstacles) > 0 and len(with_obstacles) > 0:
            # Calculate means for both traditional and hopscotch metrics
            pandolf_no_obs = no_obstacles['metabolic_rate_pandolf'].mean()
            pandolf_with_obs = with_obstacles['metabolic_rate_pandolf'].mean()
            pandolf_pct = ((pandolf_with_obs - pandolf_no_obs) / pandolf_no_obs) * 100
            
            # Hopscotch metrics if available
            hopscotch_metrics = {}
            if 'composite_hopscotch_power' in no_obstacles.columns:
                hop_no_obs = no_obstacles['composite_hopscotch_power'].mean()
                hop_with_obs = with_obstacles['composite_hopscotch_power'].mean()
                hop_pct = ((hop_with_obs - hop_no_obs) / hop_no_obs) * 100 if hop_no_obs > 0 else 0
                hopscotch_metrics['hopscotch_no_obs'] = hop_no_obs
                hopscotch_metrics['hopscotch_with_obs'] = hop_with_obs
                hopscotch_metrics['hopscotch_pct'] = hop_pct
            
            obstacle_results.append({
                'condition': condition,
                'condition_name': condition_labels.get(condition, condition),
                'pandolf_no_obstacles': pandolf_no_obs,
                'pandolf_with_obstacles': pandolf_with_obs,
                'pandolf_percent_increase': pandolf_pct,
                'n_no_obstacles': len(no_obstacles),
                'n_with_obstacles': len(with_obstacles),
                **hopscotch_metrics
            })
            
            condition_name = condition_labels.get(condition, condition.upper())
            print(f"   {condition_name}:")
            print(f"      Pandolf: {pandolf_pct:+.1f}% increase with obstacles")
            if 'hopscotch_pct' in hopscotch_metrics:
                print(f"      Hopscotch: {hopscotch_metrics['hopscotch_pct']:+.1f}% increase with obstacles")
    
    # Save obstacle analysis
    if obstacle_results:
        obstacle_df = pd.DataFrame(obstacle_results)
        obstacle_df.to_csv(output_dir / 'obstacle_effects.csv', index=False)
        
        # Create summary
        summary_text = "OBSTACLE EFFECTS ANALYSIS\n"
        summary_text += "=" * 40 + "\n\n"
        summary_text += "Effect of obstacles on metabolic energy expenditure\n"
        summary_text += "by movement condition in hopscotch task.\n\n"
        
        for result in obstacle_results:
            summary_text += f"Condition: {result['condition_name']} ({result['condition']})\n"
            summary_text += f"  Pandolf - No obstacles: {result['pandolf_no_obstacles']:.1f} W (n={result['n_no_obstacles']})\n"
            summary_text += f"  Pandolf - With obstacles: {result['pandolf_with_obstacles']:.1f} W (n={result['n_with_obstacles']})\n"
            summary_text += f"  Pandolf change: {result['pandolf_percent_increase']:+.1f}%\n"
            
            if 'hopscotch_no_obs' in result:
                summary_text += f"  Hopscotch - No obstacles: {result['hopscotch_no_obs']:.1f} W\n"
                summary_text += f"  Hopscotch - With obstacles: {result['hopscotch_with_obs']:.1f} W\n"
                summary_text += f"  Hopscotch change: {result['hopscotch_pct']:+.1f}%\n"
            
            summary_text += "\n"
        
        with open(output_dir / 'obstacle_analysis.txt', 'w') as f:
            f.write(summary_text)
    
    return obstacle_results

def main():
    """Main analysis function."""
    try:
        analyze_by_condition_and_obstacles()
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
