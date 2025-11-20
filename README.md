# Hopscotch Motion Capture Analysis

**Dataset**: https://osf.io/kvq3p/

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd hopscotch
```

2. **Install dependencies using uv:**
```bash
uv sync
```

This will create a virtual environment and install all required dependencies including:
- pandas, numpy, scipy for data processing
- matplotlib, seaborn for visualization
- c3d, ezc3d for C3D file support
- PyYAML for configuration files

## Quick Start

### Basic Usage

```python
from src.energy_pipeline import EnergyAnalysisPipeline
from pathlib import Path

# Initialize pipeline
pipeline = EnergyAnalysisPipeline()

# Run complete analysis
results = pipeline.run_full_pipeline(
    data_path=Path("data"),
    output_dir=Path("results"),
    file_pattern="*.tsv"  # or "*.c3d" for C3D files
)

print(f"Processed {results['total_trials']} trials")
print(f"Mean metabolic rate: {results['mean_metabolic_rate']:.1f} W")
```

### Step-by-Step Usage

```python
# 1. Load data
pipeline = EnergyAnalysisPipeline()
raw_data = pipeline.load_data(Path("data"), "*.tsv")

# 2. Calculate energy metrics
body_masses = {'23': 75.0, '24': 68.0}  # subject_id: mass_kg
energy_metrics = pipeline.calculate_energy_metrics(body_masses)

# 3. Run group analysis
group_analyzer = pipeline.run_group_analysis()

# 4. Get results
condition_summary = group_analyzer.get_condition_summary()
comparison_results = group_analyzer.compare_conditions('metabolic_rate_pandolf')

# 5. Save results
pipeline.save_results(Path("results"))
```

## Configuration

### Configuration File

Create a `config.yaml` file to customize the analysis:

```yaml
# Marker and sampling configuration
marker_labels_file: 'marker_labels.txt'
sampling_rate: 100.0
default_body_mass: 70.0

# Data processing
cache_file: 'mocap_data.feather'
log_level: 'INFO'

# Statistical analysis
test_type: 'independent'  # or 'paired'

# Subject-specific body masses (kg)
body_masses:
  '23': 75.0
  '24': 68.0
  '25': 72.0

# Custom subject/condition mapping (optional)
subject_config:
  'custom_file.tsv': 
    subject: 'S01'
    condition: 'A'
    session: '1'
```

Use the configuration:

```python
pipeline = EnergyAnalysisPipeline(config_file=Path("config.yaml"))
```

### Marker Labels

The pipeline expects a `marker_labels.txt` file with marker names:

```text
head
foot_front_r
foot_back_r
knee_under_r
knee_over_r
wrist_r
elbow_r
shoulder_r
hip_front_r
hip_back_r
foot_back_l
foot_front_l
knee_under_l
knee_over_l
hip_front_l
hip_back_l
wrist_l
elbow_l
shoulder_l
floor_start_l
floor_start_r
floor_end_r
floor_end_l
```

## Data Format Requirements

### TSV Files

- Tab-separated values with headers
- Columns: `Frame`, `Time`, `marker_name.X`, `marker_name.Y`, `marker_name.Z`
- Filename format: `{subject}_{condition}{session}.tsv` (e.g., `23_h0.tsv`)

### C3D Files

- Standard C3D format with point data
- Supported via `c3d` and `ezc3d` libraries
- Filename format: `{subject}_{condition}{session}.c3d`

### Subject/Condition Extraction

The pipeline automatically extracts metadata from filenames:

| Filename | Subject | Condition | Session |
|----------|---------|-----------|---------|
| `23_h0.tsv` | 23 | h | 0 |
| `24_s1.c3d` | 24 | s | 1 |
| `01_k2.tsv` | 01 | k | 2 |


## License

[MIT License](./LICENSE)
