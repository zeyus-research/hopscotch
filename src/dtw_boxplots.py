"""
Script to create box plots from DTW results.

Between-condition plots: Compare different conditions across all participants
Within-participant plots: Show all condition/obstacle combinations for each participant
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_dtw_data(filepath):
    """Load DTW results from TSV file and exclude trials with missing data."""
    df = pd.read_csv(filepath, sep='\t')

    # Define exclusion criteria: (subject, condition, obstacles)
    # These trials have missing data and should be excluded
    exclusions = [
        (92, 'h', 1),
        (76, 'k', 1),
        (60, 'k', 1),  # Note: K -> k (lowercase)
        (74, 'k', 1)
    ]

    # Create exclusion mask
    # Exclude rows where either subject_1 or subject_2 matches exclusion criteria
    exclude_mask = pd.Series([False] * len(df))

    for subj, cond, obs in exclusions:
        # Check if this trial appears as subject_1
        mask1 = (
            (df['subject_1'] == subj) &
            (df['condition_1'] == cond) &
            (df['obstacles_1'] == obs)
        )
        # Check if this trial appears as subject_2
        mask2 = (
            (df['subject_2'] == subj) &
            (df['condition_2'] == cond) &
            (df['obstacles_2'] == obs)
        )
        exclude_mask = exclude_mask | mask1 | mask2

    # Apply exclusion
    n_before = len(df)
    df = df[~exclude_mask].copy()
    n_after = len(df)
    n_excluded = n_before - n_after

    print(f"Excluded {n_excluded} rows with missing data")
    print(f"Remaining rows: {n_after}")

    return df

def create_between_condition_boxplots(df, output_dir='analysis'):
    """
    Create box plots for between-condition comparisons.

    6 subplots comparing:
    - K0 -> H0 (control no obstacles vs fast no obstacles)
    - K0 -> S0 (control no obstacles vs fun no obstacles)
    - H0 -> S0 (fast no obstacles vs fun no obstacles)
    - K1 -> H1 (control 1 obstacle vs fast 1 obstacle)
    - K1 -> S1 (control 1 obstacle vs fun 1 obstacle)
    - H1 -> S1 (fast 1 obstacle vs fun 1 obstacle)
    """
    # Filter for within-participant comparisons (where conditions differ)
    within_df = df[df['dtw_type'] == 'within'].copy()

    # Define condition mapping
    condition_map = {'k': 'Control (K)', 'h': 'Fast (H)', 's': 'Fun (S)'}

    # Get global y-axis limits from all within data
    y_min = within_df['dtw_distance'].min()
    y_max = within_df['dtw_distance'].max()
    y_range = y_max - y_min
    y_lim = [y_min - 0.05 * y_range, y_max + 0.05 * y_range]

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Between-Condition DTW Distance Comparisons (Across All Participants)',
                 fontsize=16, fontweight='bold')

    # Define the comparisons
    comparisons = [
        # No obstacles (row 0)
        {'cond1': 'k', 'cond2': 'h', 'obs': 0, 'label': 'K0 → H0', 'pos': (0, 0)},
        {'cond1': 'k', 'cond2': 's', 'obs': 0, 'label': 'K0 → S0', 'pos': (0, 1)},
        {'cond1': 'h', 'cond2': 's', 'obs': 0, 'label': 'H0 → S0', 'pos': (0, 2)},
        # 1 obstacle (row 1)
        {'cond1': 'k', 'cond2': 'h', 'obs': 1, 'label': 'K1 → H1', 'pos': (1, 0)},
        {'cond1': 'k', 'cond2': 's', 'obs': 1, 'label': 'K1 → S1', 'pos': (1, 1)},
        {'cond1': 'h', 'cond2': 's', 'obs': 1, 'label': 'H1 → S1', 'pos': (1, 2)},
    ]

    for comp in comparisons:
        # Filter data for this comparison (both orderings)
        mask = (
            (
                (within_df['condition_1'] == comp['cond1']) &
                (within_df['condition_2'] == comp['cond2']) &
                (within_df['obstacles_1'] == comp['obs']) &
                (within_df['obstacles_2'] == comp['obs'])
            ) |
            (
                (within_df['condition_1'] == comp['cond2']) &
                (within_df['condition_2'] == comp['cond1']) &
                (within_df['obstacles_1'] == comp['obs']) &
                (within_df['obstacles_2'] == comp['obs'])
            )
        )

        data = within_df[mask]['dtw_distance']

        ax = axes[comp['pos']]

        # Create box plot
        bp = ax.boxplot([data], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        # Add individual points
        y = data.values
        x = [1] * len(y)
        ax.scatter(x, y, alpha=0.4, s=30, color='darkblue')

        # Set title and labels
        c1_name = condition_map[comp['cond1']]
        c2_name = condition_map[comp['cond2']]
        obs_str = 'No obstacles' if comp['obs'] == 0 else '1 obstacle'
        ax.set_title(f"{comp['label']}: {c1_name} vs {c2_name}\n({obs_str})",
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('DTW Distance', fontsize=10)
        ax.set_xticks([1])
        ax.set_xticklabels([comp['label']])
        ax.set_ylim(y_lim)

        # Add statistics
        ax.text(0.02, 0.98, f"n = {len(data)}\nMean = {data.mean():.1f}\nMedian = {data.median():.1f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'dtw_between_condition_boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved between-condition box plots to: {output_file}")
    plt.close()

def create_between_participant_combined_boxplot(df, output_dir='analysis'):
    """
    Create a single box plot showing between-participant comparisons
    grouped by condition and obstacles (6 groups on x-axis).
    """
    # Filter for between-participant comparisons
    between_df = df[df['dtw_type'] == 'between'].copy()

    # Define condition mapping
    condition_map = {'k': 'Control', 'h': 'Fast', 's': 'Fun'}

    # Get global y-axis limits
    y_min = between_df['dtw_distance'].min()
    y_max = between_df['dtw_distance'].max()
    y_range = y_max - y_min
    y_lim = [y_min - 0.05 * y_range, y_max + 0.05 * y_range]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Between-Participant DTW Distance by Condition and Obstacles',
                 fontsize=16, fontweight='bold')

    # Define the groups
    groups = [
        {'cond': 'k', 'obs': 0, 'label': 'K0\nControl\nNo obstacles'},
        {'cond': 'k', 'obs': 1, 'label': 'K1\nControl\n1 obstacle'},
        {'cond': 'h', 'obs': 0, 'label': 'H0\nFast\nNo obstacles'},
        {'cond': 'h', 'obs': 1, 'label': 'H1\nFast\n1 obstacle'},
        {'cond': 's', 'obs': 0, 'label': 'S0\nFun\nNo obstacles'},
        {'cond': 's', 'obs': 1, 'label': 'S1\nFun\n1 obstacle'},
    ]

    # Collect data for each group
    data_by_group = []
    labels = []
    colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred', 'lightgreen', 'darkgreen']

    for group in groups:
        # Filter for this condition and obstacle combination
        mask = (
            (between_df['condition_1'] == group['cond']) &
            (between_df['obstacles_1'] == group['obs'])
        )
        data = between_df[mask]['dtw_distance']
        data_by_group.append(data.values)
        labels.append(group['label'])

    # Create box plots
    bp = ax.boxplot(data_by_group, widths=0.6, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points with jitter
    for i, data in enumerate(data_by_group):
        y = data
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.3, s=20, color='black')

    # Add mean line for each participant
    # Get all unique participants
    all_subjects = pd.concat([between_df['subject_1'], between_df['subject_2']]).unique()

    for subject in all_subjects:
        participant_means = []
        for group in groups:
            # Get data where this participant appears with this condition/obstacle combo
            mask = (
                ((between_df['subject_1'] == subject) &
                 (between_df['condition_1'] == group['cond']) &
                 (between_df['obstacles_1'] == group['obs'])) |
                ((between_df['subject_2'] == subject) &
                 (between_df['condition_2'] == group['cond']) &
                 (between_df['obstacles_2'] == group['obs']))
            )
            subject_data = between_df[mask]['dtw_distance']
            if len(subject_data) > 0:
                participant_means.append(subject_data.mean())
            else:
                participant_means.append(np.nan)

        # Only plot if participant has data for all groups
        if not all(np.isnan(participant_means)):
            ax.plot(range(1, len(groups)+1), participant_means, '-',
                   linewidth=1, alpha=0.35, color='gray', zorder=5)

    # Add overall mean line connecting all groups
    means = [data.mean() for data in data_by_group]
    ax.plot(range(1, len(groups)+1), means, 'ro-', linewidth=2.5,
            markersize=8, label='Overall Mean', zorder=10)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', marker='o', linewidth=2.5,
               markersize=8, label='Overall Mean'),
        Line2D([0], [0], color='gray', linewidth=1, alpha=0.35,
               label='Participant Means')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set labels and formatting
    ax.set_ylabel('DTW Distance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Condition and Obstacles', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, len(groups)+1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(y_lim)
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text box
    stats_text = "Statistics per group:\n"
    for i, (group, data) in enumerate(zip(groups, data_by_group)):
        stats_text += f"{group['label'].split()[0]}: n={len(data)}, mean={data.mean():.0f}, median={np.median(data):.0f}\n"

    ax.text(0.02, 0.98, stats_text.strip(),
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'dtw_between_participant_combined_boxplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved between-participant combined box plot to: {output_file}")
    plt.close()

def create_within_participant_boxplots(df, output_dir='analysis'):
    """
    Create box plots for within-participant comparisons.

    9 subplots grouped by condition_1, condition_2, obstacles_1, obstacles_2:
    - Different conditions, no obstacles: H0 vs K0, H0 vs S0, K0 vs S0
    - Different conditions, 1 obstacle: H1 vs K1, H1 vs S1, K1 vs S1
    - Same condition, different obstacles: H0 vs H1, K0 vs K1, S0 vs S1
    """
    # Filter for within-participant comparisons
    within_df = df[df['dtw_type'] == 'within'].copy()

    # Define condition mapping
    condition_map = {'k': 'Control (K)', 'h': 'Fast (H)', 's': 'Fun (S)'}

    # Get global y-axis limits from all within data
    y_min = within_df['dtw_distance'].min()
    y_max = within_df['dtw_distance'].max()
    y_range = y_max - y_min
    y_lim = [y_min - 0.05 * y_range, y_max + 0.05 * y_range]

    # Create figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Within-Participant DTW Distance Comparisons (All Participants Combined)',
                 fontsize=16, fontweight='bold')

    # Define the comparisons - organized by type
    comparisons = [
        # Row 0: Different conditions, no obstacles
        {'cond1': 'h', 'cond2': 'k', 'obs1': 0, 'obs2': 0, 'label': 'H0 vs K0', 'pos': (0, 0)},
        {'cond1': 'h', 'cond2': 's', 'obs1': 0, 'obs2': 0, 'label': 'H0 vs S0', 'pos': (0, 1)},
        {'cond1': 'k', 'cond2': 's', 'obs1': 0, 'obs2': 0, 'label': 'K0 vs S0', 'pos': (0, 2)},
        # Row 1: Different conditions, 1 obstacle
        {'cond1': 'h', 'cond2': 'k', 'obs1': 1, 'obs2': 1, 'label': 'H1 vs K1', 'pos': (1, 0)},
        {'cond1': 'h', 'cond2': 's', 'obs1': 1, 'obs2': 1, 'label': 'H1 vs S1', 'pos': (1, 1)},
        {'cond1': 'k', 'cond2': 's', 'obs1': 1, 'obs2': 1, 'label': 'K1 vs S1', 'pos': (1, 2)},
        # Row 2: Same condition, different obstacles
        {'cond1': 'h', 'cond2': 'h', 'obs1': 0, 'obs2': 1, 'label': 'H0 vs H1', 'pos': (2, 0)},
        {'cond1': 'k', 'cond2': 'k', 'obs1': 0, 'obs2': 1, 'label': 'K0 vs K1', 'pos': (2, 1)},
        {'cond1': 's', 'cond2': 's', 'obs1': 0, 'obs2': 1, 'label': 'S0 vs S1', 'pos': (2, 2)},
    ]

    for comp in comparisons:
        # Filter data for this comparison (both orderings)
        mask = (
            (
                (within_df['condition_1'] == comp['cond1']) &
                (within_df['condition_2'] == comp['cond2']) &
                (within_df['obstacles_1'] == comp['obs1']) &
                (within_df['obstacles_2'] == comp['obs2'])
            ) |
            (
                (within_df['condition_1'] == comp['cond2']) &
                (within_df['condition_2'] == comp['cond1']) &
                (within_df['obstacles_1'] == comp['obs2']) &
                (within_df['obstacles_2'] == comp['obs1'])
            )
        )

        data = within_df[mask]['dtw_distance']

        ax = axes[comp['pos']]

        # Create box plot with all participants combined
        bp = ax.boxplot([data], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        # Add individual points
        y = data.values
        x = [1] * len(y)
        ax.scatter(x, y, alpha=0.4, s=30, color='darkgreen')

        # Set title and labels
        c1_name = condition_map[comp['cond1']]
        c2_name = condition_map[comp['cond2']]
        ax.set_title(f"{comp['label']}: {c1_name} vs {c2_name}",
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('DTW Distance', fontsize=10)
        ax.set_xticks([1])
        ax.set_xticklabels([comp['label']])
        ax.set_ylim(y_lim)

        # Add statistics
        n_participants = within_df[mask]['subject_1'].nunique()
        ax.text(0.02, 0.98,
                f"n = {len(data)}\nParticipants = {n_participants}\nMean = {data.mean():.1f}\nMedian = {data.median():.1f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'dtw_within_participant_boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved within-participant box plots to: {output_file}")
    plt.close()

def create_within_participant_combined_boxplot(df, output_dir='analysis'):
    """
    Create a single box plot showing within-participant comparisons
    grouped by condition combinations (9 groups on x-axis).
    """
    # Filter for within-participant comparisons
    within_df = df[df['dtw_type'] == 'within'].copy()

    # Define condition mapping
    condition_map = {'k': 'Control', 'h': 'Fast', 's': 'Fun'}

    # Get global y-axis limits
    y_min = within_df['dtw_distance'].min()
    y_max = within_df['dtw_distance'].max()
    y_range = y_max - y_min
    y_lim = [y_min - 0.05 * y_range, y_max + 0.05 * y_range]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.suptitle('Within-Participant DTW Distance by Condition Comparisons',
                 fontsize=16, fontweight='bold')

    # Define the groups - matching the subplot organization
    groups = [
        # Different conditions, no obstacles
        {'cond1': 'h', 'cond2': 'k', 'obs1': 0, 'obs2': 0, 'label': 'H0 vs K0'},
        {'cond1': 'h', 'cond2': 's', 'obs1': 0, 'obs2': 0, 'label': 'H0 vs S0'},
        {'cond1': 'k', 'cond2': 's', 'obs1': 0, 'obs2': 0, 'label': 'K0 vs S0'},
        # Different conditions, 1 obstacle
        {'cond1': 'h', 'cond2': 'k', 'obs1': 1, 'obs2': 1, 'label': 'H1 vs K1'},
        {'cond1': 'h', 'cond2': 's', 'obs1': 1, 'obs2': 1, 'label': 'H1 vs S1'},
        {'cond1': 'k', 'cond2': 's', 'obs1': 1, 'obs2': 1, 'label': 'K1 vs S1'},
        # Same condition, different obstacles
        {'cond1': 'h', 'cond2': 'h', 'obs1': 0, 'obs2': 1, 'label': 'H0 vs H1'},
        {'cond1': 'k', 'cond2': 'k', 'obs1': 0, 'obs2': 1, 'label': 'K0 vs K1'},
        {'cond1': 's', 'cond2': 's', 'obs1': 0, 'obs2': 1, 'label': 'S0 vs S1'},
    ]

    # Collect data for each group
    data_by_group = []
    labels = []
    colors = ['lightcoral', 'lightgreen', 'lightblue',
              'darkred', 'darkgreen', 'darkblue',
              'orange', 'purple', 'brown']

    for group in groups:
        # Filter for this comparison (both orderings)
        mask = (
            (
                (within_df['condition_1'] == group['cond1']) &
                (within_df['condition_2'] == group['cond2']) &
                (within_df['obstacles_1'] == group['obs1']) &
                (within_df['obstacles_2'] == group['obs2'])
            ) |
            (
                (within_df['condition_1'] == group['cond2']) &
                (within_df['condition_2'] == group['cond1']) &
                (within_df['obstacles_1'] == group['obs2']) &
                (within_df['obstacles_2'] == group['obs1'])
            )
        )
        data = within_df[mask]['dtw_distance']
        data_by_group.append(data.values)
        labels.append(group['label'])

    # Create box plots
    bp = ax.boxplot(data_by_group, widths=0.6, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Color each box differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points with jitter
    for i, data in enumerate(data_by_group):
        y = data
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.3, s=20, color='black')

    # Add mean line for each participant
    # Get all unique participants
    all_subjects = within_df['subject_1'].unique()

    for subject in all_subjects:
        participant_means = []
        for group in groups:
            # Get data where this participant appears with this condition/obstacle combo
            mask = (
                (
                    (within_df['subject_1'] == subject) &
                    (within_df['condition_1'] == group['cond1']) &
                    (within_df['condition_2'] == group['cond2']) &
                    (within_df['obstacles_1'] == group['obs1']) &
                    (within_df['obstacles_2'] == group['obs2'])
                ) |
                (
                    (within_df['subject_1'] == subject) &
                    (within_df['condition_1'] == group['cond2']) &
                    (within_df['condition_2'] == group['cond1']) &
                    (within_df['obstacles_1'] == group['obs2']) &
                    (within_df['obstacles_2'] == group['obs1'])
                )
            )
            subject_data = within_df[mask]['dtw_distance']
            if len(subject_data) > 0:
                participant_means.append(subject_data.mean())
            else:
                participant_means.append(np.nan)

        # Only plot if participant has data (may not have all groups)
        if not all(np.isnan(participant_means)):
            ax.plot(range(1, len(groups)+1), participant_means, '-',
                   linewidth=1, alpha=0.35, color='gray', zorder=5)

    # Add overall mean line connecting all groups
    means = [data.mean() for data in data_by_group]
    ax.plot(range(1, len(groups)+1), means, 'ro-', linewidth=2.5,
            markersize=8, label='Overall Mean', zorder=10)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', marker='o', linewidth=2.5,
               markersize=8, label='Overall Mean'),
        Line2D([0], [0], color='gray', linewidth=1, alpha=0.35,
               label='Participant Means')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set labels and formatting
    ax.set_ylabel('DTW Distance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Condition Comparisons', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, len(groups)+1))
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.set_ylim(y_lim)
    ax.grid(axis='y', alpha=0.3)

    # Add vertical lines to separate groups
    ax.axvline(x=3.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=6.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Add group labels
    ax.text(2, y_lim[1]*0.98, 'No obstacles', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(5, y_lim[1]*0.98, '1 obstacle', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(8, y_lim[1]*0.98, 'Obstacle effect', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add statistics text box
    stats_text = "Statistics per group:\n"
    for i, (group, data) in enumerate(zip(groups, data_by_group)):
        stats_text += f"{group['label']}: n={len(data)}, mean={data.mean():.0f}, median={np.median(data):.0f}\n"

    ax.text(0.02, 0.75, stats_text.strip(),
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=8, family='monospace')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'dtw_within_participant_combined_boxplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved within-participant combined box plot to: {output_file}")
    plt.close()

def main():
    """Main function to generate all plots."""
    # Load data
    dtw_file = 'hopscotch_results/dtw_joint_angles_results_20260123_151036.tsv'
    print(f"Loading DTW data from: {dtw_file}")
    df = load_dtw_data(dtw_file)
    print(f"Loaded {len(df)} rows")
    print(f"DTW types: {df['dtw_type'].value_counts().to_dict()}")

    # Create between-condition plots
    print("\nCreating between-condition box plots...")
    create_between_condition_boxplots(df)

    # Create between-participant combined plot
    print("\nCreating between-participant combined box plot...")
    create_between_participant_combined_boxplot(df)

    # Create within-participant plots
    print("\nCreating within-participant box plots...")
    create_within_participant_boxplots(df)

    # Create within-participant combined plot
    print("\nCreating within-participant combined box plot...")
    create_within_participant_combined_boxplot(df)

    print("\n✓ All plots generated successfully!")

if __name__ == '__main__':
    main()
