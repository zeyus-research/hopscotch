"""
Script to create relational visualizations from DTW results.

Creates heatmaps, MDS plots, and network graphs to show the structure
of DTW distances between participants and conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

# Set style
sns.set_style("white")
plt.rcParams['figure.figsize'] = (15, 10)

def load_dtw_data(filepath):
    """Load DTW results from TSV file and exclude trials with missing data."""
    df = pd.read_csv(filepath, sep='\t')

    # Define exclusion criteria: (subject, condition, obstacles)
    exclusions = [
        (92, 'h', 1),
        (76, 'k', 1),
        (60, 'k', 1),
        (74, 'k', 1)
    ]

    # Create exclusion mask
    exclude_mask = pd.Series([False] * len(df))

    for subj, cond, obs in exclusions:
        mask1 = (
            (df['subject_1'] == subj) &
            (df['condition_1'] == cond) &
            (df['obstacles_1'] == obs)
        )
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

def create_condition_label(cond, obs):
    """Create a readable label for condition and obstacle combination."""
    cond_map = {'k': 'Control', 'h': 'Fast', 's': 'Fun'}
    obs_str = 'No-obs' if obs == 0 else '1-obs'
    return f"{cond_map[cond]}-{obs_str}"

def create_heatmaps_between(df, output_dir='hopscotch_results'):
    """Create heatmaps showing between-participant DTW distances for each condition."""
    between_df = df[df['dtw_type'] == 'between'].copy()

    # Get all condition-obstacle combinations
    conditions = [
        ('k', 0, 'Control, No obstacles'),
        ('k', 1, 'Control, 1 obstacle'),
        ('h', 0, 'Fast, No obstacles'),
        ('h', 1, 'Fast, 1 obstacle'),
        ('s', 0, 'Fun, No obstacles'),
        ('s', 1, 'Fun, 1 obstacle'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Between-Participant DTW Distance Heatmaps', fontsize=16, fontweight='bold')

    for idx, (cond, obs, title) in enumerate(conditions):
        ax = axes[idx // 3, idx % 3]

        # Filter for this condition
        mask = (
            (between_df['condition_1'] == cond) &
            (between_df['obstacles_1'] == obs)
        )
        subset = between_df[mask].copy()

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Get unique subjects
        subjects = sorted(pd.concat([subset['subject_1'], subset['subject_2']]).unique())

        # Create distance matrix
        n_subjects = len(subjects)
        dist_matrix = np.zeros((n_subjects, n_subjects))

        for _, row in subset.iterrows():
            i = subjects.index(row['subject_1'])
            j = subjects.index(row['subject_2'])
            dist_matrix[i, j] = row['dtw_distance']
            dist_matrix[j, i] = row['dtw_distance']

        # Plot heatmap
        sns.heatmap(dist_matrix, ax=ax, cmap='viridis',
                   xticklabels=[str(int(s)) for s in subjects],
                   yticklabels=[str(int(s)) for s in subjects],
                   cbar_kws={'label': 'DTW Distance'},
                   square=True, linewidths=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Participant ID')
        ax.set_ylabel('Participant ID')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / 'dtw_heatmaps_between_participants.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved between-participant heatmaps to: {output_file}")
    plt.close()

def create_mds_plot_between(df, output_dir='hopscotch_results'):
    """Create MDS plot showing relative positions of participants for each condition."""
    between_df = df[df['dtw_type'] == 'between'].copy()

    conditions = [
        ('k', 0), ('k', 1),
        ('h', 0), ('h', 1),
        ('s', 0), ('s', 1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('MDS: Between-Participant Spatial Relationships', fontsize=16, fontweight='bold')

    for idx, (cond, obs) in enumerate(conditions):
        ax = axes[idx // 3, idx % 3]

        # Filter for this condition
        mask = (
            (between_df['condition_1'] == cond) &
            (between_df['obstacles_1'] == obs)
        )
        subset = between_df[mask].copy()

        title = create_condition_label(cond, obs)

        if len(subset) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Get unique subjects
        subjects = sorted(pd.concat([subset['subject_1'], subset['subject_2']]).unique())
        n_subjects = len(subjects)

        # Create distance matrix
        dist_matrix = np.zeros((n_subjects, n_subjects))
        for _, row in subset.iterrows():
            i = subjects.index(row['subject_1'])
            j = subjects.index(row['subject_2'])
            dist_matrix[i, j] = row['dtw_distance']
            dist_matrix[j, i] = row['dtw_distance']

        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(dist_matrix)

        # Plot
        ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6, c=range(n_subjects), cmap='tab20')

        # Add labels
        for i, subj in enumerate(subjects):
            ax.annotate(str(int(subj)), (coords[i, 0], coords[i, 1]),
                       fontsize=8, ha='center', va='center')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

        # Add stress info
        ax.text(0.02, 0.98, f'Stress: {mds.stress_:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_file = output_path / 'dtw_mds_between_participants.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved between-participant MDS plots to: {output_file}")
    plt.close()

def create_within_heatmap(df, output_dir='hopscotch_results'):
    """Create heatmap showing within-participant comparisons across conditions."""
    within_df = df[df['dtw_type'] == 'within'].copy()

    # Get all participants
    participants = sorted(within_df['subject_1'].unique())

    # Define condition pairs
    condition_pairs = [
        ('h', 0, 'k', 0, 'H0-K0'),
        ('h', 0, 's', 0, 'H0-S0'),
        ('k', 0, 's', 0, 'K0-S0'),
        ('h', 1, 'k', 1, 'H1-K1'),
        ('h', 1, 's', 1, 'H1-S1'),
        ('k', 1, 's', 1, 'K1-S1'),
        ('h', 0, 'h', 1, 'H0-H1'),
        ('k', 0, 'k', 1, 'K0-K1'),
        ('s', 0, 's', 1, 'S0-S1'),
    ]

    # Create matrix: participants x condition pairs
    data_matrix = np.zeros((len(participants), len(condition_pairs)))

    for p_idx, participant in enumerate(participants):
        for c_idx, (c1, o1, c2, o2, label) in enumerate(condition_pairs):
            mask = (
                (
                    (within_df['subject_1'] == participant) &
                    (within_df['condition_1'] == c1) &
                    (within_df['obstacles_1'] == o1) &
                    (within_df['condition_2'] == c2) &
                    (within_df['obstacles_2'] == o2)
                ) |
                (
                    (within_df['subject_1'] == participant) &
                    (within_df['condition_1'] == c2) &
                    (within_df['obstacles_1'] == o2) &
                    (within_df['condition_2'] == c1) &
                    (within_df['obstacles_2'] == o1)
                )
            )
            values = within_df[mask]['dtw_distance']
            if len(values) > 0:
                data_matrix[p_idx, c_idx] = values.mean()
            else:
                data_matrix[p_idx, c_idx] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    sns.heatmap(data_matrix.T, ax=ax, cmap='viridis',
               xticklabels=[str(int(p)) for p in participants],
               yticklabels=[cp[4] for cp in condition_pairs],
               cbar_kws={'label': 'DTW Distance'},
               linewidths=0.5)

    ax.set_title('Within-Participant DTW Distances: Condition Comparisons by Participant',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Participant ID', fontsize=12)
    ax.set_ylabel('Condition Comparison', fontsize=12)

    # Add vertical separators
    ax.axhline(y=3, color='white', linewidth=3)
    ax.axhline(y=6, color='white', linewidth=3)

    # Add group labels
    ax.text(-1, 1.5, 'No obs', rotation=0, ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(-1, 4.5, '1 obs', rotation=0, ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(-1, 7.5, 'Obs effect', rotation=0, ha='right', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_path = Path(output_dir)
    output_file = output_path / 'dtw_heatmap_within_participants.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved within-participant heatmap to: {output_file}")
    plt.close()

def create_condition_network(df, output_dir='hopscotch_results'):
    """Create network graph showing relationships between participants in different conditions."""
    between_df = df[df['dtw_type'] == 'between'].copy()

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Network Graphs: Between-Participant Relationships (edge thickness = similarity)',
                fontsize=16, fontweight='bold')

    conditions = [
        ('k', 0, 'Control, No obstacles'),
        ('k', 1, 'Control, 1 obstacle'),
        ('h', 0, 'Fast, No obstacles'),
        ('h', 1, 'Fast, 1 obstacle'),
        ('s', 0, 'Fun, No obstacles'),
        ('s', 1, 'Fun, 1 obstacle'),
    ]

    for idx, (cond, obs, title) in enumerate(conditions):
        ax = axes[idx // 3, idx % 3]

        # Filter for this condition
        mask = (
            (between_df['condition_1'] == cond) &
            (between_df['obstacles_1'] == obs)
        )
        subset = between_df[mask].copy()

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
            continue

        # Create network
        G = nx.Graph()

        # Add edges with weights (inverse of distance for similarity)
        max_dist = subset['dtw_distance'].max()
        for _, row in subset.iterrows():
            s1, s2 = int(row['subject_1']), int(row['subject_2'])
            # Convert distance to similarity (higher = more similar)
            similarity = 1 / (1 + row['dtw_distance'] / max_dist)
            G.add_edge(s1, s2, weight=similarity, distance=row['dtw_distance'])

        # Layout
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

        # Draw network
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300,
                                      node_color='lightblue', alpha=0.7)

        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        for (u, v), weight in zip(edges, weights):
            nx.draw_networkx_edges(G, pos, [(u, v)], ax=ax,
                                  width=weight/max_weight * 5,
                                  alpha=0.5, edge_color='gray')

        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add info
        avg_dist = subset['dtw_distance'].mean()
        ax.text(0.02, 0.98, f'Avg dist: {avg_dist:.0f}\nParticipants: {len(G.nodes())}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_file = output_path / 'dtw_network_between_participants.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved between-participant network graphs to: {output_file}")
    plt.close()

def main():
    """Main function to generate all visualizations."""
    # Load data
    dtw_file = 'hopscotch_results/dtw_results_20250822_154121.tsv'
    print(f"Loading DTW data from: {dtw_file}")
    df = load_dtw_data(dtw_file)
    print(f"Loaded {len(df)} rows")
    print(f"DTW types: {df['dtw_type'].value_counts().to_dict()}")

    print("\n" + "="*60)
    print("Creating relational visualizations...")
    print("="*60)

    # Create heatmaps
    print("\n1. Creating between-participant heatmaps...")
    create_heatmaps_between(df)

    # Create MDS plots
    print("\n2. Creating between-participant MDS plots...")
    create_mds_plot_between(df)

    # Create within-participant heatmap
    print("\n3. Creating within-participant heatmap...")
    create_within_heatmap(df)

    # Create network graphs
    print("\n4. Creating network graphs...")
    create_condition_network(df)

    print("\n" + "="*60)
    print("✓ All relational visualizations generated successfully!")
    print("="*60)

if __name__ == '__main__':
    main()
