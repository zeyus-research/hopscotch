"""
Create network graphs from DTW results data.
Generates a condition network based on within-subject comparisons.
Saves visualizations to the analysis/network_graphs directory.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from netgraph import Graph

# Load the data
data_path = "analysis/dtw_results_20250822_154121.tsv"
df = pd.read_csv(data_path, sep='\t')

print(f"Total comparisons: {len(df)}")
print(f"\nData columns: {df.columns.tolist()}")
print(f"\nDTW types: {df['dtw_type'].value_counts()}")
print(f"\nSubjects: {sorted(df['subject_1'].unique())}")
print(f"\nConditions: {sorted(df['condition_1'].unique())}")
print(f"\nObstacles: {sorted(df['obstacles_1'].unique())}")

# Summary statistics
print(f"\n--- DTW Distance Statistics ---")
print(f"Overall mean: {df['dtw_distance'].mean():.2f}")
print(f"Overall std: {df['dtw_distance'].std():.2f}")
print(f"\nBy type:")
print(df.groupby('dtw_type')['dtw_distance'].describe())

# Create output directory
output_dir = Path("analysis/network_graphs")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("CREATING NETWORK GRAPHS")
print("="*80)


# ============================================================================
# CONDITION NETWORK (Within-subject comparisons)
# ============================================================================
print("\nCreating Condition Network...")
cmap = plt.colormaps.get_cmap('RdYlGn_r')

# Filter within-subject comparisons
within_df = df[df['dtw_type'] == 'within'].copy()

# Create condition-obstacle labels
def make_condition_label(cond, obs):
    cond_map = {'h': 'Ext', 'k': 'Control', 's': 'Int'}
    obs_map = {0: '⁻', 1: '⁺'}
    return f"{cond_map.get(cond, cond)}{obs_map.get(obs, obs)}"

# Create a graph where nodes are condition-obstacle combinations
G_conditions = nx.Graph()

# Add nodes for each condition-obstacle combination
conditions = []
for cond in ['h', 'k', 's']:
    for obs in [0, 1]:
        label = make_condition_label(cond, obs)
        conditions.append((cond, obs, label))
        G_conditions.add_node(label, condition=cond, obstacle=obs)

print(f"   Nodes (condition-obstacle pairs): {G_conditions.number_of_nodes()}")

# Add edges with DTW distance as weight (average across all subjects)
edge_data_cond = []
for _, row in within_df.iterrows():
    c1, o1 = row['condition_1'], row['obstacles_1']
    c2, o2 = row['condition_2'], row['obstacles_2']
    label1 = make_condition_label(c1, o1)
    label2 = make_condition_label(c2, o2)
    dist = row['dtw_distance']
    edge_data_cond.append({'c1': label1, 'c2': label2, 'distance': dist})

edge_cond_df = pd.DataFrame(edge_data_cond)
# Average DTW distance for each condition pair
edge_cond_summary = edge_cond_df.groupby(['c1', 'c2'])['distance'].mean().reset_index()

for _, row in edge_cond_summary.iterrows():
    G_conditions.add_edge(row['c1'], row['c2'], weight=row['distance'])

print(f"   Edges (condition pair comparisons): {G_conditions.number_of_edges()}")

# Visualize condition network
fig, ax = plt.subplots(figsize=(12, 12))

# Get edge weights for visualization
edges_cond = G_conditions.edges()
weights_cond = [G_conditions[u][v]['weight'] for u, v in edges_cond]

# Normalize weights for edge thickness (inverse: smaller distance = thicker edge)
max_weight_cond = max(weights_cond)
min_weight_cond = min(weights_cond)
edge_widths_cond = {(u, v): 5 * (1 - (w - min_weight_cond) / (max_weight_cond - min_weight_cond)) + 1
                    for (u, v), w in zip(edges_cond, weights_cond)}

# Edge colors based on distance - manually apply colormap
norm_cond = plt.Normalize(vmin=min_weight_cond, vmax=max_weight_cond)
edge_colors_cond = {(u, v): cmap(norm_cond(w)) for (u, v), w in zip(edges_cond, weights_cond)}

# Node colors based on condition type
node_colors_cond = {}
node_shapes_cond = {}
# {'Ext⁻': 's', 'Int⁻': '>', 'Control⁻': 'o', 'Control⁺': 'o', 'Ext⁺': 's', 'Int⁺': '>'}
node_proxy_artists = []
for node in G_conditions.nodes():
    node_data = G_conditions.nodes[node]
    label = ''
    if node_data['condition'] == 'h':
        node_shapes_cond[node] = 's'  # square for external
        if node_data['obstacle'] == 0:
            node_colors_cond[node] = '#fa7a6b'
            label = 'External Motivation'
        else:
            node_colors_cond[node] = '#fba69d'
    elif node_data['condition'] == 'k':
        node_shapes_cond[node] = 'o'  # circle for control
        if node_data['obstacle'] == 0:
            node_colors_cond[node] = '#7bea7b'
            label = 'Control'
        else:
            node_colors_cond[node] = '#a7f1a7'
    else:
        node_shapes_cond[node] = '>'  # triangle for internal
        if node_data['obstacle'] == 0:
            node_colors_cond[node] = '#7ccae9'
            label = 'Internal Motivation'
        else:
            node_colors_cond[node] = '#a8dcf0'
    if not label == '':
        node_proxy_artists.append(
            plt.Line2D(
                [],
                [],
                linestyle='None',
                marker=node_shapes_cond[node],
                color='w',
                markerfacecolor=node_colors_cond[node],
                markersize=10,
                label=label
            )
        )

# Use edge weights as distances for layout
# Scale edge weights to reasonable range for visualization (0.1 to 0.9)
max_dist_cond = max(weights_cond)
min_dist_cond = min(weights_cond)
edge_layout_cond = {(u, v): 0.2 + 0.7 * (G_conditions[u][v]['weight'] - min_dist_cond) / (max_dist_cond - min_dist_cond)
                    for u, v in G_conditions.edges()}

# Edge labels with distances
edge_labels_cond = {(u, v): f"{G_conditions[u][v]['weight']:.0f}" for u, v in G_conditions.edges()}

# Create the netgraph visualization with larger scale to reduce overlap
plot_instance = Graph(
    G_conditions,
    node_layout='geometric',
    node_layout_kwargs=dict(edge_length=edge_layout_cond, tol=1e-6),
    scale=(4, 4),
    node_color=node_colors_cond,
    node_size=15,
    node_edge_width=0.5,
    node_shape=node_shapes_cond,
    node_labels=True,
    node_label_fontdict=dict(size=9, fontweight='bold'),
    edge_width=edge_widths_cond,
    edge_alpha=0.7,
    edge_labels=edge_labels_cond,
    edge_label_fontdict=dict(size=8),
    ax=ax,
    seed=1
)

ax.set_title('Condition Network\n(Within-subject comparisons, averaged across all subjects)',
             fontsize=14, fontweight='bold', pad=20)

ax.legend(handles=node_proxy_artists, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'condition_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'condition_network.png'}")
plt.close()

# ============================================================================
# HIERARCHICAL CLUSTERING (All comparisons)
# ============================================================================
print("\nCreating Hierarchical Clustering Visualization...")

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Create a combined label for each subject-condition-obstacle combination
def make_full_label(subj, cond, obs):
    cond_map = {'h': 'Ext', 'k': 'Ctrl', 's': 'Int'}
    obs_map = {0: '⁻', 1: '⁺'}
    return f"S{subj}_{cond_map.get(cond, cond)}{obs_map.get(obs, obs)}"

# Get all unique subject-condition-obstacle combinations
all_combos = []
for subject in sorted(df['subject_1'].unique()):
    for condition in ['h', 'k', 's']:
        for obstacle in [0, 1]:
            combo = (subject, condition, obstacle)
            all_combos.append(combo)

n_combos = len(all_combos)
print(f"   Total combinations (subject × condition × obstacle): {n_combos}")

# Create distance matrix
# Initialize with zeros
distance_matrix = np.zeros((n_combos, n_combos))

# Create mapping from (subject, condition, obstacle) to index
combo_to_idx = {combo: i for i, combo in enumerate(all_combos)}

# Fill in distances from within-subject comparisons
within_df = df[df['dtw_type'] == 'within'].copy()
for _, row in within_df.iterrows():
    combo1 = (row['subject_1'], row['condition_1'], row['obstacles_1'])
    combo2 = (row['subject_2'], row['condition_2'], row['obstacles_2'])

    if combo1 in combo_to_idx and combo2 in combo_to_idx:
        i, j = combo_to_idx[combo1], combo_to_idx[combo2]
        distance_matrix[i, j] = row['dtw_distance']
        distance_matrix[j, i] = row['dtw_distance']  # symmetric

# Fill in distances from between-subject comparisons
between_df = df[df['dtw_type'] == 'between'].copy()
for _, row in between_df.iterrows():
    combo1 = (row['subject_1'], row['condition_1'], row['obstacles_1'])
    combo2 = (row['subject_2'], row['condition_2'], row['obstacles_2'])

    if combo1 in combo_to_idx and combo2 in combo_to_idx:
        i, j = combo_to_idx[combo1], combo_to_idx[combo2]
        distance_matrix[i, j] = row['dtw_distance']
        distance_matrix[j, i] = row['dtw_distance']  # symmetric

# Convert to condensed distance matrix for hierarchical clustering
condensed_dist = squareform(distance_matrix, checks=False)

# Perform hierarchical clustering using average linkage
Z = linkage(condensed_dist, method='average')

# Create labels for dendrogram
labels = [make_full_label(s, c, o) for s, c, o in all_combos]

# Create color mapping for subjects
subject_colors = {}
color_palette = plt.cm.tab20(np.linspace(0, 1, len(sorted(df['subject_1'].unique()))))
for i, subj in enumerate(sorted(df['subject_1'].unique())):
    subject_colors[subj] = color_palette[i]

# Create label colors based on subject
label_colors = [subject_colors[combo[0]] for combo in all_combos]

# Plot dendrogram
fig, ax = plt.subplots(figsize=(20, 12))

dendro = dendrogram(
    Z,
    labels=labels,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=6,
    color_threshold=0.7*max(Z[:, 2])  # Color threshold for visual distinction
)

ax.set_title('Hierarchical Clustering of All Subject-Condition-Obstacle Combinations\n' +
             '(Lower in tree = more similar movement patterns)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Subject_Condition', fontsize=12)
ax.set_ylabel('DTW Distance', fontsize=12)

# Color the x-axis labels by subject
xlabels = ax.get_xticklabels()
label_positions = dendro['leaves']
for i, (label, pos) in enumerate(zip(xlabels, label_positions)):
    subject_id = all_combos[pos][0]
    label.set_color(subject_colors[subject_id])

plt.tight_layout()
plt.savefig(output_dir / 'hierarchical_clustering.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'hierarchical_clustering.png'}")
plt.close()

# ============================================================================
# SUBJECT NETWORK (Between-subject comparisons)
# ============================================================================
print("\nCreating Subject Network...")

# Filter between-subject comparisons
between_df = df[df['dtw_type'] == 'between'].copy()

# Create a graph where nodes are subjects
G_subjects = nx.Graph()

# Add nodes for each subject
subjects = sorted(df['subject_1'].unique())
for subject in subjects:
    G_subjects.add_node(subject)

print(f"   Nodes (subjects): {G_subjects.number_of_nodes()}")

# Add edges with average DTW distance across all matched conditions
edge_data_subj = []
for _, row in between_df.iterrows():
    s1, s2 = row['subject_1'], row['subject_2']
    dist = row['dtw_distance']
    edge_data_subj.append({'s1': s1, 's2': s2, 'distance': dist})

edge_subj_df = pd.DataFrame(edge_data_subj)
# Average DTW distance for each subject pair (across all condition-obstacle combinations)
edge_subj_summary = edge_subj_df.groupby(['s1', 's2'])['distance'].agg(['mean', 'std', 'count']).reset_index()

# Filter to keep only the most similar connections (bottom 25th percentile of distances)
distance_threshold = edge_subj_summary['mean'].quantile(0.25)
edge_subj_filtered = edge_subj_summary[edge_subj_summary['mean'] <= distance_threshold].copy()

print(f"   Distance threshold (25th percentile): {distance_threshold:.0f}")
print(f"   Edges before filtering: {len(edge_subj_summary)}")
print(f"   Edges after filtering: {len(edge_subj_filtered)}")

for _, row in edge_subj_filtered.iterrows():
    G_subjects.add_edge(row['s1'], row['s2'],
                       weight=row['mean'],
                       std=row['std'],
                       count=row['count'])

print(f"   Edges in network (most similar subjects): {G_subjects.number_of_edges()}")
print(f"   Average comparisons per edge: {edge_subj_filtered['count'].mean():.1f}")

# Visualize subject network
fig, ax = plt.subplots(figsize=(16, 16))

# Get edge weights for visualization
edges_subj = G_subjects.edges()
weights_subj = [G_subjects[u][v]['weight'] for u, v in edges_subj]

if len(weights_subj) > 0:
    # Normalize weights for edge thickness (inverse: smaller distance = thicker edge)
    max_weight_subj = max(weights_subj)
    min_weight_subj = min(weights_subj)
    edge_widths_subj = {(u, v): 5 * (1 - (w - min_weight_subj) / (max_weight_subj - min_weight_subj)) + 2
                        for (u, v), w in zip(edges_subj, weights_subj)}

    # Edge colors based on distance
    norm_subj = plt.Normalize(vmin=min_weight_subj, vmax=max_weight_subj)
    edge_colors_subj = {(u, v): cmap(norm_subj(w)) for (u, v), w in zip(edges_subj, weights_subj)}
else:
    edge_widths_subj = {}
    edge_colors_subj = {}

# Node colors - all same for subjects
node_colors_subj = {node: '#7eb3d4' for node in G_subjects.nodes()}

# Use spring layout for better visualization of sparse network
plot_instance_subj = Graph(
    G_subjects,
    node_layout='spring',
    node_layout_kwargs=dict(k=3, iterations=100),
    node_color=node_colors_subj,
    node_size=12,
    node_edge_width=0.5,
    node_labels=True,
    node_label_fontdict=dict(size=11, fontweight='bold'),
    edge_width=edge_widths_subj if edge_widths_subj else 2,
    edge_color=edge_colors_subj if edge_colors_subj else 'gray',
    edge_alpha=0.6,
    edge_labels=False,  # Remove edge labels to reduce clutter
    ax=ax,
    seed=42
)

ax.set_title('Subject Network - Most Similar Pairs\n(Between-subject comparisons, showing top 25% most similar, averaged across all condition-obstacle pairs)',
             fontsize=14, fontweight='bold', pad=20)

# Add a text box with legend
textstr = f'Nodes: {G_subjects.number_of_nodes()} subjects\n'
textstr += f'Edges: {G_subjects.number_of_edges()} connections\n'
textstr += f'Distance range: {min_weight_subj:.0f} - {max_weight_subj:.0f}\n\n'
textstr += 'Edge thickness: Thicker = more similar\n'
textstr += 'Edge color: Green = similar, Red = dissimilar'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'subject_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'subject_network.png'}")
plt.close()


print("\n" + "="*80)
print("NETWORK ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nNetwork Statistics:")
print(f"  Condition network: {G_conditions.number_of_nodes()} nodes, {G_conditions.number_of_edges()} edges")
print(f"  Subject network: {G_subjects.number_of_nodes()} nodes, {G_subjects.number_of_edges()} edges")
