"""
Create network graphs from DTW results data.
Generates three types of networks:
1. Subject network - between-subject comparisons
2. Condition network - within-subject comparisons
3. Multi-layer network - combined view
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from netgraph import Graph

# Load the data
data_path = "/Users/au662726/github_projects/hopscotch/hopscotch_results/dtw_results_20250822_154121.tsv"
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
output_dir = Path("hopscotch_results/network_graphs")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("CREATING NETWORK GRAPHS")
print("="*80)

# ============================================================================
# 1. SUBJECT NETWORK (Between-subject comparisons)
# ============================================================================
print("\n1. Creating Subject Network...")

# Filter between-subject comparisons
between_df = df[df['dtw_type'] == 'between'].copy()

# Create a graph where nodes are subjects and edges are DTW distances
G_subjects = nx.Graph()

# Add all subjects as nodes
subjects = sorted(df['subject_1'].unique())
G_subjects.add_nodes_from(subjects)

# Add edges with DTW distance as weight (average across all condition comparisons)
edge_data = []
for _, row in between_df.iterrows():
    s1, s2 = row['subject_1'], row['subject_2']
    dist = row['dtw_distance']
    edge_data.append({'s1': s1, 's2': s2, 'distance': dist})

edge_df = pd.DataFrame(edge_data)
# Average DTW distance for each subject pair
edge_summary = edge_df.groupby(['s1', 's2'])['distance'].mean().reset_index()

for _, row in edge_summary.iterrows():
    G_subjects.add_edge(row['s1'], row['s2'], weight=row['distance'])

print(f"   Nodes (subjects): {G_subjects.number_of_nodes()}")
print(f"   Edges (subject pairs): {G_subjects.number_of_edges()}")

# Visualize subject network
fig, ax = plt.subplots(figsize=(12, 12))

# Get edge weights for visualization
edges = G_subjects.edges()
weights = [G_subjects[u][v]['weight'] for u, v in edges]

# Normalize weights for edge thickness (inverse: smaller distance = thicker edge)
max_weight = max(weights)
min_weight = min(weights)
edge_widths = {(u, v): 3 * (1 - (w - min_weight) / (max_weight - min_weight)) + 0.5
               for (u, v), w in zip(edges, weights)}

# Edge colors based on distance - manually apply colormap
cmap = plt.colormaps.get_cmap('RdYlGn_r')
norm = plt.Normalize(vmin=min_weight, vmax=max_weight)
edge_colors = {(u, v): cmap(norm(w)) for (u, v), w in zip(edges, weights)}

# Use edge weights as distances for layout (higher DTW = farther apart)
# Scale edge weights to reasonable range for visualization (0.1 to 0.9)
max_dist = max(weights)
min_dist = min(weights)
edge_layout = {(u, v): 0.1 + 0.8 * (G_subjects[u][v]['weight'] - min_dist) / (max_dist - min_dist)
               for u, v in G_subjects.edges()}

# Create the netgraph visualization
plot_instance = Graph(
    G_subjects,
    node_layout='geometric',  # Layout based on edge lengths
    node_layout_kwargs=dict(edge_length=edge_layout),
    scale=(4, 4),
    node_color='lightblue',
    node_size=3,  # Reduced from 8
    node_edge_width=0.3,
    node_labels=True,
    node_label_fontdict=dict(size=8, fontweight='bold'),
    edge_width=edge_widths,
    edge_color=edge_colors,
    edge_alpha=0.7,
    ax=ax
)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Average DTW Distance', shrink=0.8)

ax.set_title('Subject Network\n(Between-subject comparisons, averaged across all conditions)',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'subject_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'subject_network.png'}")
plt.close()

# ============================================================================
# 2. CONDITION NETWORK (Within-subject comparisons)
# ============================================================================
print("\n2. Creating Condition Network...")

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
    #node_alpha=node_alphas_cond,
    node_size=15,  # Reduced from 15
    node_edge_width=0.5,
    node_shape=node_shapes_cond,
    node_labels=True,
    node_label_fontdict=dict(size=9, fontweight='bold'),
    edge_width=edge_widths_cond,
    # edge_color=edge_colors_cond,
    edge_alpha=0.7,
    edge_labels=edge_labels_cond,
    edge_label_fontdict=dict(size=8),
    ax=ax,
    seed=1
)


# Add colorbar
# sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=min_weight_cond, vmax=max_weight_cond))
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, label='Average DTW Distance', shrink=0.8)

ax.set_title('Condition Network\n(Within-subject comparisons, averaged across all subjects)',
             fontsize=14, fontweight='bold', pad=20)


ax.legend(handles=node_proxy_artists, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'condition_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'condition_network.png'}")
plt.close()
exit()
# ============================================================================
# 2b. PARTICIPANT-LEVEL CONDITION NETWORK (With clustering analysis)
# ============================================================================
print("\n2b. Creating Participant-level Condition Network with Clustering...")

# Create a graph where each node is a participant-condition-obstacle combination
G_participant_cond = nx.Graph()

# Add nodes for each participant-condition-obstacle combination
participant_nodes = []
for subject in subjects:
    for cond in ['h', 'k', 's']:
        for obs in [0, 1]:
            cond_short = {'h': 'F', 'k': 'C', 's': 'N'}[cond]  # Fast, Control, fuN
            obs_short = 'O' if obs == 1 else '_'
            node_id = f"S{subject}_{cond_short}{obs_short}"
            G_participant_cond.add_node(node_id,
                                       subject=subject,
                                       condition=cond,
                                       obstacle=obs)
            participant_nodes.append(node_id)

print(f"   Nodes (participant-condition pairs): {G_participant_cond.number_of_nodes()}")

# Add edges based on within-subject DTW distances
for _, row in within_df.iterrows():
    subj = row['subject_1']
    c1, o1 = row['condition_1'], row['obstacles_1']
    c2, o2 = row['condition_2'], row['obstacles_2']

    cond_short_1 = {'h': 'F', 'k': 'C', 's': 'N'}[c1]
    cond_short_2 = {'h': 'F', 'k': 'C', 's': 'N'}[c2]
    obs_short_1 = 'O' if o1 == 1 else '_'
    obs_short_2 = 'O' if o2 == 1 else '_'

    node1 = f"S{subj}_{cond_short_1}{obs_short_1}"
    node2 = f"S{subj}_{cond_short_2}{obs_short_2}"

    dist = row['dtw_distance']
    G_participant_cond.add_edge(node1, node2, weight=dist)

print(f"   Edges (within-participant comparisons): {G_participant_cond.number_of_edges()}")

# Perform community detection to find clusters
from networkx.algorithms import community
communities = community.greedy_modularity_communities(G_participant_cond, weight='weight')
print(f"   Detected {len(communities)} communities/clusters")

# Create color map for communities
community_colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
node_to_community = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_to_community[node] = i

# Visualize participant-level network
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

# Prepare edge data
edges_part = G_participant_cond.edges()
weights_part = [G_participant_cond[u][v]['weight'] for u, v in edges_part]
max_w = max(weights_part)
min_w = min(weights_part)

# Edge properties for both plots
edge_widths_part = {(u, v): 0.5 for u, v in edges_part}
edge_alphas_part = {(u, v): 0.1 + 0.4 * (1 - (w - min_w) / (max_w - min_w))
                    for (u, v), w in zip(edges_part, weights_part)}
# Scale edge weights to reasonable range for visualization (0.1 to 0.9)
edge_layout_part = {(u, v): 0.1 + 0.8 * (G_participant_cond[u][v]['weight'] - min_w) / (max_w - min_w)
                    for u, v in edges_part}

# Left panel: Colored by community/cluster
node_colors_comm = {node: tuple(community_colors[node_to_community[node]])
                    for node in G_participant_cond.nodes()}

plot_instance1 = Graph(
    G_participant_cond,
    node_layout='geometric',
    node_layout_kwargs=dict(edge_length=edge_layout_part),
    scale=(5, 5),
    node_color=node_colors_comm,
    node_size=1.5,  # Reduced from 3
    node_edge_width=0.2,
    node_labels=True,
    node_label_fontdict=dict(size=3),
    edge_width=edge_widths_part,
    edge_color='gray',
    edge_alpha=edge_alphas_part,
    ax=ax1
)

ax1.set_title(f'Participant-Condition Network (Colored by Cluster)\n{len(communities)} clusters detected via modularity optimization',
              fontsize=12, fontweight='bold')

# Right panel: Colored by condition type (using same layout)
node_colors_cond_part = {}
for node in G_participant_cond.nodes():
    node_data = G_participant_cond.nodes[node]
    if node_data['condition'] == 'h':
        node_colors_cond_part[node] = 'salmon'
    elif node_data['condition'] == 'k':
        node_colors_cond_part[node] = 'lightgreen'
    else:  # 's'
        node_colors_cond_part[node] = 'skyblue'

# Reuse the layout from the first plot
node_positions = plot_instance1.node_positions

plot_instance2 = Graph(
    G_participant_cond,
    node_layout=node_positions,
    node_color=node_colors_cond_part,
    node_size=1.5,  # Reduced from 3
    node_edge_width=0.2,
    node_labels=True,
    node_label_fontdict=dict(size=3),
    edge_width=edge_widths_part,
    edge_color='gray',
    edge_alpha=edge_alphas_part,
    ax=ax2
)

ax2.set_title('Participant-Condition Network (Colored by Condition)\nSame layout as left panel',
              fontsize=12, fontweight='bold')

legend_elements_2 = [
    Patch(facecolor='salmon', label='Fast (h)'),
    Patch(facecolor='lightgreen', label='Control (k)'),
    Patch(facecolor='skyblue', label='Fun (s)')
]
ax2.legend(handles=legend_elements_2, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'participant_condition_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'participant_condition_network.png'}")
plt.close()

# Print cluster composition analysis
print("\n   Cluster Composition Analysis:")
for i, comm in enumerate(communities):
    conditions_count = {'h': 0, 'k': 0, 's': 0}
    obstacles_count = {0: 0, 1: 0}
    subjects_in_cluster = set()

    for node in comm:
        node_data = G_participant_cond.nodes[node]
        conditions_count[node_data['condition']] += 1
        obstacles_count[node_data['obstacle']] += 1
        subjects_in_cluster.add(node_data['subject'])

    print(f"     Cluster {i+1}: {len(comm)} nodes, {len(subjects_in_cluster)} subjects")
    print(f"       Conditions: Fast={conditions_count['h']}, Control={conditions_count['k']}, Fun={conditions_count['s']}")
    print(f"       Obstacles: No={obstacles_count[0]}, Yes={obstacles_count[1]}")

# ============================================================================
# 2c. MULTIPLEX NETWORK (Both within and between comparisons)
# ============================================================================
print("\n2c. Creating Multiplex Network (Within + Between Subject Comparisons)...")

# Create a new graph with the same nodes as participant-condition network
G_multiplex = nx.Graph()
G_multiplex.add_nodes_from(G_participant_cond.nodes(data=True))

# Add within-subject edges (from existing data)
within_edges = []
for u, v, data in G_participant_cond.edges(data=True):
    G_multiplex.add_edge(u, v, weight=data['weight'], edge_type='within')
    within_edges.append((u, v, data['weight']))

print(f"   Added {len(within_edges)} within-subject edges")

# Add between-subject edges (for same condition-obstacle combinations)
between_edges = []
for _, row in between_df.iterrows():
    s1, s2 = row['subject_1'], row['subject_2']
    c1, o1 = row['condition_1'], row['obstacles_1']
    c2, o2 = row['condition_2'], row['obstacles_2']

    # Only add edge if conditions and obstacles match
    if c1 == c2 and o1 == o2:
        cond_short = {'h': 'F', 'k': 'C', 's': 'N'}[c1]
        obs_short = 'O' if o1 == 1 else '_'

        node1 = f"S{s1}_{cond_short}{obs_short}"
        node2 = f"S{s2}_{cond_short}{obs_short}"

        dist = row['dtw_distance']

        # Only add edge if it doesn't already exist (shouldn't happen, but safety check)
        if not G_multiplex.has_edge(node1, node2):
            G_multiplex.add_edge(node1, node2, weight=dist, edge_type='between')
            between_edges.append((node1, node2, dist))

print(f"   Added {len(between_edges)} between-subject edges")
print(f"   Total edges: {G_multiplex.number_of_edges()}")

# Separate edges by type for visualization
within_edge_list = [(u, v) for u, v, d in G_multiplex.edges(data=True) if d['edge_type'] == 'within']
between_edge_list = [(u, v) for u, v, d in G_multiplex.edges(data=True) if d['edge_type'] == 'between']

# Perform community detection on the full multiplex network
communities_multi = community.greedy_modularity_communities(G_multiplex, weight='weight')
print(f"   Detected {len(communities_multi)} communities in multiplex network")

# Create color map for communities
community_colors_multi = plt.cm.tab20(np.linspace(0, 1, len(communities_multi)))
node_to_community_multi = {}
for i, comm in enumerate(communities_multi):
    for node in comm:
        node_to_community_multi[node] = i

# Create the visualization
fig, axes = plt.subplots(2, 2, figsize=(24, 24))

# Prepare edge data
within_weights = [G_multiplex[u][v]['weight'] for u, v in within_edge_list]
between_weights = [G_multiplex[u][v]['weight'] for u, v in between_edge_list]

# Filter between edges to show only strongest 20%
if between_weights:
    threshold_between = np.percentile(between_weights, 20)
    strong_between_edges = [(u, v) for u, v in between_edge_list if G_multiplex[u][v]['weight'] <= threshold_between]
else:
    strong_between_edges = []

# Prepare edge layout based on all edges
# Scale edge weights to reasonable range for visualization (0.1 to 0.9)
all_multi_weights = [G_multiplex[u][v]['weight'] for u, v in G_multiplex.edges()]
max_multi_w = max(all_multi_weights)
min_multi_w = min(all_multi_weights)
edge_layout_multi = {(u, v): 0.1 + 0.8 * (G_multiplex[u][v]['weight'] - min_multi_w) / (max_multi_w - min_multi_w)
                     for u, v in G_multiplex.edges()}

# Prepare node colors for different plots
node_colors_multi = {node: tuple(community_colors_multi[node_to_community_multi[node]])
                     for node in G_multiplex.nodes()}

node_colors_cond_multi = {}
for node in G_multiplex.nodes():
    node_data = G_multiplex.nodes[node]
    if node_data['condition'] == 'h':
        node_colors_cond_multi[node] = 'salmon'
    elif node_data['condition'] == 'k':
        node_colors_cond_multi[node] = 'lightgreen'
    else:  # 's'
        node_colors_cond_multi[node] = 'skyblue'

node_colors_obs = {}
for node in G_multiplex.nodes():
    node_data = G_multiplex.nodes[node]
    if node_data['obstacle'] == 0:
        node_colors_obs[node] = 'lightcoral'
    else:
        node_colors_obs[node] = 'mediumseagreen'

# --- Plot 1: Colored by cluster, showing within-subject edges ---
ax = axes[0, 0]

# Create subgraph with only within edges
G_within = G_multiplex.edge_subgraph(within_edge_list).copy()
edge_widths_within = {}
if within_weights:
    max_w_within = max(within_weights)
    min_w_within = min(within_weights)
    edge_widths_within = {(u, v): 1.5 * (1 - (G_multiplex[u][v]['weight'] - min_w_within) / (max_w_within - min_w_within)) + 0.3
                          for u, v in within_edge_list}

plot_multi_1 = Graph(
    G_within,
    node_layout='geometric',
    node_layout_kwargs=dict(edge_length=edge_layout_multi),
    scale=(5, 5),
    node_color=node_colors_multi,
    node_size=1.2,  # Reduced from 2.5
    node_edge_width=0.15,
    node_labels=True,
    node_label_fontdict=dict(size=2.5),
    edge_width=edge_widths_within,
    edge_color='blue',
    edge_alpha=0.4,
    ax=ax
)

ax.set_title(f'Multiplex Network - Within-Subject Edges Only\n(Colored by cluster, {len(communities_multi)} clusters detected)',
             fontsize=11, fontweight='bold')

# Get the layout for reuse
node_positions_multi = plot_multi_1.node_positions

# --- Plot 2: Colored by cluster, showing between-subject edges ---
ax = axes[0, 1]

if strong_between_edges:
    G_between = G_multiplex.edge_subgraph(strong_between_edges).copy()
    strong_between_weights = [G_multiplex[u][v]['weight'] for u, v in strong_between_edges]
    max_w_between = max(strong_between_weights)
    min_w_between = min(strong_between_weights)
    edge_widths_between = {(u, v): 1.5 * (1 - (G_multiplex[u][v]['weight'] - min_w_between) / (max_w_between - min_w_between)) + 0.3
                           for u, v in strong_between_edges}

    plot_multi_2 = Graph(
        G_between,
        node_layout=node_positions_multi,
        node_color=node_colors_multi,
        node_size=1.2,  # Reduced from 2.5
        node_edge_width=0.15,
        node_labels=True,
        node_label_fontdict=dict(size=2.5),
        edge_width=edge_widths_between,
        edge_color='red',
        edge_alpha=0.4,
        ax=ax
    )

ax.set_title(f'Multiplex Network - Between-Subject Edges Only\n(Showing strongest 20% of {len(between_edge_list)} between-subject connections)',
             fontsize=11, fontweight='bold')

# --- Plot 3: Colored by condition, showing both edge types ---
ax = axes[1, 0]

# Create combined edge lists with colors
combined_edges_3 = within_edge_list + strong_between_edges
edge_colors_3 = {}
for u, v in within_edge_list:
    edge_colors_3[(u, v)] = 'blue'
for u, v in strong_between_edges:
    edge_colors_3[(u, v)] = 'red'

G_combined_3 = G_multiplex.edge_subgraph(combined_edges_3).copy()

plot_multi_3 = Graph(
    G_combined_3,
    node_layout=node_positions_multi,
    node_color=node_colors_cond_multi,
    node_size=1.2,  # Reduced from 2.5
    node_edge_width=0.15,
    node_labels=True,
    node_label_fontdict=dict(size=2.5),
    edge_width=0.5,
    edge_color=edge_colors_3,
    edge_alpha=0.3,
    ax=ax
)

ax.set_title('Multiplex Network - Both Edge Types\n(Blue=within-subject, Red=between-subject)',
             fontsize=11, fontweight='bold')

legend_elements_multi = [
    Patch(facecolor='salmon', label='Fast (h)'),
    Patch(facecolor='lightgreen', label='Control (k)'),
    Patch(facecolor='skyblue', label='Fun (s)'),
    Patch(facecolor='white', edgecolor='blue', label='Within-subject edge'),
    Patch(facecolor='white', edgecolor='red', label='Between-subject edge')
]
ax.legend(handles=legend_elements_multi, loc='upper right', fontsize=9)

# --- Plot 4: Colored by obstacle, showing both edge types ---
ax = axes[1, 1]

plot_multi_4 = Graph(
    G_combined_3,
    node_layout=node_positions_multi,
    node_color=node_colors_obs,
    node_size=1.2,  # Reduced from 2.5
    node_edge_width=0.15,
    node_labels=True,
    node_label_fontdict=dict(size=2.5),
    edge_width=0.5,
    edge_color=edge_colors_3,
    edge_alpha=0.3,
    ax=ax
)

ax.set_title('Multiplex Network - Colored by Obstacle Condition\n(Blue=within-subject, Red=between-subject)',
             fontsize=11, fontweight='bold')

legend_elements_obs = [
    Patch(facecolor='lightcoral', label='No obstacles'),
    Patch(facecolor='mediumseagreen', label='With obstacles'),
    Patch(facecolor='white', edgecolor='blue', label='Within-subject edge'),
    Patch(facecolor='white', edgecolor='red', label='Between-subject edge')
]
ax.legend(handles=legend_elements_obs, loc='upper right', fontsize=9)

plt.suptitle('Multiplex Network: Combined Within & Between Subject Comparisons',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'multiplex_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'multiplex_network.png'}")
plt.close()

# Print multiplex cluster analysis
print("\n   Multiplex Network Cluster Composition:")
for i, comm in enumerate(communities_multi):
    conditions_count = {'h': 0, 'k': 0, 's': 0}
    obstacles_count = {0: 0, 1: 0}
    subjects_in_cluster = set()

    for node in comm:
        node_data = G_multiplex.nodes[node]
        conditions_count[node_data['condition']] += 1
        obstacles_count[node_data['obstacle']] += 1
        subjects_in_cluster.add(node_data['subject'])

    print(f"     Cluster {i+1}: {len(comm)} nodes, {len(subjects_in_cluster)} subjects")
    print(f"       Conditions: Fast={conditions_count['h']}, Control={conditions_count['k']}, Fun={conditions_count['s']}")
    print(f"       Obstacles: No={obstacles_count[0]}, Yes={obstacles_count[1]}")

# ============================================================================
# 3. MULTI-LAYER NETWORK (Combined view)
# ============================================================================
print("\n3. Creating Multi-layer Network...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Left panel: Subject network (simplified) - only show strongest connections
threshold = np.percentile(weights, 30)
strong_edges_subj = [(u, v) for u, v, d in G_subjects.edges(data=True) if d['weight'] <= threshold]
G_subjects_strong = G_subjects.edge_subgraph(strong_edges_subj).copy()

# Edge layout for subjects - use the already scaled edge_layout from earlier
# (edge_layout is already scaled in the subject network section above)

plot_subjects = Graph(
    G_subjects_strong,
    node_layout='geometric',
    node_layout_kwargs=dict(edge_length=edge_layout),
    scale=(3, 3),
    node_color='lightblue',
    node_size=2.5,  # Reduced from 6
    node_edge_width=0.3,
    node_labels=True,
    node_label_fontdict=dict(size=8, fontweight='bold'),
    edge_width=2,
    edge_color='blue',
    edge_alpha=0.5,
    ax=ax1
)

ax1.set_title('Between-Subject Network\n(Strongest 30% connections shown)',
              fontsize=12, fontweight='bold')

# Right panel: Condition network - only show strongest connections
threshold_cond_multi = np.percentile(weights_cond, 30)
strong_edges_cond_multi = [(u, v) for u, v, d in G_conditions.edges(data=True) if d['weight'] <= threshold_cond_multi]
G_conditions_strong = G_conditions.edge_subgraph(strong_edges_cond_multi).copy()

plot_conditions = Graph(
    G_conditions_strong,
    node_layout='geometric',
    node_layout_kwargs=dict(edge_length=edge_layout_cond),
    scale=(3, 3),
    node_color=node_colors_cond,
    node_size=4,  # Reduced from 12
    node_edge_width=0.5,
    node_labels=True,
    node_label_fontdict=dict(size=9, fontweight='bold'),
    edge_width=3,
    edge_color='red',
    edge_alpha=0.5,
    ax=ax2
)

ax2.set_title('Within-Subject Network\n(Strongest 30% connections shown)',
              fontsize=12, fontweight='bold')
ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.suptitle('Multi-layer Network: Between vs Within Comparisons',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_dir / 'multilayer_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'multilayer_network.png'}")
plt.close()

print("\n" + "="*80)
print("NETWORK ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nNetwork Statistics:")
print(f"  Subject network: {G_subjects.number_of_nodes()} nodes, {G_subjects.number_of_edges()} edges")
print(f"  Condition network: {G_conditions.number_of_nodes()} nodes, {G_conditions.number_of_edges()} edges")
print(f"  Participant-condition network: {G_participant_cond.number_of_nodes()} nodes, {G_participant_cond.number_of_edges()} edges, {len(communities)} clusters")
print(f"  Multiplex network: {G_multiplex.number_of_nodes()} nodes, {G_multiplex.number_of_edges()} edges ({len(within_edge_list)} within + {len(between_edge_list)} between), {len(communities_multi)} clusters")
