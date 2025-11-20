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
fig, ax = plt.subplots(figsize=(14, 14))
pos = nx.spring_layout(G_subjects, k=2, iterations=50, seed=42)

# Get edge weights for visualization
edges = G_subjects.edges()
weights = [G_subjects[u][v]['weight'] for u, v in edges]

# Normalize weights for edge thickness (inverse: smaller distance = thicker edge)
max_weight = max(weights)
min_weight = min(weights)
edge_widths = [5 * (1 - (w - min_weight) / (max_weight - min_weight)) + 0.5 for w in weights]

# Edge colors based on distance (similar subjects = darker/warmer)
edge_colors = weights

nx.draw_networkx_nodes(G_subjects, pos, node_color='lightblue',
                       node_size=800, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_subjects, pos, font_size=10, font_weight='bold', ax=ax)
edges_drawn = nx.draw_networkx_edges(G_subjects, pos, width=edge_widths,
                                     edge_color=edge_colors, edge_cmap=plt.cm.RdYlGn_r,
                                     alpha=0.6, ax=ax)

plt.colorbar(edges_drawn, ax=ax, label='Average DTW Distance')
plt.title('Subject Network\n(Between-subject comparisons, averaged across all conditions)',
          fontsize=14, fontweight='bold', pad=20)
plt.axis('off')
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
    cond_map = {'h': 'Fast', 'k': 'Control', 's': 'Fun'}
    obs_map = {0: 'No obstacles', 1: 'With obstacles'}
    return f"{cond_map.get(cond, cond)}\n({obs_map.get(obs, obs)})"

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
fig, ax = plt.subplots(figsize=(12, 10))

# Position nodes in a circle
pos_cond = nx.circular_layout(G_conditions)

# Get edge weights for visualization
edges_cond = G_conditions.edges()
weights_cond = [G_conditions[u][v]['weight'] for u, v in edges_cond]

# Normalize weights for edge thickness
max_weight_cond = max(weights_cond)
min_weight_cond = min(weights_cond)
edge_widths_cond = [8 * (1 - (w - min_weight_cond) / (max_weight_cond - min_weight_cond)) + 1 for w in weights_cond]

# Node colors based on condition type
node_colors = []
for node in G_conditions.nodes():
    node_data = G_conditions.nodes[node]
    if node_data['condition'] == 'h':
        node_colors.append('salmon')
    elif node_data['condition'] == 'k':
        node_colors.append('lightgreen')
    else:  # 's'
        node_colors.append('skyblue')

nx.draw_networkx_nodes(G_conditions, pos_cond, node_color=node_colors,
                       node_size=2500, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_conditions, pos_cond, font_size=9, font_weight='bold', ax=ax)
edges_drawn_cond = nx.draw_networkx_edges(G_conditions, pos_cond, width=edge_widths_cond,
                                          edge_color=weights_cond, edge_cmap=plt.cm.RdYlGn_r,
                                          alpha=0.7, ax=ax)

# Add edge labels with distances
edge_labels = {(u, v): f"{G_conditions[u][v]['weight']:.0f}" for u, v in G_conditions.edges()}
nx.draw_networkx_edge_labels(G_conditions, pos_cond, edge_labels, font_size=8, ax=ax)

plt.colorbar(edges_drawn_cond, ax=ax, label='Average DTW Distance')
plt.title('Condition Network\n(Within-subject comparisons, averaged across all subjects)',
          fontsize=14, fontweight='bold', pad=20)

# Add legend for conditions
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='salmon', label='Fast'),
    Patch(facecolor='lightgreen', label='Control'),
    Patch(facecolor='skyblue', label='Fun')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'condition_network.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / 'condition_network.png'}")
plt.close()

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

# Left panel: Colored by community/cluster
pos_part = nx.spring_layout(G_participant_cond, k=1, iterations=50, seed=42)

node_colors_comm = [community_colors[node_to_community[node]] for node in G_participant_cond.nodes()]

nx.draw_networkx_nodes(G_participant_cond, pos_part, node_color=node_colors_comm,
                       node_size=200, alpha=0.8, ax=ax1)
nx.draw_networkx_labels(G_participant_cond, pos_part, font_size=4, ax=ax1)

# Draw edges with transparency based on weight
edges_part = G_participant_cond.edges()
weights_part = [G_participant_cond[u][v]['weight'] for u, v in edges_part]
max_w = max(weights_part)
min_w = min(weights_part)
edge_alphas = [0.1 + 0.4 * (1 - (w - min_w) / (max_w - min_w)) for w in weights_part]

for (u, v), alpha in zip(edges_part, edge_alphas):
    nx.draw_networkx_edges(G_participant_cond, pos_part, [(u, v)],
                          width=0.5, alpha=alpha, edge_color='gray', ax=ax1)

ax1.set_title(f'Participant-Condition Network (Colored by Cluster)\n{len(communities)} clusters detected via modularity optimization',
              fontsize=12, fontweight='bold')
ax1.axis('off')

# Right panel: Colored by condition type
node_colors_cond_part = []
for node in G_participant_cond.nodes():
    node_data = G_participant_cond.nodes[node]
    if node_data['condition'] == 'h':
        node_colors_cond_part.append('salmon')
    elif node_data['condition'] == 'k':
        node_colors_cond_part.append('lightgreen')
    else:  # 's'
        node_colors_cond_part.append('skyblue')

nx.draw_networkx_nodes(G_participant_cond, pos_part, node_color=node_colors_cond_part,
                       node_size=200, alpha=0.8, ax=ax2)
nx.draw_networkx_labels(G_participant_cond, pos_part, font_size=4, ax=ax2)

for (u, v), alpha in zip(edges_part, edge_alphas):
    nx.draw_networkx_edges(G_participant_cond, pos_part, [(u, v)],
                          width=0.5, alpha=alpha, edge_color='gray', ax=ax2)

ax2.set_title('Participant-Condition Network (Colored by Condition)\nSame layout as left panel',
              fontsize=12, fontweight='bold')

legend_elements_2 = [
    Patch(facecolor='salmon', label='Fast (h)'),
    Patch(facecolor='lightgreen', label='Control (k)'),
    Patch(facecolor='skyblue', label='Fun (s)')
]
ax2.legend(handles=legend_elements_2, loc='upper right', fontsize=10)
ax2.axis('off')

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

# Use same layout for all plots for comparison
pos_multi = nx.spring_layout(G_multiplex, k=0.3, iterations=100, seed=42, weight='weight')

# --- Plot 1: Colored by cluster, showing within-subject edges ---
ax = axes[0, 0]
node_colors_multi = [community_colors_multi[node_to_community_multi[node]] for node in G_multiplex.nodes()]

nx.draw_networkx_nodes(G_multiplex, pos_multi, node_color=node_colors_multi,
                       node_size=150, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_multiplex, pos_multi, font_size=3, ax=ax)

# Draw only within-subject edges
within_weights = [G_multiplex[u][v]['weight'] for u, v in within_edge_list]
if within_weights:
    max_w_within = max(within_weights)
    min_w_within = min(within_weights)
    within_widths = [1.5 * (1 - (w - min_w_within) / (max_w_within - min_w_within)) + 0.3 for w in within_weights]

    nx.draw_networkx_edges(G_multiplex, pos_multi, edgelist=within_edge_list,
                          width=within_widths, alpha=0.4, edge_color='blue', ax=ax)

ax.set_title(f'Multiplex Network - Within-Subject Edges Only\n(Colored by cluster, {len(communities_multi)} clusters detected)',
             fontsize=11, fontweight='bold')
ax.axis('off')

# --- Plot 2: Colored by cluster, showing between-subject edges ---
ax = axes[0, 1]
nx.draw_networkx_nodes(G_multiplex, pos_multi, node_color=node_colors_multi,
                       node_size=150, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_multiplex, pos_multi, font_size=3, ax=ax)

# Draw only between-subject edges (filtered to show strongest connections)
between_weights = [G_multiplex[u][v]['weight'] for u, v in between_edge_list]
if between_weights:
    # Only show strongest 20% of between-subject connections to reduce clutter
    threshold_between = np.percentile(between_weights, 20)
    strong_between_edges = [(u, v) for u, v in between_edge_list if G_multiplex[u][v]['weight'] <= threshold_between]
    strong_between_weights = [G_multiplex[u][v]['weight'] for u, v in strong_between_edges]

    if strong_between_weights:
        max_w_between = max(strong_between_weights)
        min_w_between = min(strong_between_weights)
        between_widths = [1.5 * (1 - (w - min_w_between) / (max_w_between - min_w_between)) + 0.3 for w in strong_between_weights]

        nx.draw_networkx_edges(G_multiplex, pos_multi, edgelist=strong_between_edges,
                              width=between_widths, alpha=0.4, edge_color='red', ax=ax)

ax.set_title(f'Multiplex Network - Between-Subject Edges Only\n(Showing strongest 20% of {len(between_edge_list)} between-subject connections)',
             fontsize=11, fontweight='bold')
ax.axis('off')

# --- Plot 3: Colored by condition, showing both edge types ---
ax = axes[1, 0]
node_colors_cond_multi = []
for node in G_multiplex.nodes():
    node_data = G_multiplex.nodes[node]
    if node_data['condition'] == 'h':
        node_colors_cond_multi.append('salmon')
    elif node_data['condition'] == 'k':
        node_colors_cond_multi.append('lightgreen')
    else:  # 's'
        node_colors_cond_multi.append('skyblue')

nx.draw_networkx_nodes(G_multiplex, pos_multi, node_color=node_colors_cond_multi,
                       node_size=150, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_multiplex, pos_multi, font_size=3, ax=ax)

# Draw both edge types with different colors
if within_weights:
    nx.draw_networkx_edges(G_multiplex, pos_multi, edgelist=within_edge_list,
                          width=0.5, alpha=0.3, edge_color='blue', ax=ax)

if between_weights and strong_between_edges:
    nx.draw_networkx_edges(G_multiplex, pos_multi, edgelist=strong_between_edges,
                          width=0.5, alpha=0.3, edge_color='red', ax=ax)

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
ax.axis('off')

# --- Plot 4: Colored by obstacle, showing both edge types ---
ax = axes[1, 1]
node_colors_obs = []
for node in G_multiplex.nodes():
    node_data = G_multiplex.nodes[node]
    if node_data['obstacle'] == 0:
        node_colors_obs.append('lightcoral')
    else:
        node_colors_obs.append('mediumseagreen')

nx.draw_networkx_nodes(G_multiplex, pos_multi, node_color=node_colors_obs,
                       node_size=150, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G_multiplex, pos_multi, font_size=3, ax=ax)

# Draw both edge types
if within_weights:
    nx.draw_networkx_edges(G_multiplex, pos_multi, edgelist=within_edge_list,
                          width=0.5, alpha=0.3, edge_color='blue', ax=ax)

if between_weights and strong_between_edges:
    nx.draw_networkx_edges(G_multiplex, pos_multi, edgelist=strong_between_edges,
                          width=0.5, alpha=0.3, edge_color='red', ax=ax)

ax.set_title('Multiplex Network - Colored by Obstacle Condition\n(Blue=within-subject, Red=between-subject)',
             fontsize=11, fontweight='bold')

legend_elements_obs = [
    Patch(facecolor='lightcoral', label='No obstacles'),
    Patch(facecolor='mediumseagreen', label='With obstacles'),
    Patch(facecolor='white', edgecolor='blue', label='Within-subject edge'),
    Patch(facecolor='white', edgecolor='red', label='Between-subject edge')
]
ax.legend(handles=legend_elements_obs, loc='upper right', fontsize=9)
ax.axis('off')

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

# Left panel: Subject network (simplified)
pos_subjects = nx.spring_layout(G_subjects, k=2, iterations=50, seed=42)
nx.draw_networkx_nodes(G_subjects, pos_subjects, node_color='lightblue',
                       node_size=600, alpha=0.9, ax=ax1)
nx.draw_networkx_labels(G_subjects, pos_subjects, font_size=8, ax=ax1)

# Only show strongest connections (bottom 30% of distances)
threshold = np.percentile(weights, 30)
strong_edges = [(u, v) for u, v, d in G_subjects.edges(data=True) if d['weight'] <= threshold]
nx.draw_networkx_edges(G_subjects, pos_subjects, edgelist=strong_edges,
                      width=2, alpha=0.5, edge_color='blue', ax=ax1)

ax1.set_title('Between-Subject Network\n(Strongest 30% connections shown)',
              fontsize=12, fontweight='bold')
ax1.axis('off')

# Right panel: Condition network
nx.draw_networkx_nodes(G_conditions, pos_cond, node_color=node_colors,
                       node_size=2000, alpha=0.9, ax=ax2)
nx.draw_networkx_labels(G_conditions, pos_cond, font_size=8, ax=ax2)

# Only show strongest connections
threshold_cond = np.percentile(weights_cond, 30)
strong_edges_cond = [(u, v) for u, v, d in G_conditions.edges(data=True) if d['weight'] <= threshold_cond]
nx.draw_networkx_edges(G_conditions, pos_cond, edgelist=strong_edges_cond,
                      width=3, alpha=0.5, edge_color='red', ax=ax2)

ax2.set_title('Within-Subject Network\n(Strongest 30% connections shown)',
              fontsize=12, fontweight='bold')
ax2.legend(handles=legend_elements, loc='upper left', fontsize=9)
ax2.axis('off')

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
