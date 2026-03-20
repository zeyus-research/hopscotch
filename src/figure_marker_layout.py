"""
Publication-ready reference figure: marker placement and joint angle calculations.

Produces a static schematic — no data file required.
Coordinate convention matches the Qualisys export used in this project:
  Z = up, X = lateral (frontal view), Y = depth (sagittal view).

Run:
    python src/figure_marker_layout.py
Output:
    analysis/figure_marker_layout.pdf
    analysis/figure_marker_layout.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── style ─────────────────────────────────────────────────────────────────────
try:
    plt.rcParams['font.family'] = 'Arial'
except Exception:
    pass
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})

BONE    = '#2c3e50'   # dark navy  – skeleton segments
FRONT   = '#2980b9'   # blue       – anterior/lateral markers
BACK    = '#85c1e9'   # light blue – posterior hip markers
DERIVED = '#aaaaaa'   # grey       – computed midpoints (not physical markers)
ANGLE   = '#c0392b'   # red        – angle arcs & labels
FRAME   = '#27ae60'   # green      – torso coordinate-frame axes


# ── helpers ───────────────────────────────────────────────────────────────────
def seg(ax, p1, p2, color=BONE, lw=2.0, **kw):
    kw.setdefault('solid_capstyle', 'round')
    kw.setdefault('zorder', 2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, **kw)


def dot(ax, p, color=FRONT, size=55, zorder=6, **kw):
    ax.scatter(p[0], p[1], facecolors=color, edgecolors='white',
               s=size, linewidths=0.8, zorder=zorder, **kw)


def angle_arc(ax, apex, p1, p2, r=8, color=ANGLE, label='', label_r=None):
    """Draw the interior angle arc at *apex* between directions apex→p1 and apex→p2."""
    v1 = p1 - apex
    v2 = p2 - apex
    a1 = np.degrees(np.arctan2(v1[1], v1[0]))
    a2 = np.degrees(np.arctan2(v2[1], v2[0]))
    # always sweep the short way
    diff = (a2 - a1 + 360) % 360
    if diff > 180:
        a1, a2 = a2, a1
    theta1, theta2 = min(a1, a2), max(a1, a2)
    ax.add_patch(Arc(apex, 2 * r, 2 * r, angle=0,
                     theta1=theta1, theta2=theta2,
                     color=color, lw=1.8, zorder=7))
    if label:
        mid = np.radians((theta1 + theta2) / 2)
        lr = label_r or (r + 7)
        ax.text(apex[0] + lr * np.cos(mid), apex[1] + lr * np.sin(mid),
                label, color=color, fontsize=9, ha='center', va='center',
                fontstyle='italic')


def panel_label(ax, text):
    ax.set_title(text, loc='left', fontsize=10, fontweight='bold', pad=5)


# ── schematic marker positions (frontal view, X = lateral, Z = up, units ≈ mm)
# Arms held slightly away from body so segments are clearly visible.
# ─────────────────────────────────────────────────────────────────────────────
M = {}
M['head']          = np.r_[   0., 1650.]
M['shoulder_r']    = np.r_[-200., 1480.]
M['shoulder_l']    = np.r_[ 200., 1480.]
M['elbow_r']       = np.r_[-280., 1220.]
M['elbow_l']       = np.r_[ 280., 1220.]
M['wrist_r']       = np.r_[-330., 1000.]
M['wrist_l']       = np.r_[ 330., 1000.]
M['hip_front_r']   = np.r_[-140.,  930.]
M['hip_front_l']   = np.r_[ 140.,  930.]
M['hip_back_r']    = np.r_[ -90.,  930.]   # more medial in frontal view
M['hip_back_l']    = np.r_[  90.,  930.]
M['knee_over_r']   = np.r_[-140.,  700.]   # distal femur side
M['knee_under_r']  = np.r_[-140.,  640.]   # proximal tibia side
M['knee_over_l']   = np.r_[ 140.,  700.]
M['knee_under_l']  = np.r_[ 140.,  640.]
M['foot_front_r']  = np.r_[-260.,   80.]
M['foot_back_r']   = np.r_[-100.,   80.]
M['foot_front_l']  = np.r_[ 260.,   80.]
M['foot_back_l']   = np.r_[ 100.,   80.]

# computed midpoints (not physical markers)
M['hip_r']        = (M['hip_front_r'] + M['hip_back_r']) / 2
M['hip_l']        = (M['hip_front_l'] + M['hip_back_l']) / 2
M['knee_r']       = (M['knee_over_r'] + M['knee_under_r']) / 2
M['knee_l']       = (M['knee_over_l'] + M['knee_under_l']) / 2
M['ankle_r']      = (M['foot_front_r'] + M['foot_back_r']) / 2
M['ankle_l']      = (M['foot_front_l'] + M['foot_back_l']) / 2
M['shoulder_mid'] = (M['shoulder_r'] + M['shoulder_l']) / 2
M['hip_mid']      = (M['hip_front_r'] + M['hip_front_l']) / 2


# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 13))
gs = GridSpec(3, 2, figure=fig,
              width_ratios=[1.8, 1.0],
              hspace=0.40, wspace=0.10,
              left=0.03, right=0.97, top=0.96, bottom=0.03)

ax_body  = fig.add_subplot(gs[:, 0])
ax_elbow = fig.add_subplot(gs[0, 1])
ax_knee  = fig.add_subplot(gs[1, 1])
ax_hip   = fig.add_subplot(gs[2, 1])

for ax in (ax_body, ax_elbow, ax_knee, ax_hip):
    ax.set_aspect('equal')
    ax.axis('off')


# ═══════════════════════════════════════════════════════════════════════════════
# A – Full-body frontal view
# ═══════════════════════════════════════════════════════════════════════════════

# skeleton segments
BONES = [
    ('shoulder_r', 'elbow_r'), ('elbow_r', 'wrist_r'),
    ('shoulder_l', 'elbow_l'), ('elbow_l', 'wrist_l'),
    ('shoulder_r', 'shoulder_l'),
    ('shoulder_r', 'hip_front_r'), ('shoulder_l', 'hip_front_l'),
    ('hip_front_r', 'hip_front_l'),
    ('hip_front_r', 'knee_over_r'), ('knee_over_r', 'knee_under_r'),
    ('knee_under_r', 'ankle_r'),
    ('ankle_r', 'foot_back_r'), ('foot_front_r', 'foot_back_r'),
    ('hip_front_l', 'knee_over_l'), ('knee_over_l', 'knee_under_l'),
    ('knee_under_l', 'ankle_l'),
    ('ankle_l', 'foot_back_l'), ('foot_front_l', 'foot_back_l'),
    ('head', 'shoulder_mid'),
]
for m1, m2 in BONES:
    seg(ax_body, M[m1], M[m2])

# head circle
ax_body.add_patch(Circle(M['head'], 90, fill=False, color=BONE, lw=2.0, zorder=2))

# physical markers (anterior/lateral = solid blue, posterior hip = lighter)
for k in ['shoulder_r', 'shoulder_l', 'elbow_r', 'elbow_l', 'wrist_r', 'wrist_l',
          'hip_front_r', 'hip_front_l',
          'knee_over_r', 'knee_under_r', 'knee_over_l', 'knee_under_l',
          'foot_front_r', 'foot_back_r', 'foot_front_l', 'foot_back_l', 'head']:
    dot(ax_body, M[k], FRONT, size=70)
for k in ('hip_back_r', 'hip_back_l'):
    dot(ax_body, M[k], BACK, size=70)

# computed midpoints shown as open circles
for k in ('hip_r', 'hip_l', 'ankle_r', 'ankle_l'):
    ax_body.scatter(M[k][0], M[k][1], facecolors='none', edgecolors=DERIVED,
                    s=55, linewidths=1.2, zorder=5)

# ── marker labels ──
def lbl(ax, pos, text, dx=0, dy=0, ha='left', va='center', fs=7.5):
    ax.text(pos[0] + dx, pos[1] + dy, text, fontsize=fs,
            ha=ha, va=va, color=BONE)

# participant's right side (figure's left) – right-aligned
lbl(ax_body, M['shoulder_r'],   'Shoulder',              dx=-15, ha='right')
lbl(ax_body, M['elbow_r'],      'Elbow',                 dx=-15, ha='right')
lbl(ax_body, M['wrist_r'],      'Wrist',                 dx=-15, ha='right')
lbl(ax_body, M['hip_front_r'],  'Hip (ant.)',             dx=-15, ha='right', dy=15)
lbl(ax_body, M['hip_back_r'],   'Hip (post.)',            dx=-15, ha='right', dy=-20)
lbl(ax_body, M['knee_over_r'],  'Knee (distal femur)',    dx=-15, ha='right')
lbl(ax_body, M['knee_under_r'], 'Knee (proximal tibia)', dx=-15, ha='right')
lbl(ax_body, M['foot_front_r'], 'Foot (ant.)',            dx=-15, ha='right', dy=-18)
lbl(ax_body, M['foot_back_r'],  'Foot (post.)',           dx=15,  ha='left',  dy=-18)

# participant's left side (figure's right) – left-aligned
lbl(ax_body, M['shoulder_l'],   'Shoulder',              dx=15, ha='left')
lbl(ax_body, M['elbow_l'],      'Elbow',                 dx=15, ha='left')
lbl(ax_body, M['wrist_l'],      'Wrist',                 dx=15, ha='left')
lbl(ax_body, M['hip_front_l'],  'Hip (ant.)',             dx=15, ha='left',  dy=15)
lbl(ax_body, M['hip_back_l'],   'Hip (post.)',            dx=15, ha='left',  dy=-20)
lbl(ax_body, M['knee_over_l'],  'Knee (distal femur)',    dx=15, ha='left')
lbl(ax_body, M['knee_under_l'], 'Knee (proximal tibia)', dx=15, ha='left')
lbl(ax_body, M['foot_front_l'], 'Foot (ant.)',            dx=15, ha='left',  dy=-18)
lbl(ax_body, M['foot_back_l'],  'Foot (post.)',           dx=-15, ha='right', dy=-18)

lbl(ax_body, M['head'], 'Head', dx=110, ha='left')

# R / L indicators
ax_body.text(-480, 100, 'R', fontsize=12, fontweight='bold',
             color='#999999', ha='center')
ax_body.text( 480, 100, 'L', fontsize=12, fontweight='bold',
             color='#999999', ha='center')

# legend
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=FRONT,
           markersize=9, label='Anterior / lateral marker'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BACK,
           markersize=9, label='Posterior hip marker'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=DERIVED, markersize=9,
           label='Computed midpoint (not a physical marker)'),
]
ax_body.legend(handles=legend_handles, loc='lower center',
               framealpha=0.9, fontsize=8, ncol=1,
               bbox_to_anchor=(0.5, -0.01))

ax_body.set_xlim(-580, 580)
ax_body.set_ylim(-80, 1800)
panel_label(ax_body, 'A    Marker placement (frontal view)')


# ═══════════════════════════════════════════════════════════════════════════════
# B – Elbow flexion
# Angle defined by shoulder → elbow ← wrist (0° = fully extended)
# ═══════════════════════════════════════════════════════════════════════════════
E = {
    'shoulder': np.r_[  0., 48.],
    'elbow':    np.r_[  0.,  0.],
    'wrist':    np.r_[ 22., -24.],
}
seg(ax_elbow, E['shoulder'], E['elbow'])
seg(ax_elbow, E['elbow'],    E['wrist'])
for p in E.values():
    dot(ax_elbow, p)

angle_arc(ax_elbow, E['elbow'], E['shoulder'], E['wrist'],
          r=14, label='θ', label_r=22)

ax_elbow.text(*E['shoulder'] + [3,  2],  'Shoulder marker', fontsize=8, ha='left')
ax_elbow.text(*E['elbow']    + [-3, -7], 'Elbow marker',    fontsize=8, ha='right', va='top')
ax_elbow.text(*E['wrist']    + [ 3,  0], 'Wrist marker',    fontsize=8, ha='left')

ax_elbow.set_xlim(-24, 52)
ax_elbow.set_ylim(-40, 58)
panel_label(ax_elbow, 'B    Elbow flexion')


# ═══════════════════════════════════════════════════════════════════════════════
# C – Knee flexion
# Two separate markers straddle the joint: knee_over (distal femur) and
# knee_under (proximal tibia).  The angle is between the femur direction
# (knee_over → hip centre) and the tibia direction (knee_under → ankle centre).
# ═══════════════════════════════════════════════════════════════════════════════
K = {
    'hip':         np.r_[ 0., 58.],
    'knee_over':   np.r_[ 0., 20.],
    'knee_under':  np.r_[ 0.,  8.],
    'ankle':       np.r_[18., -32.],
}
seg(ax_knee, K['hip'],        K['knee_over'])
seg(ax_knee, K['knee_under'], K['ankle'])
for k, p in K.items():
    dot(ax_knee, p)

# Arc at knee_over: between femur direction (→ hip) and tibia direction extended
shank_dir = K['ankle'] - K['knee_under']
shank_ref = K['knee_over'] + shank_dir * 0.7   # extend tibia dir from knee_over
angle_arc(ax_knee, K['knee_over'], K['hip'], shank_ref,
          r=14, label='θ', label_r=22)

# annotation showing the two markers are distinct
ax_knee.annotate('', xy=K['knee_under'] + [6, 0], xytext=K['knee_over'] + [6, 0],
                 arrowprops=dict(arrowstyle='<->', color='#888888', lw=1.0))
ax_knee.text(K['knee_over'][0] + 8,
             (K['knee_over'][1] + K['knee_under'][1]) / 2,
             'separate\nmarkers', fontsize=7, color='#666666',
             ha='left', va='center')

ax_knee.text(*K['hip']        + [3,  2], 'Hip centre\n(midpoint)',         fontsize=8, ha='left')
ax_knee.text(*K['knee_over']  + [-3, 3], 'Knee\n(distal femur)',           fontsize=8, ha='right')
ax_knee.text(*K['knee_under'] + [-3,-2], 'Knee\n(proximal tibia)',         fontsize=8, ha='right', va='top')
ax_knee.text(*K['ankle']      + [3,  0], 'Ankle centre\n(midpoint)',       fontsize=8, ha='left')

ax_knee.set_xlim(-28, 48)
ax_knee.set_ylim(-44, 70)
panel_label(ax_knee, 'C    Knee flexion')


# ═══════════════════════════════════════════════════════════════════════════════
# D – Hip flexion (relative to pelvis coordinate frame)
# The pelvis frame is built from the four hip markers; hip flexion is the angle
# of the thigh from vertical (neutral = leg straight down, positive = forward).
# ═══════════════════════════════════════════════════════════════════════════════
H = {
    'hip':  np.r_[ 0.,  0.],
    'knee': np.r_[20., -40.],
}
seg(ax_hip, H['hip'], H['knee'])
for p in H.values():
    dot(ax_hip, p)

# Pelvis frame axes
ax_hip.annotate('', xy=H['hip'] + [0, 42], xytext=H['hip'],
                arrowprops=dict(arrowstyle='->', color=FRAME, lw=2.0,
                                mutation_scale=12))
ax_hip.annotate('', xy=H['hip'] + [30, 0], xytext=H['hip'],
                arrowprops=dict(arrowstyle='->', color=FRAME, lw=2.0,
                                mutation_scale=12))
ax_hip.text(2,  45, 'Vertical',  fontsize=8, color=FRAME, ha='left')
ax_hip.text(32,  3, 'Forward',   fontsize=8, color=FRAME, ha='left')

# Neutral reference (straight down)
seg(ax_hip, H['hip'], H['hip'] + [0, -44],
    color='#cccccc', lw=1.3, linestyle='--', zorder=1)
ax_hip.text(2, -46, 'Neutral\n(extended)', fontsize=7.5,
            color='#aaaaaa', ha='left')

angle_arc(ax_hip, H['hip'], H['hip'] + [0, -44], H['knee'],
          r=16, label='θ_flex', label_r=27)

ax_hip.text(*H['hip']  + [ 3,  2], 'Hip centre',  fontsize=8, ha='left')
ax_hip.text(*H['knee'] + [ 3,  0], 'Knee centre', fontsize=8, ha='left')

# small box labelling the coordinate frame
rect = plt.Rectangle((-34, -14), 30, 13, fill=True,
                      facecolor='#f0f9f4', edgecolor=FRAME, lw=0.8, zorder=1)
ax_hip.add_patch(rect)
ax_hip.text(-19, -7.5, 'Pelvis coordinate\nframe',
            fontsize=7, color=FRAME, ha='center', va='center')

ax_hip.set_xlim(-40, 68)
ax_hip.set_ylim(-55, 57)
panel_label(ax_hip, 'D    Hip flexion (sagittal plane)')


# ── save ──────────────────────────────────────────────────────────────────────
Path('analysis').mkdir(exist_ok=True)
plt.savefig('analysis/figure_marker_layout.pdf', dpi=300, bbox_inches='tight')
plt.savefig('analysis/figure_marker_layout.png', dpi=300, bbox_inches='tight')
print('Saved: analysis/figure_marker_layout.pdf')
print('Saved: analysis/figure_marker_layout.png')
