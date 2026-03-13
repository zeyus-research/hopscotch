"""
Visual validation of joint angle calculations from joint_angles.py.

Shows a 2D stick figure with angle arcs overlaid at each joint,
with a frame slider so you can scrub through a trial and verify
the computed angles match the visible pose geometry.

Coordinate system (Qualisys/this dataset): Z is up, X is lateral, Y is forward/depth.
- Sagittal view (side): Y–Z plane
- Frontal view (front): X–Z plane
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Arc
from pathlib import Path

sys.path.insert(0, 'src')
from processing.mocap_loader import MocapDataLoader
from joint_angles import calculate_all_joint_angles


# ── skeleton connectivity ──────────────────────────────────────────────────────
BONES = [
    ('shoulder_r', 'elbow_r'), ('elbow_r', 'wrist_r'),
    ('shoulder_l', 'elbow_l'), ('elbow_l', 'wrist_l'),
    ('shoulder_r', 'shoulder_l'),
    ('shoulder_r', 'hip_front_r'), ('shoulder_l', 'hip_front_l'),
    ('hip_front_r', 'hip_front_l'), ('hip_back_r', 'hip_back_l'),
    ('hip_front_r', 'hip_back_r'), ('hip_front_l', 'hip_back_l'),
    ('hip_front_r', 'knee_over_r'), ('knee_over_r', 'knee_under_r'),
    ('knee_under_r', 'foot_front_r'), ('knee_under_r', 'foot_back_r'),
    ('foot_front_r', 'foot_back_r'),
    ('hip_front_l', 'knee_over_l'), ('knee_over_l', 'knee_under_l'),
    ('knee_under_l', 'foot_front_l'), ('knee_under_l', 'foot_back_l'),
    ('foot_front_l', 'foot_back_l'),
    ('head', 'shoulder_r'), ('head', 'shoulder_l'),
]

ALL_MARKERS = sorted(set(m for pair in BONES for m in pair))


def get_pos(row: pd.Series, marker: str) -> np.ndarray:
    return np.array([row[f'{marker}.X'], row[f'{marker}.Y'], row[f'{marker}.Z']])


def midpoint(row: pd.Series, m1: str, m2: str) -> np.ndarray:
    return (get_pos(row, m1) + get_pos(row, m2)) / 2.0


# ── joint centre computations matching joint_angles.py exactly ────────────────
def joint_centres(row: pd.Series) -> dict:
    """Compute the same joint centres used inside joint_angles.py."""
    return {
        # elbows
        'elbow_r':   get_pos(row, 'elbow_r'),
        'elbow_l':   get_pos(row, 'elbow_l'),
        # knees: midpoint of over+under
        'knee_r':    midpoint(row, 'knee_over_r', 'knee_under_r'),
        'knee_l':    midpoint(row, 'knee_over_l', 'knee_under_l'),
        # hips: midpoint of front+back
        'hip_r':     midpoint(row, 'hip_front_r', 'hip_back_r'),
        'hip_l':     midpoint(row, 'hip_front_l', 'hip_back_l'),
        # ankles: midpoint of foot front+back
        'ankle_r':   midpoint(row, 'foot_front_r', 'foot_back_r'),
        'ankle_l':   midpoint(row, 'foot_front_l', 'foot_back_l'),
        # shoulders
        'shoulder_r': get_pos(row, 'shoulder_r'),
        'shoulder_l': get_pos(row, 'shoulder_l'),
        # trunk
        'shoulder_mid': (get_pos(row, 'shoulder_r') + get_pos(row, 'shoulder_l')) / 2,
        'hip_mid':    (midpoint(row, 'hip_front_r', 'hip_back_r') +
                       midpoint(row, 'hip_front_l', 'hip_back_l')) / 2,
        # head
        'head':      get_pos(row, 'head'),
    }


# ── arc drawing ───────────────────────────────────────────────────────────────
def draw_angle_arc(ax, joint_2d, prox_2d, dist_2d, angle_deg,
                   color='red', radius_frac=0.20, label=True):
    """
    Draw a small arc at joint_2d spanning the angle between the two limb vectors.
    """
    v1 = prox_2d - joint_2d
    v2 = dist_2d - joint_2d
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if l1 < 1e-6 or l2 < 1e-6:
        return

    r = min(l1, l2) * radius_frac
    a1 = np.degrees(np.arctan2(v1[1], v1[0]))
    a2 = np.degrees(np.arctan2(v2[1], v2[0]))

    diff = (a2 - a1 + 360) % 360
    if diff > 180:
        a1, a2 = a2, a1  # always sweep the short way

    arc = Arc(joint_2d, 2 * r, 2 * r,
              angle=0, theta1=min(a1, a2), theta2=max(a1, a2),
              color=color, lw=2, zorder=5)
    ax.add_patch(arc)

    if label:
        mid_ang = np.radians((a1 + a2) / 2)
        lx = joint_2d[0] + r * 1.55 * np.cos(mid_ang)
        ly = joint_2d[1] + r * 1.55 * np.sin(mid_ang)
        ax.text(lx, ly, f'{angle_deg:.0f}°',
                fontsize=7, color=color, ha='center', va='center',
                fontweight='bold', zorder=6)


def proj2d_sag(p3d):   # side view: Y → x-axis, Z → y-axis
    return np.array([p3d[1], p3d[2]])

def proj2d_front(p3d): # front view: X → x-axis, Z → y-axis
    return np.array([p3d[0], p3d[2]])


def draw_joints(ax, row, jc, angles, proj):
    """Draw all joint arcs for one view projection."""

    def p(key):   # joint centre projected
        return proj(jc[key])

    def m(marker): # raw marker projected
        return proj(get_pos(row, marker))

    a = angles  # shorthand

    # ── elbow (joint_angles.py: pi - angle(shoulder→elbow, wrist→elbow)) ──────
    for side, color in [('r', 'tomato'), ('l', 'steelblue')]:
        col = f'elbow_{side}'
        if col in a:
            draw_angle_arc(ax, p(f'elbow_{side}'),
                           m(f'shoulder_{side}'), m(f'wrist_{side}'),
                           np.degrees(a[col]), color)

    # ── knee (joint_angles.py: pi - angle(hip→knee_over, ankle→knee_under)) ───
    # knee_over is on the thigh, knee_under on the shank — draw arc at their
    # midpoint, with each limb vector going to its own segment marker.
    for side, color in [('r', 'firebrick'), ('l', 'royalblue')]:
        col = f'knee_{side}'
        if col in a:
            knee_mid_2d  = (proj(get_pos(row, f'knee_over_{side}')) +
                            proj(get_pos(row, f'knee_under_{side}'))) / 2
            hip_2d       = p(f'hip_{side}')
            ankle_2d     = p(f'ankle_{side}')
            draw_angle_arc(ax, knee_mid_2d, hip_2d, ankle_2d,
                           np.degrees(a[col]), color)

    # ── ankle (joint_angles.py: angle(-shank, foot_front-foot_back)) ──────────
    for side, color in [('r', 'darkorange'), ('l', 'dodgerblue')]:
        col = f'ankle_{side}'
        if col in a:
            # prox direction: from ankle toward knee (= -shank)
            knee_2d  = p(f'knee_{side}')
            ankle_2d = p(f'ankle_{side}')
            # "proximal" point is knee (so vector = knee - ankle = -shank direction)
            foot_front_2d = m(f'foot_front_{side}')
            draw_angle_arc(ax, ankle_2d,
                           knee_2d, foot_front_2d,
                           np.degrees(a[col]), color)

    # ── shoulder elevation (joint_angles.py: arm direction vs torso frame) ────
    # Visualised as: trunk direction vs upper-arm direction at the shoulder marker.
    # The torso frame Y-axis ~ shoulder_mid → hip_mid (down), so we use hip_mid
    # as proximal and elbow as distal for a sensible arc.
    for side, color in [('r', 'darkorchid'), ('l', 'mediumseagreen')]:
        col = f'shoulder_{side}_elevation'
        if col in a:
            draw_angle_arc(ax, p(f'shoulder_{side}'),
                           p('hip_mid'), m(f'elbow_{side}'),
                           np.degrees(a[col]), color)

    # ── hip flexion (joint_angles.py: thigh vs torso frame Z-axis) ────────────
    for side, color in [('r', 'crimson'), ('l', 'teal')]:
        col = f'hip_{side}_flexion'
        if col in a:
            draw_angle_arc(ax, p(f'hip_{side}'),
                           p('shoulder_mid'), p(f'knee_{side}'),
                           np.degrees(a[col]), color)

    # ── hip abduction (better in frontal view, still drawn both) ──────────────
    for side, color in [('r', 'chocolate'), ('l', 'cadetblue')]:
        col = f'hip_{side}_abduction'
        if col in a:
            # Reference: vertical (hip_mid → shoulder_mid), distal: knee
            draw_angle_arc(ax, p(f'hip_{side}'),
                           p('hip_mid'), p(f'knee_{side}'),
                           np.degrees(a[col]), color,
                           radius_frac=0.12, label=False)  # small, no label (overlaps flex)

    # ── trunk lean (joint_angles.py: forward lean & lateral lean) ─────────────
    if 'trunk_forward_lean' in a:
        # Draw at hip_mid, prox = down (hip_mid + [0,0,-1] projected),
        # dist = shoulder_mid
        hip_2d = p('hip_mid')
        sho_2d = p('shoulder_mid')
        # "vertical" reference in the projected plane
        vert_3d = jc['hip_mid'] + np.array([0, 0, -200])  # 200 mm downward
        vert_2d = proj(vert_3d)
        draw_angle_arc(ax, hip_2d, vert_2d, sho_2d,
                       np.degrees(a['trunk_forward_lean']), 'gray',
                       radius_frac=0.12)

    # ── head (joint_angles.py: head_forward, head_lateral) ───────────────────
    if 'head_forward' in a:
        draw_angle_arc(ax, p('shoulder_mid'),
                       p('hip_mid'), p('head'),
                       np.degrees(a['head_forward']), 'olive',
                       radius_frac=0.15)


# ── frame renderer ────────────────────────────────────────────────────────────
def plot_frame(trial_df, angles_df, frame_idx, axes, title_prefix=''):
    row = trial_df.iloc[frame_idx]
    ang = angles_df.iloc[frame_idx]
    jc  = joint_centres(row)

    ax_sag, ax_front = axes
    ax_sag.cla(); ax_front.cla()

    for proj, ax, view_label in [
        (proj2d_sag,   ax_sag,   'Side view (Y–Z)'),
        (proj2d_front, ax_front, 'Front view (X–Z)'),
    ]:
        # Bones
        for m1, m2 in BONES:
            try:
                p1 = proj(get_pos(row, m1))
                p2 = proj(get_pos(row, m2))
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        'k-', lw=1.5, alpha=0.5, zorder=2)
            except KeyError:
                pass

        # Markers
        for marker in ALL_MARKERS:
            try:
                p = proj(get_pos(row, marker))
                ax.plot(p[0], p[1], 'ko', ms=3.5, zorder=3)
            except KeyError:
                pass

        # Computed joint centres (shown as open circles)
        for key in ('knee_r', 'knee_l', 'ankle_r', 'ankle_l', 'hip_r', 'hip_l',
                    'shoulder_mid', 'hip_mid'):
            p = proj(jc[key])
            ax.plot(p[0], p[1], 'o', ms=5, mfc='none', mec='gray',
                    mew=1.2, zorder=4)

        # Angle arcs
        draw_joints(ax, row, jc, ang, proj)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        t_val = row.get('Time', frame_idx)
        ax.set_title(f'{title_prefix} | {view_label}\n'
                     f'Frame {frame_idx}  t={t_val:.3f}s', fontsize=9)
        ax.set_xlabel('mm'); ax.set_ylabel('Z (mm)')

    # Legend with degree values
    def deg(col):
        v = ang.get(col, np.nan)
        return np.degrees(v) if not np.isnan(v) else float('nan')

    patches = [
        mpatches.Patch(color='firebrick',    label=f'Knee R      {deg("knee_r"):.0f}°'),
        mpatches.Patch(color='royalblue',    label=f'Knee L      {deg("knee_l"):.0f}°'),
        mpatches.Patch(color='darkorange',   label=f'Ankle R     {deg("ankle_r"):.0f}°'),
        mpatches.Patch(color='dodgerblue',   label=f'Ankle L     {deg("ankle_l"):.0f}°'),
        mpatches.Patch(color='tomato',       label=f'Elbow R     {deg("elbow_r"):.0f}°'),
        mpatches.Patch(color='steelblue',    label=f'Elbow L     {deg("elbow_l"):.0f}°'),
        mpatches.Patch(color='darkorchid',   label=f'Shoulder R elev {deg("shoulder_r_elevation"):.0f}°'),
        mpatches.Patch(color='mediumseagreen', label=f'Shoulder L elev {deg("shoulder_l_elevation"):.0f}°'),
        mpatches.Patch(color='crimson',      label=f'Hip R flex  {deg("hip_r_flexion"):.0f}°'),
        mpatches.Patch(color='teal',         label=f'Hip L flex  {deg("hip_l_flexion"):.0f}°'),
        mpatches.Patch(color='gray',         label=f'Trunk lean  {deg("trunk_forward_lean"):.0f}°'),
        mpatches.Patch(color='olive',        label=f'Head fwd    {deg("head_forward"):.0f}°'),
    ]
    ax_sag.legend(handles=patches, fontsize=6.5, loc='upper left',
                  ncol=2, framealpha=0.85)


# ── time-series panel ─────────────────────────────────────────────────────────
TS_CHANNELS = [
    # (column, label, color)
    ('knee_r',              'Knee R',       'firebrick'),
    ('knee_l',              'Knee L',       'royalblue'),
    ('ankle_r',             'Ankle R',      'darkorange'),
    ('ankle_l',             'Ankle L',      'dodgerblue'),
    ('elbow_r',             'Elbow R',      'tomato'),
    ('elbow_l',             'Elbow L',      'steelblue'),
    ('hip_r_flexion',       'Hip R flex',   'crimson'),
    ('hip_l_flexion',       'Hip L flex',   'teal'),
    ('shoulder_r_elevation','Shldr R elev', 'darkorchid'),
    ('trunk_forward_lean',  'Trunk lean',   'gray'),
]


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate joint angles from joint_angles.py')
    parser.add_argument('--subject', type=int, default=0,
                        help='Subject index (0-based)')
    parser.add_argument('--cache', default='hopscotch_data_no_floor2.feather')
    args = parser.parse_args()

    print("Loading data...")
    loader = MocapDataLoader(Path('marker_labels_no_floor.txt'))
    raw_data = loader.load_dataset(
        data_path=Path('data'), file_pattern='*.tsv', cache_file=args.cache
    )

    subjects = raw_data['filename'].unique()
    name = subjects[args.subject]
    print(f"Subject [{args.subject}]: {name}  ({args.subject+1}/{len(subjects)})")

    trial_df  = raw_data[raw_data['filename'] == name].reset_index(drop=True)
    print(f"  {len(trial_df)} frames — calculating joint angles...")
    angles_df = calculate_all_joint_angles(trial_df).reset_index(drop=True)
    n_frames  = len(trial_df)

    # ── layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Joint angle validation (joint_angles.py) — {name}', fontsize=11)

    ax_sag   = fig.add_axes([0.03, 0.28, 0.44, 0.66])
    ax_front = fig.add_axes([0.52, 0.28, 0.44, 0.66])
    ax_ts    = fig.add_axes([0.06, 0.10, 0.88, 0.14])

    ax_slider = fig.add_axes([0.06, 0.03, 0.70, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1,
                    valinit=0, valstep=1, color='steelblue')

    ax_prev = fig.add_axes([0.79, 0.02, 0.06, 0.05])
    ax_next = fig.add_axes([0.86, 0.02, 0.06, 0.05])
    btn_prev = Button(ax_prev, '◀ Prev')
    btn_next = Button(ax_next, 'Next ▶')

    # ── static time series ─────────────────────────────────────────────────────
    time = angles_df['time'].values if 'time' in angles_df.columns else np.arange(n_frames)
    for col, label, color in TS_CHANNELS:
        if col in angles_df.columns:
            ax_ts.plot(time, np.degrees(angles_df[col]),
                       color=color, lw=1, alpha=0.8, label=label)
    ax_ts.set_ylabel('°', fontsize=8)
    ax_ts.legend(fontsize=6.5, loc='upper right', ncol=5)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_xlabel('time (s)' if 'time' in angles_df.columns else 'frame')
    vline = ax_ts.axvline(time[0], color='black', lw=1.5, ls='--', zorder=10)

    # ── update ─────────────────────────────────────────────────────────────────
    def update(val):
        fi = int(slider.val)
        plot_frame(trial_df, angles_df, fi, [ax_sag, ax_front], title_prefix=name)
        vline.set_xdata([time[fi], time[fi]])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    btn_prev.on_clicked(lambda _: slider.set_val(max(0, int(slider.val) - 1)))
    btn_next.on_clicked(lambda _: slider.set_val(min(n_frames - 1, int(slider.val) + 1)))

    def on_key(event):
        if   event.key == 'right': slider.set_val(min(n_frames - 1, int(slider.val) + 1))
        elif event.key == 'left':  slider.set_val(max(0, int(slider.val) - 1))
        elif event.key == 'home':  slider.set_val(0)
        elif event.key == 'end':   slider.set_val(n_frames - 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update(0)

    print("\nControls: slider / Prev–Next buttons / ← → arrow keys / Home / End")
    print("Open circles = computed joint centres (midpoints used by joint_angles.py)")
    print("Sanity checks:")
    print("  Standing upright  → knee ~0°, ankle ~90°, hip flex ~0°")
    print("  Knee bent clearly → knee arc should look ~90°+")
    print("  Arm at side       → shoulder elevation ~0°")

    plt.show(block=True)


if __name__ == '__main__':
    main()
