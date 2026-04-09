import math
import random

# ── Geometry ──────────────────────────────────────────────────────────
TORSO_Y   = 1.25
THIGH_LEN = 0.45
LEG_LEN   = 0.40
FOOT_LEN  = 0.20
NECK_LEN  = 0.22   # vertical offset from torso centre to head centre

MASS_TORSO  = 6.0
MASS_THIGH  = 1.2
MASS_LEG    = 0.8
MASS_FOOT   = 0.3
# Head has NO physics mass — it is pinned kinematically to the torso
# so it never droops and adds zero solver cost.


def create_walker(base_x=0.0, base_z=0.0, total_mass=12.0, noise_scale=0.005):
    """
    Returns (points, lines, joint_names, constraints).

    The 'head' joint is KINEMATIC: it carries mass=0 and has no bone
    constraint.  _SingleWalker._pin_head() repositions it every step
    directly above the torso, so it is purely a render / gaze anchor.

    Leg constraints: only within each chain (torso->thigh->leg->foot).
    No cross-leg bracing.
    """

    def rn():
        return random.uniform(-noise_scale, noise_scale)

    pts    = []
    lines  = []
    constr = []

    def add_pt(x, y, z, mass, radius):
        idx = len(pts)
        pts.append({'x': float(x), 'y': float(y), 'z': float(z),
                    'mass': float(mass), 'radius': float(radius)})
        return idx

    def add_ln(i, j, color, r=0.05):
        lines.append({'from': i, 'to': j, 'color': color, 'radius': r})

    def add_bone(i, j):
        dx   = pts[i]['x'] - pts[j]['x']
        dy   = pts[i]['y'] - pts[j]['y']
        dz   = pts[i].get('z', 0.0) - pts[j].get('z', 0.0)
        dist = max(math.sqrt(dx*dx + dy*dy + dz*dz), 1e-3)
        constr.append({'joint1': i, 'joint2': j,
                       'rest_length': dist, 'stiffness': 1.0})

    # ── Torso ─────────────────────────────────────────────────────────
    torso_y = FOOT_LEN + LEG_LEN + THIGH_LEN + rn()
    torso   = add_pt(base_x, torso_y,            base_z, MASS_TORSO, 0.15)

    # ── Head — kinematic, mass=0, no constraint ────────────────────────
    head    = add_pt(base_x, torso_y + NECK_LEN, base_z, 0.0,        0.11)

    Z = 0.02  # z-offset so left/right chains render distinctly

    # ── RIGHT leg chain ───────────────────────────────────────────────
    r_foot  = add_pt(base_x, 0.0,              base_z,   MASS_FOOT,  0.07)
    r_leg   = add_pt(base_x, FOOT_LEN,         base_z,   MASS_LEG,   0.07)
    r_thigh = add_pt(base_x, FOOT_LEN+LEG_LEN, base_z,   MASS_THIGH, 0.09)

    # ── LEFT leg chain ────────────────────────────────────────────────
    l_foot  = add_pt(base_x, 0.0,              base_z+Z, MASS_FOOT,  0.07)
    l_leg   = add_pt(base_x, FOOT_LEN,         base_z+Z, MASS_LEG,   0.07)
    l_thigh = add_pt(base_x, FOOT_LEN+LEG_LEN, base_z+Z, MASS_THIGH, 0.09)

    # ── Bone constraints — legs only, NO neck ─────────────────────────
    add_bone(torso,   r_thigh)
    add_bone(r_thigh, r_leg)
    add_bone(r_leg,   r_foot)

    add_bone(torso,   l_thigh)
    add_bone(l_thigh, l_leg)
    add_bone(l_leg,   l_foot)

    # ── Visual lines ──────────────────────────────────────────────────
    TORSO_C = (0.9, 0.8, 0.7)
    RIGHT_C = (0.5, 0.75, 1.0)
    LEFT_C  = (1.0, 0.6,  0.7)
    NECK_C  = (0.85, 0.80, 0.75)

    # neck line omitted — head sphere is drawn separately by draw_humanoid_heads
    add_ln(torso,   r_thigh, TORSO_C, 0.12)
    add_ln(r_thigh, r_leg,   RIGHT_C, 0.09)
    add_ln(r_leg,   r_foot,  RIGHT_C, 0.07)
    add_ln(torso,   l_thigh, TORSO_C, 0.12)
    add_ln(l_thigh, l_leg,   LEFT_C,  0.09)
    add_ln(l_leg,   l_foot,  LEFT_C,  0.07)

    joint_names = {
        'torso':   torso,
        'head':    head,
        'r_thigh': r_thigh,
        'r_leg':   r_leg,
        'r_foot':  r_foot,
        'l_thigh': l_thigh,
        'l_leg':   l_leg,
        'l_foot':  l_foot,
    }

    return pts, lines, joint_names, constr
