import math
import random

# ── Human Proportions (~1.75 m total) ─────────────────────────────────
FOOT_LEN    = 0.06   # ankle height above ground
LEG_LEN     = 0.42   # shin / lower leg
THIGH_LEN   = 0.44   # thigh / upper leg
TORSO_LEN   = 0.50   # hip centre → shoulder centre
NECK_LEN    = 0.18   # shoulder centre → head centre
ARM_LEN     = 0.68   # shoulder → hand
HIP_W       = 0.09   # half hip width (z-offset)
SHOULDER_W  = 0.13   # half shoulder width (z-offset)

# Masses (kg) — default ratios sum to ~12
MASS_TORSO  = 2.5    # pelvis / lower torso
MASS_CHEST  = 2.5    # upper torso / shoulder mass
MASS_THIGH  = 1.3
MASS_LEG    = 0.8
MASS_FOOT   = 0.4
MASS_ARM    = 0.6
# Head is kinematic: mass = 0, no constraint


def create_walker(base_x=0.0, base_z=0.0, total_mass=12.0, noise_scale=0.005):

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

    # ── Torso (hip) & Chest (shoulder) ──────────────────────────────
    torso_y = FOOT_LEN + LEG_LEN + THIGH_LEN + rn()
    torso   = add_pt(base_x, torso_y,            base_z, MASS_TORSO, 0.13)
    chest_y = torso_y + TORSO_LEN
    chest   = add_pt(base_x, chest_y,            base_z, MASS_CHEST, 0.13)
    head    = add_pt(base_x, chest_y + NECK_LEN, base_z, 0.0,        0.10)

    # ── RIGHT leg chain ─────────────────────────────────────────────
    r_foot  = add_pt(base_x, 0.0,              base_z - HIP_W, MASS_FOOT,  0.055)
    r_leg   = add_pt(base_x, FOOT_LEN,         base_z - HIP_W, MASS_LEG,   0.065)
    r_thigh = add_pt(base_x, FOOT_LEN+LEG_LEN, base_z - HIP_W, MASS_THIGH, 0.085)

    # ── LEFT leg chain ──────────────────────────────────────────────
    l_foot  = add_pt(base_x, 0.0,              base_z + HIP_W, MASS_FOOT,  0.055)
    l_leg   = add_pt(base_x, FOOT_LEN,         base_z + HIP_W, MASS_LEG,   0.065)
    l_thigh = add_pt(base_x, FOOT_LEN+LEG_LEN, base_z + HIP_W, MASS_THIGH, 0.085)

    # ── ARMS (attached to chest/shoulder) ───────────────────────────
    r_arm = add_pt(base_x, chest_y - ARM_LEN, base_z - SHOULDER_W, MASS_ARM, 0.045)
    l_arm = add_pt(base_x, chest_y - ARM_LEN, base_z + SHOULDER_W, MASS_ARM, 0.045)

    # ── Bone constraints ────────────────────────────────────────────
    add_bone(torso, chest)   # spine

    add_bone(torso, r_thigh)
    add_bone(r_thigh, r_leg)
    add_bone(r_leg, r_foot)

    add_bone(torso, l_thigh)
    add_bone(l_thigh, l_leg)
    add_bone(l_leg, l_foot)

    add_bone(chest, r_arm)
    add_bone(chest, l_arm)

    # ── Visual lines ────────────────────────────────────────────────
    TORSO_C = (0.9, 0.8, 0.7)
    RIGHT_C = (0.5, 0.75, 1.0)
    LEFT_C  = (1.0, 0.6,  0.7)
    NECK_C  = (0.85, 0.80, 0.75)
    ARM_C   = (0.85, 0.75, 0.55)

    add_ln(torso, chest,   TORSO_C, 0.13)  # spine
    add_ln(chest, head,    NECK_C,  0.07)  # neck

    add_ln(torso, r_thigh, TORSO_C, 0.11)
    add_ln(r_thigh, r_leg, RIGHT_C, 0.08)
    add_ln(r_leg, r_foot,  RIGHT_C, 0.06)

    add_ln(torso, l_thigh, TORSO_C, 0.11)
    add_ln(l_thigh, l_leg, LEFT_C,  0.08)
    add_ln(l_leg, l_foot,  LEFT_C,  0.06)

    add_ln(chest, r_arm, ARM_C, 0.06)
    add_ln(chest, l_arm, ARM_C, 0.06)

    # ── Scale masses to requested total_mass ────────────────────────
    current_mass = sum(p['mass'] for p in pts)
    if current_mass > 0 and abs(total_mass - current_mass) > 1e-6:
        scale = total_mass / current_mass
        for p in pts:
            if p['mass'] > 0:
                p['mass'] *= scale

    joint_names = {
        'torso':   torso,   # hip / pelvis  (root body)
        'chest':   chest,   # shoulder / upper torso
        'head':    head,
        'r_thigh': r_thigh,
        'r_leg':   r_leg,
        'r_foot':  r_foot,
        'l_thigh': l_thigh,
        'l_leg':   l_leg,
        'l_foot':  l_foot,
        'r_arm':   r_arm,
        'l_arm':   l_arm,
    }

    return pts, lines, joint_names, constr
