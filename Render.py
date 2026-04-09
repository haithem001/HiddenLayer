import copy
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, HIDDEN
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import io
import time
import math

WIDTH, HEIGHT = 1600, 900

pygame.font.init()
font = pygame.font.SysFont("Arial", 16, bold=True)


def safe_line_width(width):
    width = abs(float(width))
    width = max(1.0, min(100.0, width))
    glLineWidth(width)

def safe_color(r, g, b, a=1.0):
    r = max(0.0, min(1.0, float(r)))
    g = max(0.0, min(1.0, float(g)))
    b = max(0.0, min(1.0, float(b)))
    a = max(0.0, min(1.0, float(a)))
    if a < 1.0:
        glColor4f(r, g, b, a)
    else:
        glColor3f(r, g, b)


def draw_cylinder(p1, p2, radius=0.08, color=(1.0, 1.0, 1.0), slices=8):
    try:
        x1, y1, z1 = p1['x'], p1['y'], p1['z']
        x2, y2, z2 = p2['x'], p2['y'], p2['z']
        dx = x2 - x1; dy = y2 - y1; dz = z2 - z1
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 0.001:
            return
        dx /= length; dy /= length; dz /= length

        glPushMatrix()
        glTranslatef(x1, y1, z1)
        angle_y = math.degrees(math.atan2(dx, dz))
        angle_x = math.degrees(math.asin(-dy))
        glRotatef(angle_y, 0, 1, 0)
        glRotatef(angle_x, 1, 0, 0)

        safe_color(color[0], color[1], color[2])
        q = gluNewQuadric()
        gluQuadricDrawStyle(q, GLU_FILL)
        gluQuadricNormals(q, GLU_SMOOTH)
        gluCylinder(q, radius, radius * 0.9, length, slices, 1)
        gluDeleteQuadric(q)

        q = gluNewQuadric()
        gluDisk(q, 0, radius, slices, 1)
        gluDeleteQuadric(q)

        glPushMatrix()
        glTranslatef(0, 0, length)
        q = gluNewQuadric()
        gluDisk(q, 0, radius * 0.9, slices, 1)
        gluDeleteQuadric(q)
        glPopMatrix()

        glPopMatrix()
    except Exception:
        try: glPopMatrix()
        except: pass


def draw_sphere(x, y, z, color=(1.0, 0.9, 0.7), radius=0.06, slices=8, stacks=8):
    try:
        glPushMatrix()
        glTranslatef(float(x), float(y), float(z))
        safe_color(color[0], color[1], color[2])
        q = gluNewQuadric()
        gluQuadricDrawStyle(q, GLU_FILL)
        gluSphere(q, radius, slices, stacks)
        gluDeleteQuadric(q)
        glPopMatrix()
    except Exception:
        try: glPopMatrix()
        except: pass


def _get_walker_centroid(points):
    if not points:
        return 0.0, 0.0
    xs = [p['x'] for p in points]
    zs = [p.get('z', 0.0) for p in points]
    return sum(xs) / len(xs), sum(zs) / len(zs)


def draw_trajectory(trajectory, scene_rot_y=0.0, cx=0.0, cz=0.0):
    if not trajectory or len(trajectory) < 2:
        return

    try:
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.5)

        n = len(trajectory)
        glBegin(GL_LINE_STRIP)
        for i, pt in enumerate(trajectory):
            t = i / max(n - 1, 1)

            r = t * 0.4
            g = 0.4 + t * 0.6
            b = 0.8 + t * 0.2
            a = 0.2 + t * 0.8

            glColor4f(
                max(0., min(1., r)),
                max(0., min(1., g)),
                max(0., min(1., b)),
                max(0., min(1., a)),
            )
            glVertex3f(float(pt['x']), float(pt['y']), float(pt['z']))
        glEnd()

        newest = trajectory[-1]
        glPointSize(6.0)
        glBegin(GL_POINTS)
        glColor4f(1.0, 1.0, 0.2, 1.0)
        glVertex3f(float(newest['x']), float(newest['y']), float(newest['z']))
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glPopAttrib()

    except Exception as e:
        print(f'Trajectory draw error: {e}')
        try:
            glEnd()
        except Exception:
            pass
        try:
            glEnable(GL_DEPTH_TEST)
            glPopAttrib()
        except Exception:
            pass


# ── Gaze / head drawing ───────────────────────────────────────────────

# Colours per gaze state  (R, G, B)
_STATE_COLOURS = {
    'searching': (0.12, 0.78, 0.60),   # teal
    'tracking':  (0.94, 0.62, 0.10),   # amber
    'pursuing':  (0.22, 0.72, 1.00),   # sky-blue
}
# Colours per gaze state for the FOV cone  (R, G, B, A)
_FOV_COLOURS = {
    'searching': (0.12, 0.78, 0.60, 0.10),
    'tracking':  (0.94, 0.62, 0.10, 0.20),
    'pursuing':  (0.22, 0.72, 1.00, 0.22),
}


def _draw_fov_cone(hx, hy, hz, yaw, state, half_angle,
                   cone_length=4.0, slices=24):
    """
    Draw the FOV cone projected flat onto the ground plane (y=0.02).
    Always visible regardless of whether the walker has fallen.
    The apex sits just above ground directly below the walker.
    """
    r, g, b, a = _FOV_COLOURS.get(state, (1, 1, 1, 0.1))
    # Always draw on the ground plane so it is readable even when fallen
    ground_y = 0.02

    try:
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)

        # Filled fan from walker XZ position, flat on the ground
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(r, g, b, a)
        glVertex3f(hx, ground_y, hz)
        for i in range(slices + 1):
            t = -1.0 + 2.0 * i / slices
            angle = yaw + t * half_angle
            fx =  math.sin(angle)
            fz = -math.cos(angle)
            glVertex3f(hx + fx * cone_length, ground_y, hz + fz * cone_length)
        glEnd()

        # Outline edges
        glLineWidth(1.2)
        glColor4f(r, g, b, min(a * 4, 1.0))
        glBegin(GL_LINES)
        for side in (-1, +1):
            edge_angle = yaw + side * half_angle
            glVertex3f(hx, ground_y, hz)
            glVertex3f(hx + math.sin(edge_angle) * cone_length,
                       ground_y,
                       hz - math.cos(edge_angle) * cone_length)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glPopAttrib()

    except Exception as e:
        print(f'FOV cone draw error: {e}')
        try:
            glEnd()
            glEnable(GL_DEPTH_TEST)
            glPopAttrib()
        except Exception:
            pass


def draw_head(hx, hy, hz, gaze_world_yaw, gaze_state, fov_half_angle,
              head_radius=0.18, eye_radius=0.055):
    """
    Draw a clearly readable head:
      • large coloured sphere (state colour)
      • prominent nose bump so rotation is obvious
      • two big white eyes with dark pupils on the face
      • a bold gaze ray shooting forward from the face
      • FOV cone on the ground
    """
    colour = _STATE_COLOURS.get(gaze_state, (0.9, 0.85, 0.80))
    hy = max(hy, 0.35)

    # Gaze direction vectors
    gfx =  math.sin(gaze_world_yaw)   # forward X
    gfz = -math.cos(gaze_world_yaw)   # forward Z
    grx =  gfz                         # right X
    grz = -gfx                         # right Z

    # ── Head sphere ────────────────────────────────────────────────────
    try:
        glPushMatrix()
        glTranslatef(hx, hy, hz)
        safe_color(*colour)
        q = gluNewQuadric()
        gluQuadricDrawStyle(q, GLU_FILL)
        gluQuadricNormals(q, GLU_SMOOTH)
        gluSphere(q, head_radius, 16, 16)
        gluDeleteQuadric(q)
        glPopMatrix()
    except Exception:
        try: glPopMatrix()
        except: pass

    # ── Nose — prominent bump on the face so rotation is obvious ───────
    nose_x = hx + gfx * head_radius * 0.95
    nose_y = hy
    nose_z = hz + gfz * head_radius * 0.95


    # ── Eyes — white sclera + dark pupil ───────────────────────────────


    # ── Gaze ray — bold line shooting out from the face ────────────────
    ray_len = 4.0 if gaze_state == 'searching' else 5.0
    r, g, b = colour
    try:
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(3.0 if gaze_state == 'searching' else 5.0)
        glBegin(GL_LINES)
        # Start at nose tip
        glColor4f(r, g, b, 1.0)
        glVertex3f(nose_x, nose_y, nose_z)
        # End fades out
        glColor4f(r, g, b, 0.0)
        glVertex3f(hx + gfx * ray_len, hy, hz + gfz * ray_len)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glPopAttrib()
    except Exception:
        try:
            glEnd()
            glEnable(GL_DEPTH_TEST)
            glPopAttrib()
        except: pass

    # ── FOV cone on the ground ─────────────────────────────────────────
    _draw_fov_cone(hx, hy, hz, gaze_world_yaw, gaze_state, fov_half_angle)


def draw_humanoid_heads(humans):
    """
    Draw heads + gaze cones for all living walkers.
    Head is always rendered at a fixed height above the torso XZ position,
    so it stays visible even when the walker has fallen.
    """
    for h in humans:
        if h.dead:
            continue
        if not hasattr(h, 'gaze'):
            continue
        try:
            ti  = h.joint_names['torso']
            tx  = float(h.points[ti]['x'])
            tz  = float(h.points[ti].get('z', 0.0))
            # Always place the head at a fixed height above the torso XZ,
            # regardless of whether the walker has fallen
            HEAD_DISPLAY_Y = 1.55   # roughly where the head is when standing
            draw_head(
                hx             = tx,
                hy             = HEAD_DISPLAY_Y,
                hz             = tz,
                gaze_world_yaw = h.gaze.world_yaw,
                gaze_state     = h.gaze.state,
                fov_half_angle = h.gaze.fov_half_angle,
            )
        except Exception as e:
            print(f'⚠️  Head draw: {e}')


# ── Main render entry point ────────────────────────────────────────────

def render_frame(camera_yaw, camera_pitch, camera_roll,
                 cam_x, cam_y, cam_z, points, lines, target, scene_rot_y=0.0,
                 draw_joint_spheres=False, trajectory=None,
                 population=None):            # ← population kwarg added
    try:
        if not points:
            blank = Image.new('RGB', (WIDTH, HEIGHT), 'black')
            buf = io.BytesIO(); blank.save(buf, format='JPEG'); return buf.getvalue()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

        cam_x = float(cam_x); cam_y = float(cam_y); cam_z = float(cam_z)
        camera_yaw   = float(camera_yaw)
        camera_pitch = max(min(float(camera_pitch), math.pi/2 - 0.01), -math.pi/2 + 0.01)

        fx = math.sin(camera_yaw) * math.cos(camera_pitch)
        fy = math.sin(camera_pitch)
        fz = -math.cos(camera_yaw) * math.cos(camera_pitch)

        gluLookAt(cam_x, cam_y, cam_z,
                  cam_x + fx, cam_y + fy, cam_z + fz,
                  0.0, 1.0, 0.0)

        draw_grid()

        cx, cz = _get_walker_centroid(points)

        glPushMatrix()
        glTranslatef(cx, 0.0, cz)
        glTranslatef(-cx, 0.0, -cz)

        # ── Trajectory trail ──────────────────────────────────────────
        if trajectory:
            draw_trajectory(trajectory, scene_rot_y=scene_rot_y, cx=cx, cz=cz)

        # ── Limbs (cylinders) ─────────────────────────────────────────
        if lines:
            for line in lines:
                p1 = points[line['from']]
                p2 = points[line['to']]
                draw_cylinder(p1, p2,
                              radius=line.get('radius', 0.06),
                              color=line.get('color', (1.0, 1.0, 1.0)))

        # ── Joint spheres ─────────────────────────────────────────────
        if draw_joint_spheres:
            for p in points:
                r = p.get('radius', 0.05)
                draw_sphere(p['x'], p['y'], p['z'], radius=r)

        # ── Heads + gaze cones ────────────────────────────────────────
        if population is not None:                               # ← ADDED
            draw_humanoid_heads(population.humans)               # ← ADDED

        glPopMatrix()

        # ── Target ────────────────────────────────────────────────────
        if target:
            draw_target(target['x'], target['y'], target['z'])

        draw_orientation_indicator(WIDTH, HEIGHT)
        pygame.display.flip()

        pixels = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        img = Image.frombytes('RGB', (WIDTH, HEIGHT), pixels)
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=82)
        return buf.getvalue()

    except Exception as e:
        print(f'Render error: {e}')
        import traceback; traceback.print_exc()
        blank = Image.new('RGB', (WIDTH, HEIGHT), 'black')
        buf = io.BytesIO(); blank.save(buf, format='JPEG'); return buf.getvalue()


def draw_target(x, y, z, radius=0.6, height=2.5, color=(0.2, 1.0, 0.2)):
    try:
        glPushMatrix()
        glTranslatef(x, 0.0, z)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glRotatef(-90, 1, 0, 0)

        safe_color(color[0], color[1], color[2], 0.55)
        q = gluNewQuadric()
        gluCylinder(q, radius * 0.55, radius * 0.55, height, 24, 1)
        gluDeleteQuadric(q)

        glDisable(GL_BLEND)
        glPopMatrix()

    except Exception as e:
        print(f'Target draw error: {e}')
        try: glPopMatrix()
        except: pass

def draw_grid(size=150, spacing=5):
    try:
        glLineWidth(1.0)
        glColor4f(0.25, 0.25, 0.25, 0.5)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBegin(GL_LINES)
        z = -size
        while z <= size:
            glVertex3f(-size, 0, z); glVertex3f(size, 0, z); z += spacing
        x = -size
        while x <= size:
            glVertex3f(x, 0, -size); glVertex3f(x, 0, size); x += spacing
        glEnd()
        glDisable(GL_BLEND)
    except Exception:
        try: glEnd(); glDisable(GL_BLEND)
        except: pass


def draw_orientation_indicator(viewport_width, viewport_height, size=50):
    try:
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, viewport_width, 0, viewport_height, -100, 100)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        glTranslatef(size, viewport_height - size, 0)
        glScalef(size / 3.0, size / 3.0, size / 3.0)
        rot = [[modelview[i][j] for j in range(3)] + [0] for i in range(3)] + [[0,0,0,1]]
        glMultMatrixf(rot)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.5)

        for axis, col in [((1.2,0,0),(1,0,0)), ((0,1.2,0),(0,1,0)), ((0,0,1.2),(0,0,1))]:
            glBegin(GL_LINES); glColor3f(*col); glVertex3f(0,0,0); glVertex3f(*axis); glEnd()

        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopAttrib()
    except Exception as e:
        print(f'Orientation indicator error: {e}')


def calculate_distance(p1, p2):
    dx = p1['x']-p2['x']; dy = p1['y']-p2['y']; dz = p1['z']-p2['z']
    return math.sqrt(dx*dx + dy*dy + dz*dz)
