import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, HIDDEN
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image, ImageDraw
import io
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
            glColor4f(t * 0.4, 0.4 + t * 0.6, 0.8 + t * 0.2, 0.2 + t * 0.8)
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
        try: glEnd(); glEnable(GL_DEPTH_TEST); glPopAttrib()
        except: pass


# ── Gaze / head drawing ────────────────────────────────────────────────

_STATE_COLOURS = {
    'searching': (0.12, 0.78, 0.60),
    'tracking':  (0.94, 0.62, 0.10),
    'pursuing':  (0.22, 0.72, 1.00),
}
_FOV_COLOURS = {
    'searching': (0.12, 0.78, 0.60, 0.10),
    'tracking':  (0.94, 0.62, 0.10, 0.20),
    'pursuing':  (0.22, 0.72, 1.00, 0.22),
}


def _draw_fov_cone(hx, hy, hz, yaw, state, half_angle, cone_length=4.0, slices=24):
    r, g, b, a = _FOV_COLOURS.get(state, (1, 1, 1, 0.1))
    ground_y = 0.02
    try:
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(r, g, b, a)
        glVertex3f(hx, ground_y, hz)
        for i in range(slices + 1):
            t     = -1.0 + 2.0 * i / slices
            angle = yaw + t * half_angle
            glVertex3f(hx + math.sin(angle) * cone_length,
                       ground_y,
                       hz - math.cos(angle) * cone_length)
        glEnd()
        glLineWidth(1.2)
        glColor4f(r, g, b, min(a * 4, 1.0))
        glBegin(GL_LINES)
        for side in (-1, +1):
            ea = yaw + side * half_angle
            glVertex3f(hx, ground_y, hz)
            glVertex3f(hx + math.sin(ea) * cone_length,
                       ground_y, hz - math.cos(ea) * cone_length)
        glEnd()
        glEnable(GL_DEPTH_TEST)
        glPopAttrib()
    except Exception as e:
        print(f'FOV cone draw error: {e}')
        try: glEnd(); glEnable(GL_DEPTH_TEST); glPopAttrib()
        except: pass


def draw_head(hx, hy, hz, gaze_world_yaw, gaze_state, fov_half_angle,
              head_radius=0.18):
    colour = _STATE_COLOURS.get(gaze_state, (0.9, 0.85, 0.80))
    hy = max(hy, 0.35)
    gfx =  math.sin(gaze_world_yaw)
    gfz = -math.cos(gaze_world_yaw)

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

    nose_x = hx + gfx * head_radius * 0.95
    nose_y = hy
    nose_z = hz + gfz * head_radius * 0.95
    ray_len = 4.0 if gaze_state == 'searching' else 5.0
    r, g, b = colour
    try:
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(3.0 if gaze_state == 'searching' else 5.0)
        glBegin(GL_LINES)
        glColor4f(r, g, b, 1.0)
        glVertex3f(nose_x, nose_y, nose_z)
        glColor4f(r, g, b, 0.0)
        glVertex3f(hx + gfx * ray_len, hy, hz + gfz * ray_len)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)
        glPopAttrib()
    except Exception:
        try: glEnd(); glEnable(GL_DEPTH_TEST); glPopAttrib()
        except: pass

    _draw_fov_cone(hx, hy, hz, gaze_world_yaw, gaze_state, fov_half_angle)


def draw_humanoid_heads(humans):
    for h in humans:
        if h.dead or not hasattr(h, 'gaze'):
            continue
        try:
            ti = h.joint_names['torso']
            tx = float(h.points[ti]['x'])
            tz = float(h.points[ti].get('z', 0.0))
            draw_head(
                hx             = tx,
                hy             = 1.55,
                hz             = tz,
                gaze_world_yaw = h.gaze.world_yaw,
                gaze_state     = h.gaze.state,
                fov_half_angle = h.gaze.fov_half_angle,
            )
        except Exception as e:
            print(f'Head draw error: {e}')


# ── Vision HUD overlay ─────────────────────────────────────────────────
#
# D = full visual field (outermost rect, screen-filling)
# C = clear-vision zone (70 % of D)
# B = fovea / tight zone  (matches FOV_SEARCH_HALF ≈ 10 °)
# A = pupil circle — position encodes WHERE the eye is looking
#
# Pupil position per gaze state
# ─────────────────────────────
# searching  → pupil drifts across D following the head-sweep angle.
#              The sweep offset comes from eye_data['sweep_x'] which
#              main.py writes from HeadGaze._sweep_offset.
# tracking   → pupil moves toward the detected target position (retinal_x/y
#              mapped into D-pixel space).
# pursuing   → pupil locked on target, contracts with proximity + lux.
#
# Light
# ─────
# bright (high lux) → strong dark vignette at edges, small pupil
# dark              → no vignette, large dilated pupil

_HUD_COLOURS = {
    'searching': (80,  200, 255),
    'tracking':  (80,  255, 160),
    'pursuing':  (255, 200,  60),
}


def _rect_bbox(cx, cy, w, h):
    return [int(cx - w / 2), int(cy - h / 2),
            int(cx + w / 2), int(cy + h / 2)]


def _draw_vignette(d, iw, ih, vignette_alpha):
    """Radial dark vignette at frame edges — stronger in bright scenes."""
    if vignette_alpha < 5:
        return
    cx, cy  = iw // 2, ih // 2
    r_outer = int(math.sqrt(cx*cx + cy*cy) * 1.05)
    r_inner = int(min(cx, cy) * 0.40)
    steps   = 20
    for i in range(steps):
        t   = i / steps
        t2  = (i + 1) / steps
        r1  = r_inner + int((r_outer - r_inner) * t)
        r2  = r_inner + int((r_outer - r_inner) * t2)
        mid_a = int(vignette_alpha * ((t + t2) / 2) ** 1.8)
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2],
                  outline=(0, 0, 0, mid_a), width=max(1, r2 - r1))


def draw_eye_overlay(img: Image.Image, eye_data: dict) -> Image.Image:
    """
    Composite the full vision HUD onto img.
    eye_data keys — all from Eye.get_render_data() plus 'sweep_x'/'sweep_y'
    added by main.py:
      cx, cy          – pupil pixel offset from screen centre
      radius          – pupil radius in pixels
      state           – 'searching' | 'tracking' | 'pursuing'
      target_in_fov   – bool
      ambient_lux     – 0-1  (from LightModel)
      pupil_dilation  – 0=contracted … 1=dilated
      vignette_alpha  – 0-255 vignette strength
      sweep_x, sweep_y– normalised [-1,1] head-sweep position (searching only)
      D_w,D_h,C_w,C_h,B_w,B_h  – zone pixel sizes
    """
    if not eye_data:
        return img

    try:
        state  = eye_data.get('state', 'searching')
        cr, cg, cb = _HUD_COLOURS.get(state, (200, 200, 200))

        D_w = float(eye_data['D_w']);  D_h = float(eye_data['D_h'])
        C_w = float(eye_data['C_w']);  C_h = float(eye_data['C_h'])
        B_w = float(eye_data['B_w']);  B_h = float(eye_data['B_h'])

        p_off_x = float(eye_data['cx'])
        p_off_y = float(eye_data['cy'])
        p_r     = float(eye_data['radius'])

        target_in_fov  = eye_data.get('target_in_fov', False)
        vignette_alpha = int(eye_data.get('vignette_alpha', 80))
        sweep_x        = float(eye_data.get('sweep_x', 0.0))
        sweep_y        = float(eye_data.get('sweep_y', 0.0))

        iw, ih = img.size
        hud_cx = iw // 2
        hud_cy = ih // 2

        layer = Image.new('RGBA', (iw, ih), (0, 0, 0, 0))
        d     = ImageDraw.Draw(layer)

        # ── Vignette ──────────────────────────────────────────────────
        _draw_vignette(d, iw, ih, vignette_alpha)

        fill_alpha    = 12
        outline_alpha = 155
        lw            = 2

        # ── D rect ───────────────────────────────────────────────────
        d.rectangle(_rect_bbox(hud_cx, hud_cy, D_w, D_h),
                    fill=(cr, cg, cb, fill_alpha),
                    outline=(cr, cg, cb, outline_alpha), width=lw)

        # ── C rect ───────────────────────────────────────────────────
        d.rectangle(_rect_bbox(hud_cx, hud_cy, C_w, C_h),
                    fill=(cr, cg, cb, fill_alpha),
                    outline=(cr, cg, cb, outline_alpha), width=lw)

        # ── B rect (fovea) — brightens when target is inside ─────────
        b_fill_a = 40 if target_in_fov else fill_alpha
        d.rectangle(_rect_bbox(hud_cx, hud_cy, B_w, B_h),
                    fill=(cr, cg, cb, b_fill_a),
                    outline=(cr, cg, cb, outline_alpha), width=lw)

        # Crosshair inside B
        bx0 = hud_cx - B_w / 2;  bx1 = hud_cx + B_w / 2
        by0 = hud_cy - B_h / 2;  by1 = hud_cy + B_h / 2
        d.line([(bx0, hud_cy), (bx1, hud_cy)], fill=(cr, cg, cb, 30), width=1)
        d.line([(hud_cx, by0), (hud_cx, by1)], fill=(cr, cg, cb, 30), width=1)

        # ── Sweep trail (searching) ───────────────────────────────────
        # Show where the head is currently pointing inside D
        if state == 'searching':
            sw_x = hud_cx + sweep_x * (D_w / 2.0)
            sw_y = hud_cy + sweep_y * (D_h / 2.0)
            sw_r = max(6, p_r * 0.5)
            # Faint sweep cursor
            d.ellipse([sw_x - sw_r, sw_y - sw_r, sw_x + sw_r, sw_y + sw_r],
                      fill=None, outline=(cr, cg, cb, 55), width=1)
            # Line from B-centre to sweep cursor
            d.line([(hud_cx, hud_cy), (int(sw_x), int(sw_y))],
                   fill=(cr, cg, cb, 30), width=1)

        # ── A — pupil ─────────────────────────────────────────────────
        pu_cx = hud_cx + p_off_x
        pu_cy = hud_cy + p_off_y

        pupil_fill_a = 90 if target_in_fov else 35
        d.ellipse([pu_cx - p_r, pu_cy - p_r, pu_cx + p_r, pu_cy + p_r],
                  fill=(cr, cg, cb, pupil_fill_a),
                  outline=(cr, cg, cb, outline_alpha), width=lw)

        # Centre focus dot
        dot_r = max(4, p_r * 0.22)
        d.ellipse([pu_cx - dot_r, pu_cy - dot_r,
                   pu_cx + dot_r, pu_cy + dot_r],
                  fill=(cr, cg, cb, 235))

        # Detection crosshair when locked
        if target_in_fov and state in ('tracking', 'pursuing'):
            ch = int(p_r * 0.9)
            d.line([(int(pu_cx - ch), int(pu_cy)), (int(pu_cx + ch), int(pu_cy))],
                   fill=(cr, cg, cb, 210), width=1)
            d.line([(int(pu_cx), int(pu_cy - ch)), (int(pu_cx), int(pu_cy + ch))],
                   fill=(cr, cg, cb, 210), width=1)

        # ── Zone labels ───────────────────────────────────────────────
        try:
            from PIL import ImageFont
            fnt = ImageFont.load_default()
            for label, lx, ly in [
                ('D', hud_cx + D_w / 2 - 12, hud_cy - D_h / 2 + 3),
                ('C', hud_cx + C_w / 2 - 12, hud_cy - C_h / 2 + 3),
                ('B', hud_cx + B_w / 2 - 12, hud_cy - B_h / 2 + 3),
            ]:
                d.text((int(lx), int(ly)), label,
                       fill=(cr, cg, cb, 110), font=fnt)
        except Exception:
            pass

        # ── Info text (bottom centre) ─────────────────────────────────
        try:
            from PIL import ImageFont
            fnt  = ImageFont.load_default()
            lux  = float(eye_data.get('ambient_lux',    0.5))
            dil  = float(eye_data.get('pupil_dilation',  0.5))
            dist = float(eye_data.get('target_dist',     0.0))
            line1 = f"{state.upper()}   lux {lux:.2f}   dil {dil:.2f}"
            line2 = f"{dist:.1f} m" if state == 'pursuing' else ""

            for li, txt in enumerate([line1, line2]):
                if not txt:
                    continue
                bbox = d.textbbox((0, 0), txt, font=fnt)
                tw   = bbox[2] - bbox[0]
                ty   = ih - 22 + li * 14
                d.rectangle([hud_cx - tw//2 - 4, ty - 2,
                             hud_cx + tw//2 + 4, ty + 12],
                            fill=(0, 0, 0, 110))
                d.text((hud_cx - tw // 2, ty), txt,
                       fill=(cr, cg, cb, 200), font=fnt)
        except Exception:
            pass

        img = img.convert('RGBA')
        img = Image.alpha_composite(img, layer)
        img = img.convert('RGB')

    except Exception as e:
        print(f'Vision HUD error: {e}')

    return img


# ── Main render entry point ────────────────────────────────────────────
#
# ALWAYS returns a 2-tuple: (jpeg_bytes: bytes, retinal_result: dict | None)
# retinal_result = {'detected': bool, 'retinal_x': float, 'retinal_y': float}
#                  or None in overview mode

def render_frame(camera_yaw, camera_pitch, camera_roll,
                 cam_x, cam_y, cam_z, points, lines, target, scene_rot_y=0.0,
                 draw_joint_spheres=False, trajectory=None,
                 population=None, eye_overlay=None,
                 eye_camera=None, light_model=None,
                 head_pos=None, gaze_yaw=None, gaze_pitch=None, fov_deg=None):

    retinal_result = None

    try:
        if not points:
            blank = Image.new('RGB', (WIDTH, HEIGHT), 'black')
            buf   = io.BytesIO()
            blank.save(buf, format='JPEG')
            return buf.getvalue(), None

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

        if trajectory:
            draw_trajectory(trajectory, scene_rot_y=scene_rot_y, cx=cx, cz=cz)

        if lines:
            for line in lines:
                p1 = points[line['from']]
                p2 = points[line['to']]
                draw_cylinder(p1, p2,
                              radius=line.get('radius', 0.06),
                              color=line.get('color', (1.0, 1.0, 1.0)))

        if draw_joint_spheres:
            for p in points:
                draw_sphere(p['x'], p['y'], p['z'], radius=p.get('radius', 0.05))

        if population is not None:
            draw_humanoid_heads(population.humans)

        glPopMatrix()

        if target:
            draw_target(target['x'], target['y'], target['z'])

        draw_orientation_indicator(WIDTH, HEIGHT)
        if ( # you can expose first_person_mode globally or pass it
                eye_camera is not None and light_model is not None and
                head_pos is not None and gaze_yaw is not None and
                gaze_pitch is not None and fov_deg is not None):

            hx, hy, hz = head_pos

            # Define the retinal panel size and position (adjust as you like)
            PANEL_W = 480          # or RETINA_W * 3 for nicer quality
            PANEL_H = 270          # keep ~16:9
            PANEL_X = 20           # left margin
            PANEL_Y = 20           # bottom margin (OpenGL Y=0 is bottom)

            glViewport(PANEL_X, PANEL_Y, PANEL_W, PANEL_H)

            # Optional: clear only the panel area with a dark background
            glScissor(PANEL_X, PANEL_Y, PANEL_W, PANEL_H)
            glEnable(GL_SCISSOR_TEST)
            glClearColor(0.03, 0.03, 0.05, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_SCISSOR_TEST)

            # Set up projection for the retinal view
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            aspect = PANEL_W / PANEL_H
            gluPerspective(fov_deg, aspect, 0.05, 200.0)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            # Head look direction
            fx = math.sin(gaze_yaw) * math.cos(gaze_pitch)
            fy = math.sin(gaze_pitch)
            fz = -math.cos(gaze_yaw) * math.cos(gaze_pitch)

            gluLookAt(hx, hy, hz,
                      hx + fx, hy + fy, hz + fz,
                      0.0, 1.0, 0.0)

            glEnable(GL_DEPTH_TEST)

            # Reuse the same minimal or full drawing code
            if hasattr(eye_camera, '_draw_minimal'):
                eye_camera._draw_minimal(points, lines, target)  # or call the full draw if you prefer
            else:
                # fallback to full scene draw (a bit heavier but consistent)
                # ... repeat the cylinder/line/target drawing here or extract to a helper function ...
                pass

            # Read back the retinal pixels for detection + light model
            raw = glReadPixels(PANEL_X, PANEL_Y, PANEL_W, PANEL_H,
                               GL_RGB, GL_UNSIGNED_BYTE)
            retina_px = np.frombuffer(raw, dtype=np.uint8).reshape(PANEL_H, PANEL_W, 3)
            retina_px = np.flipud(retina_px)

            rx, ry, detected = eye_camera.detect_target(retina_px)
            light_model.update(retina_px, rx, ry, detected)

            retinal_result = {
                'detected': detected,
                'retinal_x': rx,
                'retinal_y': ry,
            }

            # Restore main matrices
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            # Optional: draw a nice border around the retinal panel
            glViewport(0, 0, WIDTH, HEIGHT)   # back to full
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, WIDTH, 0, HEIGHT, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            glDisable(GL_DEPTH_TEST)
            glColor4f(0.6, 0.8, 1.0, 0.9)
            glLineWidth(4.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(PANEL_X-2, PANEL_Y-2)
            glVertex2f(PANEL_X + PANEL_W + 2, PANEL_Y-2)
            glVertex2f(PANEL_X + PANEL_W + 2, PANEL_Y + PANEL_H + 2)
            glVertex2f(PANEL_X-2, PANEL_Y + PANEL_H + 2)
            glEnd()
            glEnable(GL_DEPTH_TEST)
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()

        # Restore full viewport for any remaining 2D overlays
        glViewport(0, 0, WIDTH, HEIGHT)

        pygame.display.flip()

        # ── Read full frame for JPEG output ──
        pixels = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        img = Image.frombytes('RGB', (WIDTH, HEIGHT), pixels)
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Apply the big eye HUD overlay only on top of the final image
        if eye_overlay:
            img = draw_eye_overlay(img, eye_overlay)

        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=82)
        return buf.getvalue(), retinal_result

    except Exception as e:
        print(f'Render error: {e}')
        import traceback; traceback.print_exc()
        blank = Image.new('RGB', (WIDTH, HEIGHT), 'black')
        buf   = io.BytesIO()
        blank.save(buf, format='JPEG')
        return buf.getvalue(), None


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
