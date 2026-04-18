"""
eye_camera.py
─────────────
Renders a low-resolution "retinal" viewport from the walker's head position.
Instead of computing angles to a known target, this captures whatever is
actually in front of the head — exactly like a real eye.

The render is done into an off-screen FBO (or by saving/restoring viewport
state) and returns an H×W×3 numpy array of RGB pixels.
"""

import math
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    _GL_AVAILABLE = True
except ImportError:
    _GL_AVAILABLE = False

# Retinal resolution — small enough to be fast, big enough for colour detection
RETINA_W = 128
RETINA_H = 72

# Target colour to detect (matches draw_target in Render.py: (0.2, 1.0, 0.2))
TARGET_RGB = (0.2, 1.0, 0.2)          # float, 0-1
TARGET_R8  = int(TARGET_RGB[0] * 255)  # 51
TARGET_G8  = int(TARGET_RGB[1] * 255)  # 255
TARGET_B8  = int(TARGET_RGB[2] * 255)  # 51


class EyeCamera:
    """
    Captures a small retinal image from the walker's head perspective.

    Usage
    -----
    cam = EyeCamera()
    pixels = cam.render_retina(hx, hy, hz, gaze_yaw, gaze_pitch,
                               fov_degrees, points, lines, target)
    rx, ry, detected = cam.detect_target(pixels)
    lux = cam.compute_lux(pixels)
    """

    def __init__(self, retina_w: int = RETINA_W, retina_h: int = RETINA_H):
        self.W = retina_w
        self.H = retina_h
        self._fbo        = None
        self._rbo_color  = None
        self._rbo_depth  = None
        self._fbo_ready  = False

    # ── FBO setup ────────────────────────────────────────────────────

    def _init_fbo(self):
        """Create an off-screen FBO at retinal resolution."""
        if not _GL_AVAILABLE:
            return
        try:
            self._fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

            self._rbo_color = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self._rbo_color)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, self.W, self.H)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_RENDERBUFFER, self._rbo_color)

            self._rbo_depth = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self._rbo_depth)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, self.W, self.H)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, self._rbo_depth)

            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            self._fbo_ready = (status == GL_FRAMEBUFFER_COMPLETE)
            if not self._fbo_ready:
                print(f"⚠️  EyeCamera FBO incomplete: {status}")
        except Exception as e:
            print(f"⚠️  EyeCamera FBO init failed: {e}")
            self._fbo_ready = False

    # ── Retinal render ────────────────────────────────────────────────

    def render_retina(self,
                      hx: float, hy: float, hz: float,
                      gaze_yaw: float, gaze_pitch: float,
                      fov_degrees: float,
                      points: list, lines: list, target: dict | None,
                      draw_fn=None) -> np.ndarray:
        """
        Render a small viewport from head position (hx,hy,hz) looking along
        (gaze_yaw, gaze_pitch).

        Parameters
        ----------
        hx,hy,hz      : head world position
        gaze_yaw      : horizontal gaze angle (radians)
        gaze_pitch    : vertical   gaze angle (radians)
        fov_degrees   : horizontal field of view in degrees
        points,lines  : scene geometry (same format as Render.py)
        target        : {'x','y','z'} dict or None
        draw_fn       : optional callable(points, lines, target) that draws
                        the scene; defaults to internal minimal draw

        Returns
        -------
        np.ndarray  shape (H, W, 3)  uint8 RGB
        """
        if not _GL_AVAILABLE:
            return np.zeros((self.H, self.W, 3), dtype=np.uint8)

        if not self._fbo_ready:
            self._init_fbo()
        if not self._fbo_ready:
            # Fallback: render into the current back-buffer at a corner
            return self._render_viewport_fallback(
                hx, hy, hz, gaze_yaw, gaze_pitch, fov_degrees,
                points, lines, target, draw_fn)

        # ── Bind FBO ──────────────────────────────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.W, self.H)

        pixels = self._do_render(hx, hy, hz, gaze_yaw, gaze_pitch,
                                 fov_degrees, points, lines, target, draw_fn)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return pixels

    def _render_viewport_fallback(self,
                                  hx, hy, hz, gaze_yaw, gaze_pitch,
                                  fov_degrees, points, lines, target, draw_fn):
        """
        If FBO unavailable, render into a small corner of the main buffer
        and read it back.  The main render loop will overdraw this area.
        """
        from Main import WIDTH, HEIGHT  # import lazily to avoid circular
        glViewport(0, 0, self.W, self.H)
        pixels = self._do_render(hx, hy, hz, gaze_yaw, gaze_pitch,
                                 fov_degrees, points, lines, target, draw_fn)
        glViewport(0, 0, WIDTH, HEIGHT)
        return pixels

    def _do_render(self, hx, hy, hz, gaze_yaw, gaze_pitch,
                   fov_degrees, points, lines, target, draw_fn):
        """Set up perspective from head POV and draw the scene."""
        glClearColor(0.05, 0.05, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        aspect = self.W / self.H
        gluPerspective(fov_degrees, aspect, 0.05, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        fx = math.sin(gaze_yaw) * math.cos(gaze_pitch)
        fy = math.sin(gaze_pitch)
        fz = -math.cos(gaze_yaw) * math.cos(gaze_pitch)

        gluLookAt(hx, hy, hz,
                  hx + fx, hy + fy, hz + fz,
                  0.0, 1.0, 0.0)

        glEnable(GL_DEPTH_TEST)

        if draw_fn is not None:
            draw_fn(points, lines, target)
        else:
            self._draw_minimal(points, lines, target)

        # Read pixels
        raw = glReadPixels(0, 0, self.W, self.H, GL_RGB, GL_UNSIGNED_BYTE)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.H, self.W, 3)
        arr = np.flipud(arr)   # OpenGL origin is bottom-left

        glMatrixMode(GL_MODELVIEW);  glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        return arr

    # ── Minimal scene draw (used when no draw_fn supplied) ────────────

    def _draw_minimal(self, points, lines, target):
        """Draw just enough geometry for target detection."""
        from Render import draw_cylinder, draw_target, safe_color

        # Scene skeleton
        if lines and points:
            for line in lines:
                try:
                    p1 = points[line['from']]
                    p2 = points[line['to']]
                    draw_cylinder(p1, p2,
                                  radius=line.get('radius', 0.06),
                                  color=line.get('color', (1.0, 1.0, 1.0)))
                except Exception:
                    pass

        # Ground plane (dark, so the green target pops)
        glColor3f(0.15, 0.15, 0.15)
        glBegin(GL_QUADS)
        glVertex3f(-200, 0, -200)
        glVertex3f( 200, 0, -200)
        glVertex3f( 200, 0,  200)
        glVertex3f(-200, 0,  200)
        glEnd()

        # Target cylinder — MUST use the same colour as Render.draw_target
        if target:
            draw_target(target['x'], target['y'], target['z'])

    # ── Detection ─────────────────────────────────────────────────────

    @staticmethod
    def detect_target(pixels: np.ndarray,
                      min_pixels: int = 15) -> tuple[float, float, bool]:
        """
        Detect the green target cylinder in a retinal image.

        Parameters
        ----------
        pixels    : H×W×3 uint8 numpy array
        min_pixels: minimum green pixel count to confirm detection

        Returns
        -------
        (retinal_x, retinal_y, detected)
          retinal_x/y in [-1, 1] — centre of mass of detected pixels
          -1=left/top, +1=right/bottom
        """
        r = pixels[:, :, 0].astype(np.int16)
        g = pixels[:, :, 1].astype(np.int16)
        b = pixels[:, :, 2].astype(np.int16)

        # Match target colour: green dominant, r and b low
        green_mask = (
                (g > 120) &
                (r < 100) &
                (b < 100) &
                ((g - r) > 60) &
                ((g - b) > 60)
        )

        count = int(green_mask.sum())
        if count < min_pixels:
            return 0.0, 0.0, False

        ys, xs = np.where(green_mask)
        cx = float(xs.mean()) / pixels.shape[1] * 2.0 - 1.0   # -1=left  +1=right
        cy = float(ys.mean()) / pixels.shape[0] * 2.0 - 1.0   # -1=top   +1=bottom
        return cx, cy, True

    # ── Luminance ─────────────────────────────────────────────────────

    @staticmethod
    def compute_lux(pixels: np.ndarray) -> dict:
        """
        Compute scene luminance from retinal pixels.

        Returns
        -------
        dict with keys:
          ambient_lux   : mean scene brightness 0-1
          pupil_dilation: inverse of lux (0=contracted, 1=fully dilated)
          is_bright     : bool — scene bright enough for full colour vision
        """
        if pixels.size == 0:
            return {'ambient_lux': 0.5, 'pupil_dilation': 0.5, 'is_bright': True}

        # ITU-R BT.709 luminance
        lum = (0.2126 * pixels[:, :, 0].astype(np.float32) +
               0.7152 * pixels[:, :, 1].astype(np.float32) +
               0.0722 * pixels[:, :, 2].astype(np.float32)) / 255.0

        ambient = float(lum.mean())
        # Pupil contracts in bright light, dilates in dark
        dilation = 1.0 - float(np.clip(ambient * 1.4, 0.0, 1.0))

        return {
            'ambient_lux':    ambient,
            'pupil_dilation': dilation,
            'is_bright':      ambient > 0.15,
        }

    # ── Cleanup ───────────────────────────────────────────────────────

    def destroy(self):
        if not _GL_AVAILABLE:
            return
        try:
            if self._fbo:
                glDeleteFramebuffers(1, [self._fbo])
            if self._rbo_color:
                glDeleteRenderbuffers(1, [self._rbo_color])
            if self._rbo_depth:
                glDeleteRenderbuffers(1, [self._rbo_depth])
        except Exception:
            pass
        self._fbo_ready = False
