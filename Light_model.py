"""
light_model.py
──────────────
Computes lighting state from a retinal image captured by EyeCamera.

Drives:
  ambient_lux      — mean scene brightness (0 = pitch black, 1 = full sun)
  target_lux       — brightness in the region around the detected target
  pupil_dilation   — 0 = fully contracted (bright), 1 = fully dilated (dark)
  vignette_alpha   — how dark the HUD vignette edge should be
                       bright scene → dark vignette (high alpha)
                       dark scene   → light vignette (low alpha, uniform grey)

The vignette is applied as a CPU-side radial gradient in Render.py's
draw_eye_overlay(), using the values from get_render_data().
"""

import numpy as np


class LightModel:
    """
    Stateful luminance model with temporal smoothing (exponential moving
    average) so the pupil doesn't flicker every frame.
    """

    def __init__(self, smoothing: float = 0.12):
        """
        smoothing : EMA factor per frame (lower = more sluggish adaptation,
                    higher = instant).  0.12 ≈ 8-frame lag.
        """
        self._alpha = smoothing

        # Smoothed state
        self.ambient_lux    = 0.5
        self.target_lux     = 0.5
        self.pupil_dilation = 0.5  # 0=contracted … 1=dilated
        self.vignette_alpha = 80   # PIL alpha 0-255 for vignette edges

    # ── Update ───────────────────────────────────────────────────────

    def update(self,
               pixels: np.ndarray,
               ret_x: float = 0.0,
               ret_y: float = 0.0,
               detected: bool = False) -> None:
        """
        Analyse a retinal image and update smoothed lighting state.

        Parameters
        ----------
        pixels   : H×W×3 uint8 numpy array (from EyeCamera.render_retina)
        ret_x    : detected target retinal X in [-1, 1]
        ret_y    : detected target retinal Y in [-1, 1]
        detected : whether the target was found in this frame
        """
        if pixels is None or pixels.size == 0:
            return

        H, W = pixels.shape[:2]

        # ── Ambient luminance (whole frame) ───────────────────────────
        lum = self._luminance(pixels)
        raw_ambient = float(lum.mean())

        # ── Target-region luminance ───────────────────────────────────
        if detected:
            # Sample a 20% patch around the detected target centroid
            px = int((ret_x + 1.0) / 2.0 * W)
            py = int((ret_y + 1.0) / 2.0 * H)
            ph = max(4, H // 5)
            pw = max(4, W // 5)
            x0 = max(0, px - pw // 2);  x1 = min(W, px + pw // 2)
            y0 = max(0, py - ph // 2);  y1 = min(H, py + ph // 2)
            patch = lum[y0:y1, x0:x1]
            raw_target = float(patch.mean()) if patch.size > 0 else raw_ambient
        else:
            raw_target = raw_ambient

        # ── Smooth ───────────────────────────────────────────────────
        a = self._alpha
        self.ambient_lux = self.ambient_lux * (1 - a) + raw_ambient * a
        self.target_lux  = self.target_lux  * (1 - a) + raw_target  * a

        # ── Derived quantities ────────────────────────────────────────
        # Pupil contracts (→0) in bright light, dilates (→1) in darkness
        self.pupil_dilation = float(
            np.clip(1.0 - self.ambient_lux * 1.3, 0.0, 1.0)
        )

        # Vignette: strong dark edges when bright (realistic iris compression);
        # almost invisible when dark (no point darkening an already dark frame)
        self.vignette_alpha = int(
            np.clip(self.ambient_lux * 160, 20, 160)
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _luminance(pixels: np.ndarray) -> np.ndarray:
        """ITU-R BT.709 per-pixel luminance, shape (H,W), range [0,1]."""
        r = pixels[:, :, 0].astype(np.float32)
        g = pixels[:, :, 1].astype(np.float32)
        b = pixels[:, :, 2].astype(np.float32)
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

    # ── Snapshot ─────────────────────────────────────────────────────

    def get_render_data(self) -> dict:
        """Return lighting parameters for use in Render.py."""
        return {
            'ambient_lux':    round(self.ambient_lux,    3),
            'target_lux':     round(self.target_lux,     3),
            'pupil_dilation': round(self.pupil_dilation, 3),
            'vignette_alpha': self.vignette_alpha,
        }

    def reset(self):
        """Reset smoothing (e.g. when teleporting to a new scene)."""
        self.ambient_lux    = 0.5
        self.target_lux     = 0.5
        self.pupil_dilation = 0.5
        self.vignette_alpha = 80
