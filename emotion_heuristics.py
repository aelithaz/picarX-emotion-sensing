from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp


@dataclass
class FaceMetrics:
    smile: float
    brow_furrow: float
    mouth_open: float


class EmotionHeuristics:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # static_image_mode=False -> tracking, for speed
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _dist(self, a, b) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _lm(self, landmarks, idx, w, h):
        p = landmarks[idx]
        return (p.x * w, p.y * h)

    def estimate_emotion(
        self,
        frame_bgr,
    ) -> Tuple[str, Optional[FaceMetrics], Optional[Tuple[int, int, int, int]]]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return "no_face", None, None

        lms = res.multi_face_landmarks[0].landmark
        L = lambda i: self._lm(lms, i, w, h)

        try:
            # Key landmarks
            mouth_left = L(61)
            mouth_right = L(291)
            nose_tip = L(1)
            upper_lip = L(13)
            lower_lip = L(14)  # approximate lower lip center

            # Eye widths (for brow normalization)
            left_eye_w = self._dist(L(33), L(133))
            right_eye_w = self._dist(L(362), L(263))
            eye_w = max(1.0, (left_eye_w + right_eye_w) * 0.5)

            # Brow positions
            brow_l = L(70)
            brow_r = L(300)
            eye_line_l = ((L(33)[0] + L(133)[0]) / 2, (L(33)[1] + L(133)[1]) / 2)
            eye_line_r = ((L(362)[0] + L(263)[0]) / 2, (L(362)[1] + L(263)[1]) / 2)
            brow_eye_l = self._dist(brow_l, eye_line_l)
            brow_eye_r = self._dist(brow_r, eye_line_r)
            brow_eye = (brow_eye_l + brow_eye_r) * 0.5

            # Mouth + nose distances
            mouth_w = self._dist(mouth_left, mouth_right)
            nose_lip = self._dist(nose_tip, upper_lip)
            mouth_gap = self._dist(upper_lip, lower_lip)

            # Normalized metrics
            smile_score = mouth_w / max(1.0, nose_lip)
            brow_furrow = brow_eye / eye_w
            mouth_open = mouth_gap / max(1.0, nose_lip)

            metrics = FaceMetrics(
                smile=smile_score,
                brow_furrow=brow_furrow,
                mouth_open=mouth_open,
            )

            # Heuristic labeling

            if metrics.smile > 2.6 and metrics.mouth_open < 1.3:
                label = "happy"
            elif metrics.mouth_open >= 1.4 and metrics.smile > 2.0:
                label = "angry"
            elif metrics.smile < 2.0:
                label = "sad"
            else:
                # fallback
                label = "sad"

            # simple face bbox (use eye & mouth spread)
            x_coords = [
                mouth_left[0],
                mouth_right[0],
                L(33)[0],
                L(133)[0],
                L(362)[0],
                L(263)[0],
            ]
            y_coords = [
                mouth_left[1],
                mouth_right[1],
                L(33)[1],
                L(133)[1],
                L(362)[1],
                L(263)[1],
            ]
            x1, y1, x2, y2 = (
                int(max(0, min(x_coords) - 20)),
                int(max(0, min(y_coords) - 40)),
                int(min(w, max(x_coords) + 20)),
                int(min(h, max(y_coords) + 40)),
            )
            bbox = (x1, y1, x2, y2)

            return label, metrics, bbox
        except Exception:
            return "sad", None, None
