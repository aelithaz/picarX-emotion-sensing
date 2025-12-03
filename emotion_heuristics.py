from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort

@dataclass
class FaceMetrics:
    smile: float
    brow_furrow: float


class EmotionHeuristics:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # static_image_mode=False -> tracking, for speed
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _dist(self, a, b) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _lm(self, landmarks, idx, w, h):
        p = landmarks[idx]
        return (p.x * w, p.y * h)

    def estimate_emotion(
        self,
        frame_bgr
    ) -> Tuple[str, Optional[FaceMetrics], Optional[Tuple[int, int, int, int]]]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return "no_face", None, None

        lms = res.multi_face_landmarks[0].landmark

        # Landmarks used (MediaPipe FaceMesh indices):
        # Mouth corners: 61 (left), 291 (right)
        # Nose tip: 1
        # Upper lip: 13
        # Brow inner:  brow-left  brow-right (near 55, 285), we use  brow mid 70, 300 approximations
        # Eyes: left eye 33 (outer), 133 (inner); right eye 362 (outer), 263 (inner)
        L = lambda i: self._lm(lms, i, w, h)

        try:
            mouth_left = L(61)
            mouth_right = L(291)
            nose_tip = L(1)
            upper_lip = L(13)

            left_eye_w = self._dist(L(33), L(133))
            right_eye_w = self._dist(L(362), L(263))
            eye_w = max(1.0, (left_eye_w + right_eye_w) * 0.5)

            brow_l = L(70)
            brow_r = L(300)
            # distance from brow to eye-line (approximate): compare to eye width
            eye_line_l = ((L(33)[0] + L(133)[0]) / 2, (L(33)[1] + L(133)[1]) / 2)
            eye_line_r = ((L(362)[0] + L(263)[0]) / 2, (L(362)[1] + L(263)[1]) / 2)
            brow_eye_l = self._dist(brow_l, eye_line_l)
            brow_eye_r = self._dist(brow_r, eye_line_r)
            brow_eye = (brow_eye_l + brow_eye_r) * 0.5

            mouth_w = self._dist(mouth_left, mouth_right)
            nose_lip = self._dist(nose_tip, upper_lip)

            smile_score = mouth_w / max(1.0, nose_lip)
            brow_furrow = (brow_eye / eye_w)

            metrics = FaceMetrics(smile=smile_score, brow_furrow=brow_furrow)

            # thresholds for facial recognition
            # higher smile_score → likely happy; smaller brow_furrow → brows lowered (angry)
            if metrics.smile > 2.0 and metrics.brow_furrow >= 0.9:
                label = "happy"
            elif metrics.brow_furrow < 0.75:
                label = "angry"
            else:
                label = "neutral"

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
            return "neutral", None, None

class EmotionFerPlus:
    """
    Emotion detector using FER+ ONNX model + Haar cascade face detection.

    Provides the same estimate_emotion(frame_bgr) API as EmotionHeuristics:
        returns (label, FaceMetrics|None, bbox|None)
    """

    CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    MODEL_PATH = "/home/pi/emotion-ferplus-8.onnx"

    EMOTION_LABELS = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    ]

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {self.CASCADE_PATH}")

        self.session = ort.InferenceSession(
            self.MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _classify_emotion_from_face(self, face_gray: np.ndarray):
        face_resized = cv2.resize(face_gray, (64, 64))
        face_resized = face_resized.astype("float32")
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: face_resized})
        scores = outputs[0][0]
        idx = int(np.argmax(scores))
        return self.EMOTION_LABELS[idx], float(scores[idx])

    def estimate_emotion(
        self,
        frame_bgr
    ) -> Tuple[str, Optional[FaceMetrics], Optional[Tuple[int, int, int, int]]]:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 0:
            return "no_face", None, None

        (x, y, fw, fh) = faces[0]
        roi_gray = gray[y:y + fh, x:x + fw]

        emotion, score = self._classify_emotion_from_face(roi_gray)

        if emotion == "happiness":
            label = "happy"
        elif emotion in ("anger", "disgust", "fear", "contempt"):
            label = "angry"
        elif emotion == "sadness":
            label = "neutral"  # or "sad" if you add it
        else:
            label = "neutral"

        bbox = (int(x), int(y), int(x + fw), int(y + fh))

        return label, None, bbox
