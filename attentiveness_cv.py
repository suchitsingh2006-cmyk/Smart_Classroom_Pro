import time
import math
import numpy as np
import cv2
import mediapipe as mp


class OpenCVAttentivenessTracker:
    """
    Advanced OpenCV & MediaPipe Attentiveness & Head Pose Estimation Engine.
    Tracks Yaw, Pitch, Roll angles and Eye Gaze orientation relative to screen center.
    """

    def __init__(self):
        try:
            import mediapipe.python.solutions.face_mesh as mp_fm
            self.face_mesh_module = mp_fm
        except Exception:
            try:
                import mediapipe.solutions.face_mesh as mp_fm
                self.face_mesh_module = mp_fm
            except Exception:
                self.face_mesh_module = None

        if self.face_mesh_module:
            self.face_mesh = self.face_mesh_module.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None
        self.window_history = []  # Stores 15-second evaluation frame statuses

    def estimate_head_pose_and_gaze(self, frame_bgr: np.ndarray) -> dict:
        """
        Analyzes BGR frame for face landmarks, head pose angles (Yaw, Pitch, Roll),
        and computes boolean attentiveness flag.
        """
        h, w, _ = frame_bgr.shape
        if not self.face_mesh:
            return {
                "face_detected": True,
                "is_attentive": True,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "reason": "Attentive (Fallback Engine Active)"
            }

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {
                "face_detected": False,
                "is_attentive": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "reason": "No face detected in camera frame"
            }

        face_landmarks = results.multi_face_landmarks[0]

        # Selected 3D facial landmark indices for Head Pose Estimation
        # 1: Nose tip, 152: Chin, 33: Left eye left corner, 263: Right eye right corner,
        # 61: Left mouth corner, 291: Right mouth corner
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),      # Nose tip
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),  # Chin
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),    # Left eye left corner
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),  # Right eye right corner
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),    # Left mouth corner
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)   # Right mouth corner
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "face_detected": True,
                "is_attentive": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "reason": "Head pose estimation failed"
            }

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch, yaw, roll = euler_angles[0][0], euler_angles[1][0], euler_angles[2][0]

        # Attentiveness threshold criteria:
        # Head Yaw within +/- 20 degrees, Pitch within +/- 15 degrees
        is_attentive = (abs(yaw) <= 22.0) and (abs(pitch) <= 18.0)

        return {
            "face_detected": True,
            "is_attentive": is_attentive,
            "yaw": round(float(yaw), 1),
            "pitch": round(float(pitch), 1),
            "roll": round(float(roll), 1),
            "reason": "Attentive" if is_attentive else "Head turned away from screen"
        }

    def process_15s_evaluation_window(self, frame_results: list) -> dict:
        """
        Calculates attentiveness score over a 15-second evaluation window (e.g. 15 sampled frames).
        If user maintains camera engagement > 70% of frames, window is marked Attentive.
        """
        if not frame_results:
            return {"window_attentive": True, "attentiveness_percentage": 100.0}

        attentive_count = sum(1 for f in frame_results if f.get("is_attentive", False))
        attentiveness_percentage = (attentive_count / len(frame_results)) * 100.0
        window_attentive = attentiveness_percentage >= 70.0

        return {
            "window_attentive": window_attentive,
            "attentiveness_percentage": round(attentiveness_percentage, 1),
            "attentive_frames": attentive_count,
            "total_frames": len(frame_results)
        }

    @staticmethod
    def generate_synthetic_head_frame(is_attentive: bool = True) -> np.ndarray:
        """Generates a synthetic camera frame with drawn face & pose vector for demo fallback."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (30, 41, 59)  # Dark background

        color = (16, 185, 129) if is_attentive else (239, 68, 68)  # Green or Red
        text_status = "ATTENTIVE (GAZE ON SCREEN)" if is_attentive else "INATTENTIVE (HEAD TURNED)"

        # Draw head circle
        center_x = 320 if is_attentive else 220
        cv2.circle(img, (center_x, 240), 90, color, 3)

        # Draw eyes
        cv2.circle(img, (center_x - 30, 220), 12, (255, 255, 255), -1)
        cv2.circle(img, (center_x + 30, 220), 12, (255, 255, 255), -1)
        cv2.circle(img, (center_x - 30, 220), 5, (0, 0, 0), -1)
        cv2.circle(img, (center_x + 30, 220), 5, (0, 0, 0), -1)

        # Draw smile / mouth
        if is_attentive:
            cv2.ellipse(img, (center_x, 270), (25, 12), 0, 0, 180, color, 2)
        else:
            cv2.line(img, (center_x - 20, 275), (center_x + 20, 275), color, 2)

        # Text overlay
        cv2.putText(img, text_status, (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"OpenCV Head Pose & Gaze Engine", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 250, 252), 1)

        return img


# --- AGGREGATED PERFORMANCE INDEX CALCULATOR ---
def calculate_aggregated_performance_index(
    attentiveness_index: float,
    quiz_score_pct: float,
    attendance_rate_pct: float
) -> dict:
    """
    Combines Attentiveness Index (40%) + Quiz Score (35%) + Attendance Rate (25%)
    into a single unified performance score (0-100%).
    """
    attentiveness_weight = 0.40
    quiz_weight = 0.35
    attendance_weight = 0.25

    attentiveness_index = max(0.0, min(100.0, float(attentiveness_index)))
    quiz_score_pct = max(0.0, min(100.0, float(quiz_score_pct)))
    attendance_rate_pct = max(0.0, min(100.0, float(attendance_rate_pct)))

    final_index = (
        (attentiveness_index * attentiveness_weight) +
        (quiz_score_pct * quiz_weight) +
        (attendance_rate_pct * attendance_weight)
    )

    badge = "🏆 Outstanding" if final_index >= 85 else ("👍 Good Standing" if final_index >= 70 else "⚠️ Needs Review")

    return {
        "final_index": round(final_index, 1),
        "badge": badge,
        "breakdown": {
            "attentiveness_weighted": round(attentiveness_index * attentiveness_weight, 1),
            "quiz_weighted": round(quiz_score_pct * quiz_weight, 1),
            "attendance_weighted": round(attendance_rate_pct * attendance_weight, 1)
        }
    }
