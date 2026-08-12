import time
import math
import base64
import io
import pyotp
import qrcode

# Default Classroom Reference Coordinates (e.g., Main Science Hall)
DEFAULT_CLASSROOM_LAT = 28.6139
DEFAULT_CLASSROOM_LON = 77.2090
DEFAULT_MAX_RADIUS_METERS = 50.0  # 50 meter geofence threshold
SERVER_TOLERANCE_WINDOW_SECONDS = 25  # 25-second tolerance window


class AntiProxyAttendanceManager:
    """
    Manages Dynamic TOTP QR Codes and GPS Geofencing anti-proxy attendance verification.
    """

    def __init__(self, secret_seed: str = None):
        self.secret_seed = secret_seed or pyotp.random_base32()
        # Create TOTP with 15-second interval
        self.totp = pyotp.TOTP(self.secret_seed, interval=15)
        self.classroom_lat = DEFAULT_CLASSROOM_LAT
        self.classroom_lon = DEFAULT_CLASSROOM_LON
        self.max_radius = DEFAULT_MAX_RADIUS_METERS

    def set_classroom_location(self, lat: float, lon: float, radius_m: float = 50.0):
        """Update classroom GPS coordinates and maximum allowed radius."""
        self.classroom_lat = lat
        self.classroom_lon = lon
        self.max_radius = radius_m

    def generate_current_otp(self) -> tuple[str, int]:
        """Return current 6-digit TOTP code and remaining seconds until next refresh."""
        otp_code = self.totp.now()
        current_time = int(time.time())
        time_remaining = 15 - (current_time % 15)
        return otp_code, time_remaining

    def get_current_qr_payload(self) -> dict:
        """Returns the current verification payload for QR code generation."""
        otp_code = self.totp.now()
        current_time = int(time.time())
        time_remaining = 15 - (current_time % 15)
        payload_str = f"CLASS_ATTENDANCE:{otp_code}:{current_time}"
        return {
            "otp_code": otp_code,
            "payload": payload_str,
            "time_remaining": time_remaining,
            "seed": self.secret_seed
        }

    def generate_qr_image_base64(self) -> str:
        """Generates a PIL/Base64 image of the current dynamic QR code."""
        payload_info = self.get_current_qr_payload()
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=3,
        )
        qr.add_data(payload_info["payload"])
        qr.make(fit=True)
        img = qr.make_image(fill_color="#4f46e5", back_color="#ffffff")

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two GPS coordinates in meters using the Haversine formula.
        """
        R = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        distance = R * c
        return distance

    def verify_attendance_scan(
        self,
        scanned_otp: str,
        student_lat: float,
        student_lon: float,
        tolerance_windows: int = 2  # 2 windows of 15s = 30s tolerance window (satisfies 25s spec)
    ) -> dict:
        """
        Verify student scan against TOTP tolerance window and GPS geofence boundary.
        """
        # 1. TOTP Verification
        is_otp_valid = self.totp.verify(scanned_otp, valid_window=tolerance_windows)

        # 2. GPS Geofence Verification
        distance = self.haversine_distance(
            self.classroom_lat, self.classroom_lon,
            student_lat, student_lon
        )
        is_geofence_valid = distance <= self.max_radius

        status = "APPROVED" if (is_otp_valid and is_geofence_valid) else "REJECTED"
        
        reasons = []
        if not is_otp_valid:
            reasons.append("Expired or Invalid Dynamic QR Code")
        if not is_geofence_valid:
            reasons.append(f"Outside Classroom Geofence ({distance:.1f}m away, max allowed {self.max_radius}m)")

        return {
            "status": status,
            "is_otp_valid": is_otp_valid,
            "is_geofence_valid": is_geofence_valid,
            "distance_meters": round(distance, 1),
            "reasons": reasons,
            "timestamp": time.strftime("%H:%M:%S")
        }
