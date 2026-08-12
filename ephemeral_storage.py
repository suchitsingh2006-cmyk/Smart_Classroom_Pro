import time
from datetime import datetime, timedelta


class EphemeralStorageManager:
    """
    Manages 24-hour auto-purge for raw media files, audio recordings, and high-frequency stream logs
    while preserving aggregated metrics and final structured notes.
    """

    EXPIRATION_HOURS = 24

    @staticmethod
    def is_item_expired(created_timestamp_str: str, expiration_hours: int = 24) -> bool:
        """Checks whether a given timestamp string (%Y-%m-%d %H:%M:%S or %H:%M:%S) exceeds the expiration window."""
        try:
            if " " in created_timestamp_str:
                created_dt = datetime.strptime(created_timestamp_str, "%Y-%m-%d %H:%M:%S")
            else:
                today_date = datetime.now().strftime("%Y-%m-%d")
                created_dt = datetime.strptime(f"{today_date} {created_timestamp_str}", "%Y-%m-%d %H:%M:%S")

            cutoff = datetime.now() - timedelta(hours=expiration_hours)
            return created_dt < cutoff
        except Exception:
            return False

    @classmethod
    def auto_purge_expired_session_data(cls, session_state: dict) -> dict:
        """
        Scans session state for media gallery, audio bytes, and raw camera frames,
        purging items older than 24 hours while retaining notes, summaries, and performance metrics.
        """
        purged_stats = {
            "purged_media_count": 0,
            "purged_bytes_estimate": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

        # 1. Purge Media Gallery
        if "media_gallery" in session_state and session_state["media_gallery"]:
            original_len = len(session_state["media_gallery"])
            filtered_gallery = []
            for item in session_state["media_gallery"]:
                item_time = item.get("time", datetime.now().strftime("%H:%M:%S"))
                if cls.is_item_expired(item_time, cls.EXPIRATION_HOURS):
                    # Purge raw content bytes
                    if "content" in item:
                        purged_stats["purged_bytes_estimate"] += len(str(item["content"]))
                    purged_stats["purged_media_count"] += 1
                else:
                    filtered_gallery.append(item)
            session_state["media_gallery"] = filtered_gallery

        # 2. Purge Raw Audio Bytes from Doubt Answers > 24 hours
        if "doubts" in session_state and session_state["doubts"]:
            for d in session_state["doubts"]:
                doubt_time = d.get("timestamp", datetime.now().strftime("%H:%M:%S"))
                if cls.is_item_expired(doubt_time, cls.EXPIRATION_HOURS):
                    if d.get("reply_audio"):
                        d["reply_audio"] = None  # Purge raw audio, keep text answer
                        purged_stats["purged_media_count"] += 1

        session_state["last_auto_purge_stats"] = purged_stats
        return purged_stats
