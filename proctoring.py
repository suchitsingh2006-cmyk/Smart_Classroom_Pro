from datetime import datetime
import streamlit.components.v1 as components


class ProctoringManager:
    """
    Manages client-side tab switch tracking, window focus loss alerts, and aspect ratio monitoring.
    """

    @staticmethod
    def get_proctoring_js_code(student_id: str, min_aspect_ratio: float = 1.1) -> str:
        """
        Generates JavaScript snippet to attach visibilitychange, window blur, and resize listeners.
        """
        return f"""
        <script>
        (function() {{
            const studentId = "{student_id}";
            const minAspect = {min_aspect_ratio};
            
            function logIncident(type, details) {{
                const payload = {{
                    student_id: studentId,
                    type: type,
                    details: details,
                    timestamp: new Date().toLocaleTimeString()
                }};
                console.warn("PROCTORING ALERT:", payload);
                
                // Store in window for Streamlit bridge or send via parent message
                if (window.parent) {{
                    window.parent.postMessage({{
                        type: "PROCTORING_ALERT",
                        data: payload
                    }}, "*");
                }}
            }}

            // 1. Tab switch & visibility change
            document.addEventListener("visibilitychange", function() {{
                if (document.hidden) {{
                    logIncident("TAB_SWITCH", "Student switched away from classroom tab");
                }} else {{
                    logIncident("TAB_FOCUS_RESTORED", "Student returned to classroom tab");
                }}
            }});

            // 2. Window blur & focus loss
            window.addEventListener("blur", function() {{
                logIncident("WINDOW_BLUR", "Student lost window focus (possible multi-window / second screen)");
            }});

            // 3. Aspect Ratio & Split Screen Monitor
            window.addEventListener("resize", function() {{
                const width = window.innerWidth;
                const height = window.innerHeight;
                const ratio = width / Math.max(height, 1);
                if (ratio < minAspect || width < 650) {{
                    logIncident("SPLIT_SCREEN_DETECTION", `Window size dropped to ${{width}}x${{height}} (Aspect Ratio: ${{ratio.toFixed(2)}})`);
                }}
            }});
        }})();
        </script>
        """

    @staticmethod
    def render_client_proctor_tracker(student_id: str):
        """Renders hidden JS tracker inside Streamlit component iframe."""
        js_code = ProctoringManager.get_proctoring_js_code(student_id)
        components.html(js_code, height=0, width=0)

    @staticmethod
    def log_incident(session_state_logs: list, student_id: str, incident_type: str, details: str):
        """Adds a proctoring incident log entry to the central session logs."""
        new_log = {
            "student_id": student_id,
            "type": incident_type,
            "details": details,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        session_state_logs.insert(0, new_log)
        return new_log
