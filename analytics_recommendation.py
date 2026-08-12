import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class AnalyticsRecommendationEngine:
    """
    Generates Teacher Engagement Heatmaps and AI-driven personalized study plans for students.
    """

    @staticmethod
    def generate_personalized_study_plan(
        client_groq,
        quiz_errors: list,
        low_attention_timestamps: list,
        student_doubts: list,
        student_lang: str = "English"
    ) -> dict:
        """
        Uses Groq AI to create an actionable, personalized study plan based on student performance.
        """
        from ai_engine import TEXT_MODEL

        system_prompt = f"""You are an expert AI Educational Counselor and Study Advisor.
Analyze the student's weakness data and construct a personalized study improvement plan in {student_lang}.

Return JSON with structure:
{{
    "overall_diagnostic": "Summary of current learning state and weak spots in {student_lang}",
    "recommended_topics": [
        {{
            "topic": "Topic Name",
            "priority": "HIGH / MEDIUM / LOW",
            "reason": "Why this topic needs review based on quiz errors or attention drops"
        }}
    ],
    "actionable_study_tips": [
        "Tip 1 in {student_lang}",
        "Tip 2 in {student_lang}"
    ],
    "recommended_review_timestamps": ["04:15", "09:30"]
}}
Output raw valid JSON only."""

        user_content = f"""
Student Weakness Input:
- Quiz Errors / Missed Questions: {json.dumps(quiz_errors)}
- Low Attentiveness Timestamps: {json.dumps(low_attention_timestamps)}
- Student Doubt Query History: {json.dumps(student_doubts)}
"""

        try:
            response = client_groq.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error generating study plan: {e}")
            return {
                "overall_diagnostic": "Review Newton's Laws and Kinetic Energy concepts from today's lecture.",
                "recommended_topics": [
                    {"topic": "Newton's Second Law (F = ma)", "priority": "HIGH", "reason": "Missed Quiz Q1"},
                    {"topic": "Kinetic Energy Equations", "priority": "MEDIUM", "reason": "Attention dropped at timestamp 04:15"}
                ],
                "actionable_study_tips": [
                    "Re-watch chapter timestamp 04:15 - 06:00.",
                    "Practice 5 practice problems on Force and Acceleration."
                ],
                "recommended_review_timestamps": ["04:15", "09:30"]
            }

    @staticmethod
    def create_teacher_engagement_heatmap(attentiveness_timeline: list) -> go.Figure:
        """
        Builds a Plotly interactive engagement heatmap chart for teachers showing attentiveness levels
        across lecture timestamps.
        attentiveness_timeline: list of dicts [{"timestamp": "00:00", "attentiveness_pct": 95, "student_count": 25}, ...]
        """
        if not attentiveness_timeline:
            # Generate default sample timeline data if none provided
            attentiveness_timeline = [
                {"timestamp": "00:00", "attentiveness_pct": 95, "drop_flag": False},
                {"timestamp": "05:00", "attentiveness_pct": 90, "drop_flag": False},
                {"timestamp": "10:00", "attentiveness_pct": 65, "drop_flag": True},   # Drop
                {"timestamp": "15:00", "attentiveness_pct": 85, "drop_flag": False},
                {"timestamp": "20:00", "attentiveness_pct": 58, "drop_flag": True},   # Drop
                {"timestamp": "25:00", "attentiveness_pct": 92, "drop_flag": False},
                {"timestamp": "30:00", "attentiveness_pct": 88, "drop_flag": False}
            ]

        df = pd.DataFrame(attentiveness_timeline)

        fig = px.area(
            df,
            x="timestamp",
            y="attentiveness_pct",
            title="📊 Real-Time Classroom Attentiveness & Engagement Timeline",
            labels={"timestamp": "Lecture Timestamp", "attentiveness_pct": "Attentiveness Index (%)"},
            color_discrete_sequence=["#818cf8"]
        )

        fig.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0)",
            plot_bgcolor="rgba(30, 41, 59, 0.5)",
            font=dict(color="#f8fafc", family="Inter, sans-serif"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 100]),
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Highlight drop-off points in red
        drops = df[df["attentiveness_pct"] < 70]
        if not drops.empty:
            fig.add_trace(go.Scatter(
                x=drops["timestamp"],
                y=drops["attentiveness_pct"],
                mode="markers+text",
                name="Attention Drop (<70%)",
                marker=dict(color="#ef4444", size=12, symbol="x"),
                text=[f"⚠️ Drop: {val}%" for val in drops["attentiveness_pct"]],
                textposition="top center"
            ))

        return fig
