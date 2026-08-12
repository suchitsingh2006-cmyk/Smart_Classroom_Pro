import math
from datetime import datetime, timedelta
import json


class Flashcard:
    """Represents an individual flashcard with SM-2 spaced repetition metadata."""

    def __init__(self, card_id: int, question: str, answer: str, category: str = "General"):
        self.id = card_id
        self.question = question
        self.answer = answer
        self.category = category
        self.easiness_factor = 2.5
        self.repetition_count = 0
        self.interval_days = 1
        self.next_review_date = datetime.now().strftime("%Y-%m-%d")
        self.history = []

    def update_sm2(self, quality_score: int):
        """
        Updates card review interval using the SM-2 Spaced Repetition Algorithm.
        quality_score: 0 to 5 (0 = completely forgot, 5 = perfect recall).
        """
        quality_score = max(0, min(5, quality_score))

        # 1. Update Easiness Factor
        self.easiness_factor = self.easiness_factor + (
            0.1 - (5 - quality_score) * (0.08 + (5 - quality_score) * 0.02)
        )
        self.easiness_factor = max(1.3, self.easiness_factor)

        # 2. Update Repetition Count & Interval
        if quality_score < 3:
            self.repetition_count = 0
            self.interval_days = 1
        else:
            self.repetition_count += 1
            if self.repetition_count == 1:
                self.interval_days = 1
            elif self.repetition_count == 2:
                self.interval_days = 6
            else:
                self.interval_days = int(math.ceil(self.interval_days * self.easiness_factor))

        next_date = datetime.now() + timedelta(days=self.interval_days)
        self.next_review_date = next_date.strftime("%Y-%m-%d")

        self.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quality": quality_score,
            "interval_days": self.interval_days,
            "easiness_factor": round(self.easiness_factor, 2)
        })

    def to_dict(self) -> dict:
        """Convert flashcard object to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "easiness_factor": round(self.easiness_factor, 2),
            "repetition_count": self.repetition_count,
            "interval_days": self.interval_days,
            "next_review_date": self.next_review_date,
            "history": self.history
        }


class FlashcardDeck:
    """Manages a deck of flashcards with automated generation and review logic."""

    def __init__(self):
        self.cards = []

    def create_flashcards_from_notes(self, client_groq, note_text: str, category: str = "Class Notes") -> list:
        """Uses Groq AI to generate Q&A flashcards automatically from class notes or chapters."""
        from ai_engine import TEXT_MODEL

        system_prompt = """You are an AI Educational Flashcard Creator.
Extract key concepts, definitions, and equations from the provided notes and create Q&A flashcards.

Return JSON with structure:
{
    "flashcards": [
        {
            "question": "What is Newton's Second Law of Motion?",
            "answer": "Force equals mass times acceleration (F = m * a)."
        }
    ]
}
Output raw valid JSON only."""

        response = client_groq.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Notes/Chapter Text:\n{note_text[:2500]}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        created_cards = []
        for item in data.get("flashcards", []):
            card_id = len(self.cards) + 1
            card = Flashcard(card_id, item["question"], item["answer"], category=category)
            self.cards.append(card)
            created_cards.append(card)

        return created_cards

    def get_due_cards(self) -> list:
        """Return list of cards due for review today."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        return [c for c in self.cards if c.next_review_date <= today_str]

    def get_mastery_percentage(self) -> float:
        """Returns overall flashcard mastery percentage based on review count and ease factors."""
        if not self.cards:
            return 0.0
        mastered_count = sum(1 for c in self.cards if c.repetition_count >= 3)
        return round((mastered_count / len(self.cards)) * 100.0, 1)
