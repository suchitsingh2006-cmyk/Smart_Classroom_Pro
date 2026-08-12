import os
import json
import base64
import io
import re
import asyncio
import concurrent.futures
from PIL import Image
from dotenv import load_dotenv
from groq import Groq
import edge_tts

# Load environment variables
load_dotenv()

# Model Definitions
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-instruct"
WHISPER_MODEL = "whisper-large-v3-turbo"

# Voice Mapping for Edge-TTS
VOICE_MAPPING = {
    "Hindi": "hi-IN-SwaraNeural",
    "hi": "hi-IN-SwaraNeural",
    "Bangla": "bn-BD-NabanitaNeural",
    "bn": "bn-BD-NabanitaNeural",
    "Arabic": "ar-SA-ZariyahNeural",
    "ar": "ar-SA-ZariyahNeural",
    "English": "en-US-AvaNeural",
    "en": "en-US-AvaNeural",
    "Spanish": "es-ES-ElviraNeural",
    "es": "es-ES-ElviraNeural",
    "French": "fr-FR-DeniseNeural",
    "fr": "fr-FR-DeniseNeural",
    "German": "de-DE-KatjaNeural",
    "de": "de-DE-KatjaNeural"
}


def get_groq_client(api_key: str = None) -> Groq:
    """Initialize and return the Groq client using provided key or environment variable."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ API Key is missing. Please set GROQ_API_KEY in .env or enter it in the app sidebar.")
    return Groq(api_key=key)


# --- SAFE ASYNC RUNNER FOR STREAMLIT ---
def run_async_safe(coro):
    """Safely execute an async coroutine inside Streamlit without event loop conflicts."""
    def _run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_new_loop)
            return future.result()
    else:
        return _run_in_new_loop()


# --- EDGE-TTS AUDIO GENERATION ---
async def _async_generate_audio(text: str, voice: str) -> bytes:
    """Internal async helper to stream Edge-TTS audio to bytes."""
    communicate = edge_tts.Communicate(text, voice)
    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    return audio_stream.getvalue()


def generate_audio(text: str, language: str = "English") -> bytes:
    """Generate audio bytes for given text in specified language using Edge-TTS."""
    if not text or not text.strip():
        return b""
    voice = VOICE_MAPPING.get(language, VOICE_MAPPING.get(language.lower(), "en-US-AvaNeural"))
    cleaned_text = text[:1500].strip()
    try:
        return run_async_safe(_async_generate_audio(cleaned_text, voice))
    except Exception as e:
        print(f"Error generating Edge-TTS audio: {e}")
        return b""


# --- WHISPER AUDIO TRANSCRIPTION ---
def transcribe_audio_file(client: Groq, audio_file) -> str:
    """Transcribe audio file or recording using Groq Whisper API."""
    try:
        if hasattr(audio_file, "read"):
            file_bytes = audio_file.read()
            filename = getattr(audio_file, "name", "recorded_audio.wav")
        elif isinstance(audio_file, bytes):
            file_bytes = audio_file
            filename = "recorded_audio.wav"
        else:
            return ""

        transcription = client.audio.transcriptions.create(
            file=(filename, file_bytes),
            model=WHISPER_MODEL,
            response_format="text"
        )
        return str(transcription).strip()
    except Exception as e:
        print(f"Audio transcription error: {e}")
        return ""


# --- LATEX FORMULA & GLOSSARY LOCK ENGINE ---
def protect_latex_and_glossary(text: str) -> tuple[str, dict]:
    """
    Protects mathematical formulas ($...$, $$...$$, \\begin{equation}...\\end{equation})
    and glossary terms by replacing them with placeholder tokens during AI translation.
    """
    placeholders = {}
    counter = 0

    math_patterns = [
        r'\$\$.*?\$\$',
        r'\$.*?\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
        r'\\begin\{equation\}.*?\\end\{equation\}'
    ]

    combined_pattern = '|'.join(math_patterns)

    def replace_match(match):
        nonlocal counter
        token = f"__MATH_TOKEN_{counter}__"
        placeholders[token] = match.group(0)
        counter += 1
        return token

    protected_text = re.sub(combined_pattern, replace_match, text, flags=re.DOTALL)
    return protected_text, placeholders


def restore_latex_and_glossary(text: str, placeholders: dict) -> str:
    """Restores protected math formulas and glossary terms back into translated text."""
    restored_text = text
    for token, original in placeholders.items():
        restored_text = restored_text.replace(token, original)
    return restored_text


# --- DYNAMIC TEXT TRANSLATOR WITH LATEX LOCK ---
def translate_text(client: Groq, text: str, target_language: str) -> str:
    """Translate educational text notes or captions from English into target language with LaTeX protection."""
    if not text or not text.strip() or target_language == "English":
        return text

    protected_text, placeholders = protect_latex_and_glossary(text)

    system_prompt = f"""You are an expert educational translator.
Translate the following English text into {target_language}.

CRITICAL RULE:
Do NOT modify, translate, or remove any placeholder tokens like __MATH_TOKEN_0__, __MATH_TOKEN_1__, etc.
Keep all formatting and structural markdown tags intact.
Output only the translation."""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": protected_text}
        ],
        temperature=0.2
    )

    translated_raw = response.choices[0].message.content.strip()
    final_translated = restore_latex_and_glossary(translated_raw, placeholders)
    return final_translated


# --- LECTURE TRANSCRIPT & STRUCTURED NOTES ENGINE ---
def process_lecture_transcript(client: Groq, transcript: str, target_language: str = "English") -> dict:
    """Process lecture transcript into chapters, structured notes, markdown/LaTeX, and summary."""
    system_prompt = f"""You are an expert AI Smart Classroom Assistant.
Analyze the provided lecture transcript and generate structured dynamic class notes.

Return a JSON object with EXACTLY this structure:
{{
    "summary": "Concise high-level summary of the lecture in {target_language}",
    "structured_notes": "Detailed structured markdown class notes with headers (#, ##), bullet points, and key takeaways in {target_language}. Use LaTeX ($...$ or $$...$$) for formulas.",
    "raw_transcript": "The full original lecture transcript",
    "chapters": [
        {{
            "timestamp": "00:00",
            "title": "Short Chapter Title",
            "description": "Key concept discussed in this section"
        }}
    ],
    "key_concepts": ["Concept 1", "Concept 2", "Concept 3"],
    "glossary": [
        {{"term": "Term Name", "definition": "Definition in {target_language}"}}
    ]
}}

Important Instructions:
1. All summary, notes, chapters, and glossary definitions MUST be in {target_language}.
2. Ensure mathematical formulas are in valid LaTeX notation.
3. Output raw valid JSON only without markdown codeblocks."""

    user_prompt = f"Target Language: {target_language}\n\nLecture Transcript:\n{transcript}"

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content.strip()
    data = json.loads(content)
    data["raw_transcript"] = transcript
    return data


# --- SMART BOARD VISION OCR & MULTIMODAL ANALYSIS ---
def encode_image_to_base64(image_pil: Image.Image) -> str:
    """Convert PIL Image to base64 data string."""
    buffered = io.BytesIO()
    if image_pil.mode in ("RGBA", "P"):
        image_pil = image_pil.convert("RGB")
    image_pil.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_smartboard_image(client: Groq, image_pil: Image.Image, target_language: str = "English") -> dict:
    """Extract text, LaTeX formulas, diagrams, and flowcharts from a whiteboard/notebook image."""
    base64_image = encode_image_to_base64(image_pil)
    data_url = f"data:image/jpeg;base64,{base64_image}"

    system_prompt = f"""You are an advanced AI Smart Board & Notebook Multimodal Vision Scanner.
Analyze the uploaded whiteboard or notebook image and output raw valid JSON only.

Return a JSON object with EXACTLY this structure:
{{
    "extracted_text": "Complete raw extracted text from the image",
    "latex_formulas": [
        "\\\\mathbf{{F}} = m \\\\mathbf{{a}}",
        "E = mc^2"
    ],
    "translated_notes": "Structured notes extracted from board in {target_language}",
    "diagram_analysis": "Step-by-step breakdown of diagrams, flowcharts, graphs, or visual components on the board in {target_language}",
    "explanation": "Clear educational explanation of the concepts shown on the board in {target_language}"
}}

Important Instructions:
1. Ensure all mathematical equations or handwritten formulas are accurately transcribed into valid LaTeX syntax.
2. All notes, diagram explanations, and breakdowns MUST be in {target_language}.
3. Output raw valid JSON only without markdown codeblocks."""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Perform OCR, extract LaTeX math formulas, analyze flowcharts/diagrams, and explain in {target_language}."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    }
                ]
            }
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)


# --- ANONYMOUS DOUBT BRIDGE TRANSLATION ---
def translate_student_doubt_to_english(client: Groq, doubt_text: str, source_language: str) -> dict:
    """Translate student doubt from native language into English for the teacher."""
    system_prompt = """You are a bilingual classroom translation assistant.
Translate the student's doubt from their native language into clear, polite English for the teacher.

Return JSON with structure:
{
    "translated_english": "English translation of the doubt",
    "topic_category": "Physics/Math/Computer Science/General"
}
Output raw valid JSON only."""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Source Language: {source_language}\nStudent Doubt: {doubt_text}"}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content.strip())


def translate_teacher_reply_to_student(client: Groq, teacher_reply_en: str, target_language: str) -> dict:
    """Translate teacher's English reply back into student's native language and generate localized audio."""
    translated_text = translate_text(client, teacher_reply_en, target_language)
    audio_bytes = generate_audio(translated_text, target_language)
    return {
        "translated_text": translated_text,
        "audio_bytes": audio_bytes
    }


# --- AUTOMATED QUIZ GENERATOR ---
def generate_quiz(client: Groq, context_text: str, num_questions: int = 3) -> list:
    """Generate multiple-choice quiz questions based strictly on lecture context."""
    system_prompt = f"""You are an educational assessment expert.
Generate {num_questions} Multiple-Choice Questions (MCQs) based strictly on the provided lecture context.

Return JSON with format:
{{
    "quiz": [
        {{
            "id": 1,
            "question": "Question text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Exact text of correct option",
            "explanation": "Why this answer is correct"
        }}
    ]
}}
Output raw valid JSON only."""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Lecture Context:\n{context_text}"}
        ],
        temperature=0.4,
        response_format={"type": "json_object"}
    )
    res = json.loads(response.choices[0].message.content.strip())
    return res.get("quiz", [])


# --- RAG INTERACTIVE LECTURE AI TUTOR ---
def ask_context_tutor(client: Groq, query: str, context_text: str, chat_history: list = None, preferred_language: str = "English") -> str:
    """RAG-powered AI chatbot grounded strictly in the current session's live transcript and board notes."""
    if not context_text or not context_text.strip():
        return "No live lecture content or board notes have been processed yet today. Please ask your teacher to process a lecture first!"

    system_prompt = f"""You are the 24/7 AI Smart Classroom Tutor.
STRICT INSTRUCTION: You must answer the student's question ONLY using the provided live lecture context below.
If the answer is NOT mentioned or covered in the context, politely state: "This topic was not covered in today's class lecture context."

Respond in the student's preferred language ({preferred_language}).
Keep explanations concise, clear, and encouraging. Format formulas in LaTeX ($...$ or $$...$$).

LIVE LECTURE CONTEXT:
{context_text}"""

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
