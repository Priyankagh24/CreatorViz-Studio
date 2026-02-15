import os
import re
import shutil
from pathlib import Path

import gtts
import numpy as np
import spacy
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import imageio

# ============================================================
# PATHS & FOLDERS
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent   # example: C:/DataViz-AI
UPLOADS_DIR = BASE_DIR / "uploads"
AUDIO_UPLOAD_DIR = UPLOADS_DIR / "audio_files"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD SMALL SPACY MODEL
# ============================================================

nlp_small = spacy.load("en_core_web_sm")


# ============================================================
# SMALL SUMMARIZER
# ============================================================

def small_summarizer(text: str, max_sentences: int = 2) -> str:
    """
    Lightweight summarizer (no transformers, no huggingface).
    Extracts top sentences by simple frequency scoring.
    """
    doc = nlp_small(text)
    sentences = list(doc.sents)

    if len(sentences) <= max_sentences:
        return text.strip()

    # Word frequency scoring
    freq = {}
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        w = token.text.lower()
        freq[w] = freq.get(w, 0) + 1

    if not freq:
        return text.strip()

    max_f = max(freq.values())
    freq = {w: v / max_f for w, v in freq.items()}

    scores = {}
    for sent in sentences:
        scores[sent] = sum(freq.get(t.text.lower(), 0) for t in sent)

    best = sorted(scores, key=scores.get, reverse=True)[:max_sentences]
    return " ".join(s.text.strip() for s in best).strip()


# ============================================================
# NUMERIC DATA EXTRACTOR
# ============================================================

def extract_numeric_data(text: str):
    """
    Extract (category, value) pairs from text.
    Examples it handles:
      - "Q1 12%, Q2 8%, Q3 15%, Q4 20%"
      - "iPhone 40%, Samsung 35%, Xiaomi 15%, Others 10%"
    Fallback: ["Data"], [100]
    """
    import re  # <-- FIXED (correct indent)

    # Extract Q1/Q2/Q3/Q4 style pairs
    matches = re.findall(r'(Q[1-4])\s*(\d+(?:\.\d+)?)%', text, re.IGNORECASE)

    if matches:
        categories = [m[0].upper() for m in matches]
        values = [float(m[1]) for m in matches]
        return categories, values

    # Generic extractor: e.g. iPhone 40%, Samsung 35%
    pairs = re.findall(r'([A-Za-z][A-Za-z0-9]+)\s*(\d+(?:\.\d+)?)%', text)
    if pairs:
        categories = [p[0] for p in pairs]
        values = [float(p[1]) for p in pairs]
        return categories, values

    # Last fallback
    perc = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    words = re.findall(r'\b[A-Za-z]+\b', text)

    if perc:
        return words[:len(perc)], [float(p) for p in perc]

    return ["Data"], [100]




# ============================================================
# PROMPT CLASSIFIER → CHOOSE: bar / pie / line
# ============================================================

def classify_prompt(text: str) -> str:
    """
    Decide which chart type to use: 'line', 'bar', or 'pie'.
    Priority:
      1) Trend / time (line)
      2) Comparison (bar)
      3) Distribution with % (pie)
    """
    t = text.lower()

    # 1️⃣ Trend / time-series → line
    if any(k in t for k in ["trend", "growth", "increase", "decrease", "rise", "fall", "q1", "q2", "q3", "q4", "year", "month"]):
        return "line"

    # 2️⃣ Explicit comparison → bar
    if any(k in t for k in ["compare", "vs", "versus", "difference", "against"]):
        return "bar"

    # 3️⃣ Distribution / share wording → pie
    if any(k in t for k in ["percent", "distribution", "market share", "share", "split"]) or "%" in t:
        return "pie"

    # Default fallback
    return "bar"



# ============================================================
# NLP PIPELINE USED BY /generate_video
# ============================================================

def nlp_pipeline(text: str, data=None):
    """
    Main NLP pipeline called from routes.py:
    - summarize text
    - extract categories + numeric values
    - classify chart type
    - generate narration audio
    """
    summary = small_summarizer(text)
    categories, values = extract_numeric_data(text)
    chart_type = classify_prompt(text)

    audio_path = AUDIO_UPLOAD_DIR / "summary_audio.mp3"
    gtts.gTTS(summary, lang="en").save(str(audio_path))

    return {
        "text": summary,
        "categories": categories,
        "values": values,
        "chart_type": chart_type,
        "audio": str(audio_path)
    }


# ============================================================
# ANIMATION MAKERS (BAR / PIE / LINE)
# ============================================================

def make_bar_animation(categories, values) -> str:
    import matplotlib.pyplot as plt
    from PIL import Image
    import shutil, os

    frames_dir = BASE_DIR / "frames_bar"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---------- STATIC BAR CHART (Only 1 frame) ----------
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(categories, values, color="gold")
    ax.set_title("Comparison Chart", fontsize=18)
    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Value (%)", fontsize=14)
    ax.set_ylim(0, max(values) * 1.2)

    # Add labels above bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(values) * 0.05),
            f"{val}%",
            ha="center",
            fontsize=12
        )

    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    frame_path = frames_dir / "frame_1.png"
    plt.savefig(frame_path)
    plt.close()

    # ---------- CREATE GIF ----------
    gif_path = GENERATED_IMAGES_DIR / "animation_bar.gif"
    img = Image.open(frame_path)
    img.save(str(gif_path), save_all=True, append_images=[img] * 10, duration=150, loop=0)

    shutil.rmtree(frames_dir, ignore_errors=True)
    return str(gif_path)



def make_line_animation(categories, values):
    import matplotlib.pyplot as plt
    from PIL import Image
    import shutil, os

    frames_dir = BASE_DIR / "frames_line"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---------- STATIC LINE CHART (Only one frame) ----------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(categories, values, marker='o', linewidth=3, color='blue')
    ax.set_title("Growth Trend", fontsize=18)
    ax.set_xlabel("Quarter", fontsize=14)
    ax.set_ylabel("Value (%)", fontsize=14)
    ax.set_ylim(0, max(values) * 1.2)

    # Add data labels above points
    for cat, val in zip(categories, values):
        ax.text(cat, val + (max(values)*0.05), f"{val}%", ha='center', fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    frame_path = frames_dir / "frame_1.png"
    plt.savefig(frame_path)
    plt.close()

    # ---------- CREATE GIF ----------
    gif_path = GENERATED_IMAGES_DIR / "animation_line.gif"
    img = Image.open(frame_path)
    img.save(str(gif_path), save_all=True, append_images=[img]*10, duration=150, loop=0)

    shutil.rmtree(frames_dir, ignore_errors=True)
    return str(gif_path)




# ============================================================
# CREATE ANIMATED GIF (ENTRY USED IN /generate_video)
# ============================================================

def create_animated_gif(text: str) -> str:
    """
    Frontend calls /generate_video → routes.py calls create_animated_gif(text).
    This function:
    - runs nlp_pipeline
    - picks chart type
    - delegates to the right animation function
    - returns full GIF path
    """
    parsed = nlp_pipeline(text)
    chart_type = parsed["chart_type"]
    categories = parsed["categories"]
    values = parsed["values"]

    if chart_type == "pie":
        return make_pie_animation(categories, values)
    elif chart_type == "line":
        return make_line_animation(categories, values)
    else:  # default bar
        return make_bar_animation(categories, values)


# ============================================================
# GIF → STORYTELLING VIDEO (MP4)
# ============================================================

def convert_gif_to_storytelling_video(gif_path: str, text: str) -> str:
    """
    Converts a GIF into a storytelling MP4 video with title and explanation frames.
    Used by /generate_video route.
    """
    # -----------------------------
    # TITLE EXTRACTOR
    # -----------------------------
    def extract_title(prompt: str) -> str:
        m = re.search(r"(visualize|show|compare|create)[\s:]+(.+?)(:|,|\.|$)", prompt, re.IGNORECASE)
        if m:
            return m.group(2).strip().capitalize()

        if ":" in prompt:
            return prompt.split(":", 1)[0].strip().capitalize()

        return prompt.strip().capitalize()

    title = extract_title(text)

    # -----------------------------
    # TEXT FRAME GENERATOR
    # -----------------------------
    def create_text_frame(txt: str, size=(1920, 1080), bg="black"):
        img = Image.new("RGB", size, color=bg)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except Exception:
            font = ImageFont.load_default()

        # Compute text size safely
        try:
            bbox = draw.textbbox((0, 0), txt, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(txt, font=font)

        x = (size[0] - w) // 2
        y = (size[1] - h) // 2

        draw.text((x, y), txt, fill="white", font=font)
        return np.array(img)

    # -----------------------------
    # BUILD VIDEO FRAMES
    # -----------------------------
    frames = []
    fps = 30

    # 1️⃣ TITLE SEQUENCE (2 sec)
    title_frame = create_text_frame(title, bg="black")
    for _ in range(2 * fps):
        frames.append(title_frame)

    # 2️⃣ GIF FRAMES (about 4 seconds)
    gif = Image.open(gif_path)
    gif_frames = []

    try:
        while True:
            frame = gif.copy().convert("RGB")
            frame = frame.resize((1920, 1080), Image.LANCZOS)
            gif_frames.append(np.array(frame))
            gif.seek(len(gif_frames))
    except EOFError:
        pass

    needed = 4 * fps
    while len(gif_frames) < needed and gif_frames:
        gif_frames.extend(gif_frames)

    frames.extend(gif_frames[:needed])

    # 3️⃣ STORY TEXT SEQUENCE (3 sec)
    explanations = [
        "Analyzing data...",
        "Extracting insights...",
        "Building your story..."
    ]
    per_exp = int((3 * fps) / len(explanations)) if explanations else 0

    for exp in explanations:
        exp_frame = create_text_frame(exp)
        for _ in range(per_exp):
            frames.append(exp_frame)

    # -----------------------------
    # SAVE FINAL VIDEO
    # -----------------------------
    output_path = GENERATED_IMAGES_DIR / "data_storytelling_video.mp4"

    writer = imageio.get_writer(str(output_path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"Video saved: {output_path}")
    return str(output_path)

    # ============================================================
# PIE CHART ANIMATION
# ============================================================

def make_pie_animation(categories, values) -> str:
    import matplotlib.pyplot as plt
    from PIL import Image
    import shutil

    frames_dir = BASE_DIR / "frames_pie"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---------- STATIC PIE CHART (Only 1 frame) ----------
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
    ax.set_title("Distribution Breakdown", fontsize=18)
    plt.tight_layout()

    frame_path = frames_dir / "frame_1.png"
    plt.savefig(frame_path)
    plt.close()

    # ---------- CREATE GIF ----------
    global GENERATED_IMAGES_DIR
    GENERATED_IMAGES_DIR = BASE_DIR / "generated_images"
    GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

    gif_path = GENERATED_IMAGES_DIR / "animation_pie.gif"
    img = Image.open(frame_path)
    img.save(str(gif_path), save_all=True, append_images=[img] * 10, duration=150, loop=0)

    shutil.rmtree(frames_dir, ignore_errors=True)
    return str(gif_path)





# # Create the animated GIF
# gif_path = create_animated_gif('30% dogs use nokia and 90% use iphones')

# # Convert to storytelling video
# video_path = convert_gif_to_storytelling_video(gif_path, '30% dogs use nokia and 90% use iphones')




# --------------------------------------OLD Code Pexels API Implementation  --------------------------------



# # Load environment variables
# load_dotenv()


# PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
# print(f"Your API Key: {PEXELS_API_KEY}")

# # Text processing functions
# def parse_input_text(text):
#     data_points = {}
#     parts = text.split(',')
#     for part in parts:
#         part = part.strip()
#         if '%' in part:
#             key, value = part.split('%')
#             data_points[key.strip()] = float(value.strip().replace('%', '')) / 100
#         else:
#             try:
#                 key_value = part.split(' ', 1)
#                 if len(key_value) == 2:
#                     key, value = key_value
#                     data_points[key.strip()] = value.strip()
#                 else:
#                     try:
#                         data_points[part] = float(part)
#                     except ValueError:
#                         print(f"Could not parse part: '{part}'")
#             except ValueError:
#                 print(f"Could not parse part: '{part}'")
#                 continue

#     # Additional processing to convert data points into relevant information
#     relevant_info = {}
#     for key, value in data_points.items():
#         if isinstance(value, float):
#             relevant_info[key] = f"{value * 100}%"
#         else:
#             relevant_info[key] = value

#     return relevant_info


# def preprocess_text(text):
#     cleaned_text = text.strip().lower()
#     cleaned_text = ' '.join(sorted(set(cleaned_text.split()), key=lambda x: cleaned_text.index(x)))
#     return cleaned_text


# def analyze_sentiment(text):
#     analysis = TextBlob(text)
#     return analysis.sentiment.polarity


# def semantic_segment_transformation(text):
#     contextual_prompt = f"Contextual prompt based on: {text}"
#     return contextual_prompt



# # Video Generation Function
# def generate_video_from_text(text):
#     # data = parse_input_text(text)
#     print("Parsed Data:", text)
    
#     contextual_prompt = semantic_segment_transformation(text)
#     processed_prompt = preprocess_text(contextual_prompt)
#     visualization_prompt = f"Create an animated infographic video showing the distribution of: {processed_prompt}"

#     headers = {'Authorization': PEXELS_API_KEY}
#     params = {'query': f'infographics {visualization_prompt}', 'per_page': 5}

#     time.sleep(3)
#     response = requests.get('https://api.pexels.com/videos/search', headers=headers, params=params)
    
#     if response.status_code != 200:
#         raise Exception(f"Pexels API request failed with status code {response.status_code}")

#     videos = response.json().get('videos', [])
#     video_urls = [video['video_files'][0]['link'] for video in videos if video['video_files']]
    
#     time.sleep(3)
#     return video_urls[:3]


# # Log types
# LOG_TYPE_GPT = "GPT"
# LOG_TYPE_PEXEL = "PEXEL"

# # log directory paths
# DIRECTORY_LOG_GPT = ".logs/gpt_logs"
# DIRECTORY_LOG_PEXEL = ".logs/pexel_logs"

# # method to log response from pexel and openai
# def log_response(log_type, query,response):
#     log_entry = {
#         "query": query,
#         "response": response,
#         "timestamp": datetime.now().isoformat()
#     }
#     if log_type == LOG_TYPE_GPT:
#         if not os.path.exists(DIRECTORY_LOG_GPT):
#             os.makedirs(DIRECTORY_LOG_GPT)
#         filename = '{}_gpt3.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
#         filepath = os.path.join(DIRECTORY_LOG_GPT, filename)
#         with open(filepath, "w") as outfile:
#             outfile.write(json.dumps(log_entry) + '\n')

#     if log_type == LOG_TYPE_PEXEL:
#         if not os.path.exists(DIRECTORY_LOG_PEXEL):
#             os.makedirs(DIRECTORY_LOG_PEXEL)
#         filename = '{}_pexel.txt'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
#         filepath = os.path.join(DIRECTORY_LOG_PEXEL, filename)
#         with open(filepath, "w") as outfile:
#             outfile.write(json.dumps(log_entry) + '\n')



# --------------------------------------OLD Code Pexels API Implementation  --------------------------------