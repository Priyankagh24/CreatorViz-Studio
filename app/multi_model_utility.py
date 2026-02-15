import os
import io
import uuid
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
from gtts import gTTS
from pathlib import Path

# =========================================
# CONFIG
# =========================================

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_UPLOAD_DIR = BASE_DIR / "uploads" / "audio_files"
AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================
# DATA LOADING
# =========================================

def load_and_preprocess_data(file_path):
    try:
        logging.info(f"Loading data from {file_path}")

        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            data = pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError("Unsupported file format.")

        logging.info("Data loaded successfully")

        # 🔥 FIX: Convert numeric columns properly
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna(how="all")

        return data

    except Exception as e:
        logging.error(f"Error loading and preprocessing data: {e}")
        raise

# =========================================
# SIMPLE EDA
# =========================================

def perform_eda(data):
    logging.info("Performing EDA")

    return {
        "columns": data.columns.tolist(),
        "dtypes": data.dtypes.apply(lambda x: x.name).to_dict()
    }


# =========================================
# PROMPT ANALYSIS
# =========================================

def analyze_prompt_for_insights(prompt, eda_summary):
    logging.info("Analyzing prompt")

    text = prompt.lower()
    insights = []

    keywords = ["trend", "comparison", "distribution", "correlation", "growth"]

    for word in keywords:
        if word in text:
            insights.append(word)

    if not insights:
        insights = ["comparison"]

    # 🔥 Smart column selection
    numeric_cols = [col for col, dtype in eda_summary["dtypes"].items()
                    if dtype in ["int64", "float64"]]

    categorical_cols = [col for col, dtype in eda_summary["dtypes"].items()
                        if dtype == "object"]

    if numeric_cols and categorical_cols:
        columns = [categorical_cols[0], numeric_cols[0]]
    elif numeric_cols:
        columns = numeric_cols[:2]
    else:
        columns = eda_summary["columns"][:2]

    logging.info(f"Insights extracted: {insights}")
    logging.info(f"Columns selected: {columns}")

    return insights, columns


# =========================================
# FRAME GENERATION
# =========================================

def generate_animated_frames(data, insights, columns):
    logging.info("Generating frames")

    frames = []

    if len(columns) < 2:
        logging.warning("Not enough columns selected.")
        return frames

    x_col = columns[0]
    y_col = columns[1]

    if x_col not in data.columns or y_col not in data.columns:
        logging.warning("Selected columns not found in data.")
        return frames

    data[y_col] = pd.to_numeric(data[y_col], errors='coerce')
    data = data.dropna(subset=[y_col])

    if data[y_col].empty:
        logging.warning("No numeric data to plot.")
        return frames

    fig, ax = plt.subplots(figsize=(8, 5))

    # 🔥 Insight-based plotting
    if "growth" in insights or "trend" in insights:
        ax.plot(data[x_col].astype(str), data[y_col], marker='o')
        ax.set_title(f"{y_col} Growth Over {x_col}")

    elif "comparison" in insights:
        grouped = data.groupby(x_col)[y_col].mean()
        grouped.plot(kind="bar", ax=ax)
        ax.set_title(f"{y_col} Comparison by {x_col}")

    else:
        data[y_col].plot(kind="bar", ax=ax)
        ax.set_title(f"{y_col} Overview")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    frames.append(np.array(img))
    plt.close(fig)

    logging.info(f"Generated {len(frames)} frames")
    return frames

# =========================================
# VIDEO CREATION
# =========================================

def create_video_from_frames(frames, audio_file=None, output_file="final_production_model.mp4"):
    if not frames:
        raise ValueError("No frames to create video from.")

    clips = []

    for frame in frames:
        clip = ImageSequenceClip([frame], fps=1).set_duration(4)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")

    if audio_file and os.path.exists(audio_file):
        audio = AudioFileClip(audio_file)
        video = video.set_audio(audio)

    video.write_videofile(output_file, codec="libx264", fps=24)

    return output_file


# =========================================
# INFOGRAPHIC VIDEO PIPELINE
# =========================================

def generate_infographic_video(data, insights, columns, audio_file=None):
    frames = generate_animated_frames(data, insights, columns)

    if not frames:
        raise ValueError("No frames generated from data.")

    VIDEOS_FOLDER = BASE_DIR / "uploads" / "videos"
    VIDEOS_FOLDER.mkdir(parents=True, exist_ok=True)

    output_path = VIDEOS_FOLDER / f"pro_studio_{uuid.uuid4().hex}.mp4"

    final_video = create_video_from_frames(
        frames,
        audio_file,
        str(output_path)   # 🔥 Convert to string
    )

    return str(final_video)  # 🔥 Always return string path
# =========================================
# NARRATION
# =========================================

def generate_narration(prompt, insights, selected_columns, df):
    narration_text = (
        f"Based on your request: {prompt}. "
        f"This visualization highlights {', '.join(selected_columns)} "
        f"with focus on {', '.join(insights)}."
    )

    audio_filename = f"narration_{uuid.uuid4().hex}.mp3"
    audio_path = AUDIO_UPLOAD_DIR / audio_filename

    tts = gTTS(narration_text, lang="en")
    tts.save(str(audio_path))

    return str(audio_path)


# =========================================
# MASTER PIPELINE
# =========================================

def data_storytelling_pipeline(file_path, prompt):
    logging.info("Starting Pro Studio pipeline")

    data = load_and_preprocess_data(file_path)
    eda_summary = perform_eda(data)

    insights, selected_columns = analyze_prompt_for_insights(prompt, eda_summary)

    narration_file = generate_narration(
        prompt,
        insights,
        selected_columns,
        data
    )

    video_file = generate_infographic_video(
        data,
        insights,
        selected_columns,
        narration_file
    )

    logging.info("Pipeline completed successfully")
    return video_file