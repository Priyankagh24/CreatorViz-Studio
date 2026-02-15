"""
This script imports the necessary libraries for a comprehensive data processing, visualization, and video generation pipeline.
Libraries:
- pandas: Data manipulation and analysis.
- numpy: Numerical operations.
- seaborn: Statistical data visualization.
- sklearn.model_selection: Splitting data into training and testing sets.
- sklearn.preprocessing: Standardizing features.
- sklearn.linear_model: Implementing linear regression models.
- sklearn.metrics: Evaluating model performance.
- pandas.read_csv, pandas.read_excel: Reading data from CSV and Excel files.
- scipy.stats: Statistical functions.
- moviepy.editor: Video editing and creation.
- gtts: Text-to-speech conversion.
- matplotlib.pyplot: Plotting graphs and charts.
"""
# Loading necessary Libraries

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv, read_excel
from scipy import stats
# from moviepy.editor import VideoClip, concatenate_videoclips, VideoFileClip, AudioFileClip, ImageSequenceClip
from gtts import gTTS
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os
import dill as pickle
import requests
import time
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio
import uuid

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv, read_excel
from scipy import stats
from moviepy.editor import VideoClip, concatenate_videoclips, VideoFileClip, AudioFileClip, ImageSequenceClip
from gtts import gTTS
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os
import dill as pickle
import requests
import time
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio
import uuid
# Loading all The necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv, read_excel
from scipy import stats
from moviepy.editor import VideoClip, concatenate_videoclips
from gtts import gTTS
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
from time import perf_counter
import math
# from config_loader import load_config_ani, load_config_setup
import matplotlib.pyplot as plt
import os
import dill as pickle
import gtts
import requests
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio
import uuid
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast
import re
import os
import uuid
import gtts
from textblob import TextBlob
from langdetect import detect
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv, read_excel
from scipy import stats
from moviepy.editor import VideoClip, concatenate_videoclips, VideoFileClip, AudioFileClip, ImageSequenceClip
from gtts import gTTS
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os
import dill as pickle
import requests
import time
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio
import uuid

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv, read_excel
from scipy import stats
from moviepy.editor import VideoClip, concatenate_videoclips, VideoFileClip, AudioFileClip, ImageSequenceClip, TextClip, CompositeVideoClip
from gtts import gTTS
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os
import dill as pickle
import requests
import time
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio
import uuid
import logging
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv, read_excel
from scipy import stats
from moviepy.editor import VideoClip, concatenate_videoclips, VideoFileClip, AudioFileClip, ImageSequenceClip, TextClip, CompositeVideoClip, ImageClip
from gtts import gTTS
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os
import dill as pickle
import requests
import time
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio
import uuid
import logging
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pandas import read_csv, read_excel
from scipy import stats
from gtts import gTTS
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os
import dill as pickle
import requests
import time
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import shutil
from PIL import Image, ImageDraw, ImageFont
import imageio
import uuid
import logging
from tqdm import tqdm
# from manim import *
import gtts
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from langdetect import detect
from tqdm import tqdm
import os

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_UPLOAD_DIR = BASE_DIR / "uploads" / "audio_files"
AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



# Set the HF_HOME environment variable to change the cache path
os.environ['HF_HOME'] = r"C:\DataViz-AI\cache_models"
 # Change this to your desired cache path

# Load the model and tokenizer from .pkl files
# Load summarization model directly from Hugging Face instead of pickled files
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SUMMARY_MODEL_NAME = "facebook/bart-large-cnn"

try:
    summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME)
    print(f"[INFO] Loaded summarization model: {SUMMARY_MODEL_NAME}")
except Exception as e:
    print(f"[WARN] Could not load summarization model '{SUMMARY_MODEL_NAME}': {e}")
    summary_tokenizer = None
    summary_model = None



# Initialize sentiment analyzer
# vader_analyzer = SentimentIntensityAnalyzer()

def nlp_csv_to_video_pipeline(df: pd.DataFrame) -> dict:
    """
    Core CSV → Narration pipeline.
    Takes a DataFrame and returns:
    - narration text
    - path to generated audio file
    """
    # 1. Build rich narration text from the dataframe
    narration_text = generate_detailed_narration_from_df(df)

    # 2. Generate audio file using gTTS
    audio_filename = f"csv_narration_{uuid.uuid4().hex}.mp3"
    audio_path = AUDIO_UPLOAD_DIR / audio_filename

    tts = gtts.gTTS(narration_text, lang="en")
    tts.save(str(audio_path))

    return {
        "text": narration_text,
        "audio_path": str(audio_path)
    }


# # Example usage
# if __name__ == "__main__":
#     text = "20% of users own an iPhone, 50% own a Samsung, and the rest own a variety of brands"
#     summary = nlp_pipeline(text)
#     print(summary)

def read_data(file_path: str) -> pd.DataFrame:
    """
    Read CSV or Excel into a clean DataFrame.
    Supports: .csv, .xlsx, .xls
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        # engine='openpyxl' is needed for modern Excel files
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Drop completely empty rows/columns
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    return df


def generate_detailed_narration_from_df(df: pd.DataFrame, max_columns=8) -> str:
    """
    A fully upgraded, documentary-style storytelling engine.
    Extracts meaning, trends, outliers, correlations, and category patterns.
    """

    parts = []
    n_rows, n_cols = df.shape
    columns = df.columns[:max_columns]

    # -------------------------------------------------------------------
    # 🎬 SECTION 1 — INTRO
    # -------------------------------------------------------------------
    parts.append(f"This dataset contains {n_rows} rows and {n_cols} columns.")
    parts.append("Let's explore the key patterns and insights hidden inside this data.")

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # -------------------------------------------------------------------
    # 🎬 SECTION 2 — NUMERIC ANALYSIS
    # -------------------------------------------------------------------
    if numeric_cols:
        parts.append("")
        parts.append("### Numerical Overview")

        for col in numeric_cols[:max_columns]:
            s = df[col].dropna()
            if s.empty:
                continue

            mean = s.mean()
            min_val = s.min()
            max_val = s.max()
            std = s.std()

            parts.append(
                f"For the numeric column '{col}', values range from {min_val:.2f} to {max_val:.2f}, "
                f"with an average of {mean:.2f}. The variation, measured by standard deviation, "
                f"is {std:.2f}, indicating {'high' if std > mean else 'moderate'} fluctuation."
            )

            # Detect Outliers
            z_scores = (s - mean) / std if std != 0 else None
            if z_scores is not None:
                outliers = s[abs(z_scores) > 2]
                if len(outliers) > 0:
                    parts.append(
                        f"Notably, we observe {len(outliers)} outlier values in '{col}', "
                        f"such as {outliers.head(3).tolist()}."
                    )

            # Detect Trend if values align sequentially
            if len(s) >= 4:
                corr = np.corrcoef(range(len(s)), s)[0, 1]
                if abs(corr) > 0.5:
                    direction = "an upward" if corr > 0 else "a downward"
                    parts.append(
                        f"'{col}' shows {direction} trend across its sequence, suggesting "
                        f"a meaningful directional pattern."
                    )

    # -------------------------------------------------------------------
    # 🎬 SECTION 3 — CATEGORICAL ANALYSIS
    # -------------------------------------------------------------------
    if categorical_cols:
        parts.append("")
        parts.append("### Categorical Patterns")

        for col in categorical_cols[:max_columns]:
            s = df[col].dropna().astype(str)
            if s.empty:
                continue

            vc = s.value_counts(normalize=True)

            # Skip columns like names or IDs
            avg_len = s.str.len().mean()
            if avg_len > 20 and len(vc) > len(df) * 0.2:
                continue

            parts.append(f"The column '{col}' appears to be categorical.")

            top_values = vc.head(5)
            for val, pct in top_values.items():
                parts.append(f"• {val}: {pct * 100:.1f}%")

            dominant = top_values.index[0]
            dominant_pct = top_values.iloc[0] * 100
            parts.append(
                f"The most dominant category is '{dominant}', representing {dominant_pct:.1f}% of the data."
            )

    # -------------------------------------------------------------------
    # 🎬 SECTION 4 — CORRELATIONS & RELATIONSHIPS
    # -------------------------------------------------------------------
    if len(numeric_cols) >= 2:
        parts.append("")
        parts.append("### Relationships Between Columns")

        corr_matrix = df[numeric_cols].corr()

        # Find strong correlations
        strong_pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.65:
                    strong_pairs.append((col1, col2, corr))

        if strong_pairs:
            for col1, col2, corr in strong_pairs[:5]:
                relation = "positively" if corr > 0 else "negatively"
                parts.append(
                    f"'{col1}' and '{col2}' are {relation} correlated with a coefficient of {corr:.2f}. "
                    f"This suggests a meaningful relationship between these variables."
                )
        else:
            parts.append("No strong correlations were observed between major numeric features.")

    # -------------------------------------------------------------------
    # 🎬 SECTION 5 — FINAL STORY OUTRO
    # -------------------------------------------------------------------
    parts.append("")
    parts.append(
        "These insights together paint a comprehensive story of the dataset. "
        "The following video visualizes these findings with charts and narration."
    )

    return " ".join(parts)




def select_visualization_method(df):
    """
    Return ONLY column names.
    Visualization type will be decided later.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Prefer numeric columns
    if numeric_cols:
        return numeric_cols[:3]

    # fallback to categorical
    return categorical_cols[:3]





def create_visualizations(df, selected_columns=None):
    import matplotlib.pyplot as plt
    import uuid
    import logging
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUTS_FOLDER = BASE_DIR / "outputs"

    image_paths = []

    # Auto-detect numeric columns if none provided
    if not selected_columns:
        selected_columns = df.select_dtypes(include=['number']).columns.tolist()

    if not selected_columns:
        return []

    # 🔥 FIX: Extract only column names from tuples ("chart", "column")
    clean_columns = []
    for item in selected_columns:
        if isinstance(item, tuple) and len(item) >= 2:
            clean_columns.append(item[1])   # keep only column name
        else:
            clean_columns.append(item)

    selected_columns = clean_columns[:3]  # limit to 3 columns max

    logging.info(f"[CLEANED] Columns used for visualization: {selected_columns}")

    # Create charts
    for col in selected_columns:
        try:
            plt.figure(figsize=(8, 5))
            df[col].plot(kind='bar', title=f"{col} Distribution")

            filename = f"{uuid.uuid4().hex}.png"
            image_path = OUTPUTS_FOLDER / filename

            plt.savefig(image_path)
            plt.close()

            image_paths.append(str(image_path))
            logging.info(f"[OK] Visualization saved: {image_path}")

        except Exception as e:
            logging.error(f"[ERROR] Chart failed for column {col}: {e}")

    # Fallback
    if not image_paths:
        try:
            plt.figure(figsize=(8, 5))
            df[selected_columns[0]].plot(kind='bar', title=f"{selected_columns[0]} Distribution")

            filename = f"{uuid.uuid4().hex}.png"
            image_path = OUTPUTS_FOLDER / filename
            plt.savefig(image_path)
            plt.close()

            image_paths.append(str(image_path))
            logging.info(f"[FALLBACK] One chart generated: {image_path}")

        except Exception as e:
            logging.error(f"[CRITICAL] Fallback visualization failed: {e}")
            return []

    return image_paths






def generate_video_from_images(image_paths, output_video_path, fps=10, max_resolution=(640, 480)):
    from moviepy.editor import ImageClip, concatenate_videoclips
    from PIL import Image
    import numpy as np
    import os
    import logging
    from tqdm import tqdm

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Resize images and store in memory
    resized_images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            resized_image = image.resize(max_resolution)
            resized_images.append(np.array(resized_image))
            logger.debug(f"Resized image: {image_path}")
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            continue

    # Process images in batches
    batch_size = 50
    batch_clips = []
    for i in range(0, len(resized_images), batch_size):
        batch = resized_images[i:i + batch_size]
        clips = [
    ImageClip(img)
    .set_duration(2)
    .crossfadein(0.5)
    for img in batch
]

        batch_clip = concatenate_videoclips(clips, method="compose", padding=-0.5)
        batch_clips.append(batch_clip)

    # Combine all batches
    final_clip = concatenate_videoclips(batch_clips, method="compose")
    final_clip.fps = fps

    # Write video to file with optimized settings
    output_video_path = os.path.abspath(output_video_path)
    try:
        final_clip.write_videofile(
            output_video_path,
            codec="libx264",
            preset="ultrafast",
            bitrate="500k"
        )
    except Exception as e:
        logger.error(f"Error writing video file: {e}")
        raise

    return output_video_path


from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path

def add_auto_generated_audio_to_video(video_path, audio_path, output_filename):
    # Accept both str and Path
    video_path = str(video_path)
    audio_path = str(audio_path)

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    final_clip = video_clip.set_audio(audio_clip)
    output_path = BASE_DIR / "uploads" / "videos" / output_filename
    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    video_clip.close()
    audio_clip.close()
    final_clip.close()
    return str(output_path)




def create_infographic_video(file_path):
    df = read_data(file_path)

    # NLP narration
    summary = nlp_csv_to_video_pipeline(df)

    # Directly use numeric columns for charts
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        logging.error("No numeric columns found for visualization.")
        return None

    # Generate chart images
    image_paths = create_visualizations(df, numeric_cols[:3])

    if not image_paths:
        logging.error("No chart images generated.")
        return None

    # Build video from chart images
    video_path = generate_video_from_images(
        image_paths,
        f"video_{uuid.uuid4().hex}.mp4"
    )

    # Add narration audio
    final_video_path = add_auto_generated_audio_to_video(
        video_path,
        summary["audio_path"],
        f"final_video_{uuid.uuid4().hex}.mp4"
    )

    logging.info(f"Infographic video created successfully: {final_video_path}")
    return final_video_path




# def process_data_and_create_video(file_path):
#     # Create the infographic video
#     create_infographic_video(file_path)

# # Example usage
# if __name__ == "__main__":
#     file_path = 'D:\\1OOx-enginners-hackathon-submission-2\\data\\2015.csv'
#     process_data_and_create_video(file_path)