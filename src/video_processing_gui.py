#!/usr/bin/env python
import os
import logging
import platform
import numpy as np
import cv2
import json
import re
import argparse
import subprocess
import gc
import warnings
from dotenv import load_dotenv
from openai import OpenAI
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper
from panns_inference import AudioTagging, labels as pann_labels
import librosa
from moviepy import VideoFileClip

load_dotenv()

# --- Configuration ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Classes & Utils ---
if platform.system() == "Windows":
    PANN_MODEL_PATH = r"C:\Users\naval\panns_data\cnn14.pth"
else:
    PANN_MODEL_PATH = "cnn14.pth" 

def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def seconds_to_timestr(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# --- Models ---
def get_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolo11x.pt")

def get_blip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return model, processor, device

# --- Processing Functions ---
def extract_audio(video_path, audio_path):
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_path, logger=None)

def transcribe_audio(audio_file):
    model = whisper.load_model("small") 
    try:
        result = model.transcribe(audio_file, language="en", condition_on_previous_text=False)
        return result["text"], result.get("segments", [])
    finally:
        del model
        free_gpu()

def detect_audio_events(audio_file):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pt_path = PANN_MODEL_PATH if os.path.exists(PANN_MODEL_PATH) else None
        model = AudioTagging(checkpoint_path=pt_path, device=device)
        waveform, sr = librosa.load(audio_file, sr=32000)
        
        events = {}
        segment_len = 5 * sr
        for i in range(0, len(waveform), segment_len):
            chunk = waveform[i:i+segment_len]
            if len(chunk) < sr: continue
            chunk_tensor = torch.tensor(chunk[None, :]).float().to(device)
            clipwise_output, _ = model.inference(chunk_tensor)
            clipwise_output = clipwise_output.cpu().detach().numpy()[0]
            if np.max(clipwise_output) > 0.2: 
                idx = np.argmax(clipwise_output)
                label = pann_labels[idx]
                time_str = seconds_to_timestr(i/sr)
                if label not in events: events[label] = []
                events[label].append(time_str)
        return events
    except Exception:
        return {}
    finally:
        free_gpu()

def call_openai(prompt, model="gpt-4o"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a Video Strategist. Output the Report first, then the JSON data."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


def get_viral_prompt(report_text):
    return f"""
    You are a Master Video Editor AI.
    
    INPUT DATA:
    1. **TRANSCRIPT** (Audio with timestamps)
    2. **VISUAL LOG** (Scene descriptions)
    3. **AUDIO EVENTS** (SFX, Laughter)

    --- GOAL ---
    Create a **Chronological Viral Highlight Reel** (20-25% of total duration).
    **Style:** "Machine Gun" Pacing. High density. No overlapping clips.

    --- STEP 0: DETECT GENRE & STRATEGY ---
    Determine the video genre based on the transcript and apply the correct editing rule:

    | Detected Genre | Editing Strategy | Key Elements to Keep |
    | :--- | :--- | :--- |
    | **Podcast / Interview** | **"The Golden Nugget"** | Keep the specific question + the strongest answer. Cut small talk/pauses. |
    | **Movie / Narrative** | **"The Trailer"** | Intro Hook + Rising Action + Climax. Do NOT spoil the ending. |
    | **Tutorial / Educational** | **"The Quick Fix"** | Problem Statement + The Solution Steps. Cut the "Hey guys" intro. |
    | **Vlog / Lifestyle** | **"The Highlight Reel"** | High-energy visual moments + funniest jokes/reactions. |


    --- STEP 1: MACRO STRATEGY (The Architect) ---
    1. **Scan ALL Chapters:** Identify every distinct high-value topic (Information, Humor, Action).
    2. **Discard Low Value:** Remove small talk, logistical setup, and "host rambling".

    --- STEP 2: MICRO REFINEMENT (The Butcher) ---
    Inside each topic, keep only the top 20% of lines. 
    **CRITICAL RULES:**
    1. **Chronological Order:** Clip B MUST start *after* Clip A ends. Never output overlapping time ranges.
    2. **The 0.5s Split:** If a speaker pauses for >0.5s, SPLIT the clip into two entries. Do not include the silence.
    3. **Visual Sync:** Prioritize audio that matches the Visual Log (e.g., Audio: "This chart" / Visual: "Chart").

    --- STEP 3: SENTENCE INTEGRITY PROTOCOL ---
    * **Check the End:** Look at the text at your chosen `end` timestamp.
    * **Punctuation Match:** The clip MUST end on a full stop (.), question mark (?), or exclamation (!).

    --- AUTOMATION DATA (STRICT FLAT JSON) ---
    Output a **FLAT LIST** of clips in strictly chronological order.

    ```json
    [
        {{
            "start": "HH:MM:SS",
            "end": "HH:MM:SS",
            "topic_group": "Topic 1: [Topic Name]",
            "description": "Context for this cut.",
            "end_text": "[Last 3 words of this clip - for verification]",
            "viral_score": "9/10"
        }},
        {{
            "start": "HH:MM:SS",
            "end": "HH:MM:SS",
            "topic_group": "Topic 1: [Topic Name]",
            "description": "Next sentence (skipped silence).",
            "end_text": "[Last 3 words]",
            "viral_score": "9/10"
        }}
    ]
    ```

    --- REPORT DATA ---
    {report_text}

    --- STRICT OUTPUT RULES ---
    1. Output **ONLY valid JSON**. No Markdown (no ```json).
    2. **NO OVERLAPS:** Ensure Start Time > Previous End Time.
    3. **VERIFY TEXT:** Ensure the `end` timestamp covers the `end_text` fully.
    """

def extract_json_from_text(text):
    """Robust extraction of JSON list from a mixed text response."""
    # Pattern to find a JSON list enclosed in brackets
    pattern = r"\[\s*\{.*?\}\s*\]"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(0)
    
    # Fallback: look for markers like ```json
    start = text.find("```json")
    if start != -1:
        end = text.rfind("```")
        if end > start:
            return text[start+7:end].strip()
            
    return None

def generate_video_descriptions(report_file, json_file, readable_file, model_name):
    if not os.path.exists(report_file):
        logging.error(f"{report_file} not found.")
        return

    with open(report_file, "r", encoding="utf-8") as f:
        report_text = f.read()

    logging.info(f"Generating comprehensive report and cuts...")
    response = call_openai(get_viral_prompt(report_text), model_name)
    
    try:
        # 1. Save the FULL Readable Report (Text)
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write(response)
        logging.info(f"SUCCESS: Report saved to {readable_file}")

        # 2. Extract and Save the JSON Data
        json_str = extract_json_from_text(response)
        
        if json_str:
            try:
                data = json.loads(json_str)
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                logging.info(f"SUCCESS: JSON Data saved to {json_file}")
            except json.JSONDecodeError as e:
                logging.error(f"JSON Decode Error: {e}. Trying to fix quotes...")
                # Attempt simple fix
                fixed_str = json_str.replace("'", '"')
                data = json.loads(fixed_str)
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
        else:
            logging.error("Could not find JSON block in AI response.")

    except Exception as e:
        logging.error(f"Error processing AI response: {e}")

def process_video(args):
    # Import difflib here so you don't need to scroll to the top of your file
    import difflib 

    # Setup Logging locally
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Temp Audio
    temp_audio = "temp_process_audio.wav"
    extract_audio(args.video, temp_audio)
    
    transcript, segments = transcribe_audio(temp_audio)
    audio_events = detect_audio_events(temp_audio)
    
    if os.path.exists(temp_audio): os.remove(temp_audio)

    # Visual Analysis
    # NOTE: You can uncomment yolo if you ever need object detection back
    # yolo = get_yolo_model()
    
    blip_model, blip_proc, device = get_blip_model()
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    last_caption = ""
    
    report_data = [f"Video: {args.video}\n=== TRANSCRIPT ==="]
    for s in segments:
        report_data.append(f"[{seconds_to_timestr(s['start'])} -> {seconds_to_timestr(s['end'])}] {s['text']}")
    
    report_data.append(f"\n=== AUDIO EVENTS ===\n{str(audio_events)}")
    report_data.append("\n=== VISUAL LOG ===")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % args.sample_rate != 0: continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        try:
            inputs = blip_proc(pil_img, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_length=50, min_length=10)
            caption = blip_proc.decode(out[0], skip_special_tokens=True)
        except: caption = ""

        if not caption or len(caption) < 5: continue
        
        # --- NEW SIMILARITY LOGIC START ---
        # Calculate similarity (0.0 to 1.0) between current and last caption
        similarity = difflib.SequenceMatcher(None, caption.strip().lower(), last_caption.strip().lower()).ratio()
        
        # If > 85% similar, skip this log entry
        if similarity > 0.85:
            continue
        # --- NEW SIMILARITY LOGIC END ---

        last_caption = caption
        
        timestamp = seconds_to_timestr(frame_idx / fps)
        logging.info(f"{timestamp}: {caption}")
        report_data.append(f"Time {timestamp} | Scene: {caption}")

    cap.release()
    free_gpu()

    # Save Report
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_data))
    
    # Generate JSON
    generate_video_descriptions(args.report_out, args.json_out, args.readable_out, args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--sample_rate", type=int, default=250)
    
    # Dynamic Output Arguments
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--json_out", required=True)
    parser.add_argument("--readable_out", required=True)
    
    args = parser.parse_args()
    process_video(args)