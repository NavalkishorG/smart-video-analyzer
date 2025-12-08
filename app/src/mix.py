import os
import json
import argparse
import numpy as np
from moviepy import VideoFileClip, ImageClip, ColorClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

def create_text_image(text, width, height, duration=3):
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font_size = int(height / 10)
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) / 2
    y = (height - text_h) / 2
    
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return ImageClip(np.array(img)).with_duration(duration)

def create_highlight_reel(input_filename, output_filename, cuts_json):
    if not os.path.exists(cuts_json):
        print(f"Error: JSON '{cuts_json}' not found.")
        return

    with open(cuts_json, "r") as f:
        cuts_data = json.load(f)

    print(f"Loaded {len(cuts_data)} clips.")

    try:
        with VideoFileClip(input_filename) as video:
            w, h = video.size
            final_sequence = []

            for i, clip_data in enumerate(cuts_data):
                start = clip_data['start']
                end = clip_data['end']
                text_overlay = clip_data['description']
                
                print(f"Clip {i+1}: {text_overlay} ({start}-{end})")

                desc_clip = create_text_image(text_overlay, w, h, duration=3)
                final_sequence.append(desc_clip)

                video_clip = video.subclipped(start, end)
                final_sequence.append(video_clip)

                black_screen = ColorClip(size=(w, h), color=(0,0,0)).with_duration(1)
                final_sequence.append(black_screen)

            print(f"Saving to {output_filename}...")
            final_video = concatenate_videoclips(final_sequence, method="compose")
            final_video.write_videofile(
                output_filename, 
                codec="libx264", 
                audio_codec="aac",
                fps=24,
                preset="medium",
                threads=4
            )
            print("Done!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--cuts_json", required=True)
    parser.add_argument("--output_video", required=True)
    
    args = parser.parse_args()
    
    create_highlight_reel(args.input_video, args.output_video, args.cuts_json)