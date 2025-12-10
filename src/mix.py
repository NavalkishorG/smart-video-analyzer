import os
import json
import argparse
import textwrap  # Added for wrapping text
import numpy as np
from moviepy import VideoFileClip, ImageClip, ColorClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

def create_text_image(text, width, height, duration=2): # Reduced duration to 2s
    # Create black background
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Dynamic Font Sizing (Smaller to ensure fit)
    try:
        # Use 5% of height for font size (smaller than previous 10%)
        font_size = int(height * 0.05) 
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    # --- TEXT WRAPPING LOGIC ---
    # Estimate characters per line based on width and font size
    # This ensures text stays visible!
    avg_char_width = font_size * 0.6
    chars_per_line = int((width * 0.8) / avg_char_width) # Use 80% of screen width
    
    lines = textwrap.wrap(text, width=chars_per_line)
    
    # Calculate total text block height to center it vertically
    line_height = font_size * 1.5
    total_text_height = len(lines) * line_height
    current_y = (height - total_text_height) / 2
    
    # Draw each line centered
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) / 2
        draw.text((x, current_y), line, font=font, fill=(255, 255, 255))
        current_y += line_height
        
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
                text_overlay = clip_data['description'] # Using description as title
                
                print(f"Clip {i+1}: {text_overlay} ({start}-{end})")

                # 1. Add Title Card (Shorter duration now)
                desc_clip = create_text_image(text_overlay, w, h, duration=2)
                final_sequence.append(desc_clip)

                # 2. Add Video Segment
                video_clip = video.subclipped(start, end)
                final_sequence.append(video_clip)

                # 3. Add Black Screen (Reduced to 1s for speed)
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