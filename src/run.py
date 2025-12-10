import subprocess
import sys
import os
import pathlib
import argparse  # Added to handle command line arguments

def run_pipeline(video_filename, model_type="gpt-4o"):
    # 1. Setup Paths & Names
    if not os.path.exists(video_filename):
        print(f"❌ Error: Input file '{video_filename}' not found.")
        return

    # Extract base name (e.g. "11" from "11.mp4")
    base_name = pathlib.Path(video_filename).stem
    
    # Define Output Directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. The Raw Analysis Log
    report_file = os.path.join(output_dir, f"{base_name}_raw_log.txt")
    
    # 2. The JSON Cuts File
    json_file = os.path.join(output_dir, f"{base_name}_report.json")
    
    # 3. The Readable Text Report
    readable_report = os.path.join(output_dir, f"{base_name}_video_description.txt")
    
    # 4. Final Video
    final_output_video = os.path.join(output_dir, f"{base_name}_edit.mp4")

    print(f"=================================================")
    print(f"   🚀 PROCESSING: {video_filename}")
    print(f"   📂 OUTPUT ID:  {base_name}")
    print(f"=================================================\n")

    # --- STEP 1: ANALYSIS ---
    print(f"--- [Step 1/2] Analyzing Video ---")
    
    # We pass all necessary paths to the analysis script so it knows where to save
    analyze_cmd = [
        sys.executable, "video_processing_gui.py",
        "--video", video_filename,
        "--model", model_type,
        "--report_out", report_file,       
        "--json_out", json_file,           
        "--readable_out", readable_report  
    ]

    try:
        subprocess.run(analyze_cmd, check=True) 
        print(f"    ✅ Analysis saved to: {json_file}\n")
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Analysis Failed: {e}")
        return

    # --- STEP 2: EDITING ---
    print(f"--- [Step 2/2] Stitching Video ---")

    edit_cmd = [
        sys.executable, "mix.py",
        "--input_video", video_filename,
        "--cuts_json", json_file,          
        "--output_video", final_output_video 
    ]

    try:
        subprocess.run(edit_cmd, check=True)
        print(f"    ✅ Video saved to: {final_output_video}\n")
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Editing Failed: {e}")
        return

    print("=================================================")
    print(f"   🎉 DONE! Output folder: {output_dir}/")
    print("=================================================")

if __name__ == "__main__":
    # --- ARGUMENT PARSING LOGIC ---
    parser = argparse.ArgumentParser(description="AI Video Auto-Editor Pipeline")
    
    # This allows you to run: python run.py --video 12.mp4
    parser.add_argument("--video", type=str, default="11.mp4", help="The input video file path")
    
    # This allows you to run: python run.py --video 12.mp4 --model gpt-3.5-turbo
    parser.add_argument("--model", type=str, default="gpt-4o", help="The OpenAI model to use")
    
    args = parser.parse_args()
    
    # Pass the arguments from the terminal to the function
    run_pipeline(args.video, args.model)