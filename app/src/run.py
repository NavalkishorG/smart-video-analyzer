import subprocess
import sys
import os
import pathlib

def run_pipeline(video_filename, model_type="gpt-4o"):
    # 1. Setup Paths & Names
    if not os.path.exists(video_filename):
        print(f"❌ Error: Input file '{video_filename}' not found.")
        return

    # Extract "11" from "11.mp4"
    base_name = pathlib.Path(video_filename).stem
    
    # Define Output Directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # --- [CHANGE IS HERE] ---
    # 1. The Raw Analysis Log (stays the same)
    report_file = os.path.join(output_dir, f"raw_log_{base_name}.txt")
    
    # 2. The JSON Cuts File (Now named 'viral_report.json')
    json_file = os.path.join(output_dir, f"report_{base_name}.json")
    
    # 3. The Readable Text Report (Now named 'video_description.txt')
    readable_report = os.path.join(output_dir, f"video_description_{base_name}.txt")
    
    # 4. Final Video
    final_output_video = os.path.join(output_dir, f"{base_name}_edit.mp4")

    print(f"=================================================")
    print(f"   🚀 PROCESSING: {video_filename}")
    print(f"   📂 OUTPUT ID:  {base_name}")
    print(f"=================================================\n")

    # --- STEP 1: ANALYSIS ---
    print(f"--- [Step 1/2] Analyzing Video ---")
    
    analyze_cmd = [
        sys.executable, "video_processing_gui.py",
        "--video", video_filename,
        "--model", model_type,
        "--report_out", report_file,       
        "--json_out", json_file,           # Will now save to viral_report_11.json
        "--readable_out", readable_report  # Will now save to video_description_11.txt
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
        "--cuts_json", json_file,          # mix.py will now look for viral_report_11.json
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
    # --- ONLY CHANGE THIS LINE ---
    TARGET_VIDEO = "11.mp4" 
    
    run_pipeline(TARGET_VIDEO)