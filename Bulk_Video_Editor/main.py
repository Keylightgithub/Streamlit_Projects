'''
streamlit run /Users/isaachenry/Documents/VS_Code_intern_projects/Bulk_Video_Editor/main.py
'''

import streamlit as st
import os
import re
import subprocess
import tempfile
import zipfile
import threading
import queue
import time
from PIL import Image, ImageDraw
import pandas as pd # Imported pandas for dataframe

# --- Helper Functions (Modified for Ultimate Robustness) ---

def get_video_properties(video_path):
    """
    Gets width, height, total frame count, and duration of a video.
    MODIFICATION: Corrected the 'flat' format argument and enhanced
    error reporting to show ffprobe's specific error message.
    """
    try:
        command = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate',
            '-show_entries', 'format=duration',
            # CORRECTED: The separator option is 's', not 'p'.
            '-of', 'flat=s=_',
            video_path
        ]
        # Set check=True to raise an exception on non-zero exit codes.
        # Capture stderr to get detailed error messages from ffprobe.
        process = subprocess.run(command, check=True, capture_output=True, text=True)

        # Parse the key-value output into a dictionary
        props = {}
        for line in process.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                props[key] = value.strip('"')

        # Extract properties by key
        width_str = props.get('streams_stream_0_width', 'N/A')
        height_str = props.get('streams_stream_0_height', 'N/A')
        frames_str = props.get('streams_stream_0_nb_frames', 'N/A')
        frame_rate_str = props.get('streams_stream_0_avg_frame_rate', 'N/A')
        duration_str = props.get('format_duration', 'N/A')

        if any(v == 'N/A' for v in [width_str, height_str, duration_str]):
            st.error(f"Essential properties (width, height, or duration) are missing for {os.path.basename(video_path)}. Skipping.")
            return None, None, None, None

        width = int(width_str)
        height = int(height_str)
        duration = float(duration_str)

        if frames_str != 'N/A':
            frames = int(frames_str)
        else:
            st.warning(f"Frame count is 'N/A' for {os.path.basename(video_path)}. Calculating from duration and framerate.")
            if frame_rate_str == 'N/A' or frame_rate_str == '0/0':
                st.error(f"Cannot determine total frames for {os.path.basename(video_path)} because frame rate is also missing. Skipping.")
                return None, None, None, None
            try:
                if '/' in frame_rate_str:
                    num, den = map(int, frame_rate_str.split('/'))
                    frame_rate = num / den if den != 0 else 0
                else:
                    frame_rate = float(frame_rate_str)
            except (ValueError, ZeroDivisionError) as e:
                 st.error(f"Could not parse frame rate ('{frame_rate_str}') for {os.path.basename(video_path)}. Error: {e}. Skipping.")
                 return None, None, None, None
            if frame_rate <= 0:
                 st.error(f"Frame rate is zero or negative for {os.path.basename(video_path)}. Cannot calculate frames. Skipping.")
                 return None, None, None, None
            frames = int(duration * frame_rate)

        return width, height, frames, duration

    # MODIFICATION: Catch the process error and display ffprobe's stderr.
    except subprocess.CalledProcessError as e:
        st.error(f"ffprobe failed for {os.path.basename(video_path)}. This might indicate a corrupt or unsupported file.")
        # Display the actual error output from ffprobe for diagnostics
        st.expander("Show ffprobe error details").code(e.stderr)
        return None, None, None, None
    except (FileNotFoundError, ValueError, IndexError) as e:
        st.error(f"An unexpected error occurred while getting properties for {os.path.basename(video_path)}. Error: {e}")
        return None, None, None, None

def get_frame_at_time(video_path, time_in_seconds=1):
    """Extracts a single frame from a video at a specific time."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame_file:
        frame_path = temp_frame_file.name
        try:
            command = ['ffmpeg', '-ss', str(time_in_seconds), '-i', video_path, '-vframes', '1', '-y', frame_path]
            subprocess.run(command, check=True, capture_output=True, text=True)
            return Image.open(frame_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            st.error(f"ffmpeg error: {e}")
            if isinstance(e, FileNotFoundError): st.stop()
            return None
        finally:
            if os.path.exists(frame_path): os.remove(frame_path)

def draw_preview(frame_image, box1_props, intro_box_props):
    """Draws the specified boxes on a preview image."""
    img_with_box = frame_image.copy()
    draw = ImageDraw.Draw(img_with_box, "RGBA")
    if box1_props['w'] > 0 and box1_props['h'] > 0: draw.rounded_rectangle([box1_props['x'], box1_props['y'], box1_props['x'] + box1_props['w'], box1_props['y'] + box1_props['h']], radius=box1_props['rad'], fill=box1_props['color'])
    if intro_box_props['w'] > 0 and intro_box_props['h'] > 0: draw.rounded_rectangle([intro_box_props['x'], intro_box_props['y'], intro_box_props['x'] + intro_box_props['w'], intro_box_props['y'] + intro_box_props['h']], radius=intro_box_props['rad'], fill=intro_box_props['color'])
    return img_with_box

def create_rounded_rect_image(width, height, radius, color):
    """Creates a temporary PNG image of a rounded rectangle."""
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=color)
    temp_img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp_img_file.name)
    return temp_img_file.name

def start_ffmpeg_process(input_path, output_path, box1_props, intro_box_props, duration, enable_outro_box=False, outro_image_path=None, video_width=1920, video_height=1080):
    """
    Constructs and starts the ffmpeg subprocess.
    MODIFICATION: Now handles an optional uploaded outro image to be overlaid
    on top of the generated white outro box.
    """
    box1_img_path = create_rounded_rect_image(box1_props['w'], box1_props['h'], box1_props['rad'], box1_props['color'])
    intro_box_img_path = create_rounded_rect_image(intro_box_props['w'], intro_box_props['h'], intro_box_props['rad'], intro_box_props['color'])
    temp_imgs_to_clean = [box1_img_path, intro_box_img_path]

    # Base command with 3 inputs: video, box1, intro_box
    command = ['ffmpeg', '-progress', '-', '-nostats', '-i', input_path, '-i', box1_img_path, '-i', intro_box_img_path]

    # If an outro image is provided, add it as the 4th input
    if outro_image_path:
        command.extend(['-i', outro_image_path])
        # The outro image will be stream [3:v]

    final_video_stream = "basevideo"

    # Base filter graph for the first two overlays
    filter_complex = (
        f"[0:v][1:v]overlay=x={box1_props['x']}:y={box1_props['y']}[bg1];"
        f"[bg1][2:v]overlay=x={intro_box_props['x']}:y={intro_box_props['y']}:enable='between(t,0,30)'[{final_video_stream}]"
    )

    # Conditionally add the full-screen white outro box with fade-in.
    if enable_outro_box and duration and duration > 4:
        fade_duration = 1
        fade_start_time = duration - 4

        # The output stream of this stage will be named 'v_with_white_outro'
        stream_after_whitebox = "v_with_white_outro"

        outro_filter_chain = (
            f";color=c=white:s={video_width}x{video_height}:d={duration}[whitebg];"
            f"[whitebg]fade=in:st={fade_start_time}:d={fade_duration}:alpha=1[faded_white];"
            f"[{final_video_stream}][faded_white]overlay=0:0:shortest=1[{stream_after_whitebox}]"
        )
        filter_complex += outro_filter_chain
        final_video_stream = stream_after_whitebox # Update the final stream name

        # NEW LOGIC: If there's also an outro image, overlay it now
        if outro_image_path:
            outro_start_time = duration - 3 # Show for the last 3 seconds
            stream_after_outro_img = "v_final"

            # The outro image is input [3:v]. Overlay it on the stream that has the white box.
            outro_image_filter = (
                f";[{final_video_stream}][3:v]overlay="
                f"x=(W-w)/2:y=(H-h)/2:" # Center the image
                f"enable='between(t,{outro_start_time},{duration})'[{stream_after_outro_img}]"
            )
            filter_complex += outro_image_filter
            final_video_stream = stream_after_outro_img # Update the final stream name again

    # Final command assembly
    command.extend([
        '-filter_complex', filter_complex,
        '-map', f'[{final_video_stream}]',
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        '-y', output_path
    ])

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
    return process, temp_imgs_to_clean


def monitor_process_and_queue_updates(process, file_name, progress_queue):
    """Worker thread: watches ffmpeg, puts progress and final status in queue."""
    frame_pattern = re.compile(r"frame=\s*(\d+)")
    for line in iter(process.stdout.readline, ''):
        match = frame_pattern.search(line)
        if match:
            progress_queue.put({"file_name": file_name, "progress": int(match.group(1))})
    process.wait()
    progress_queue.put({"file_name": file_name, "status": "done", "returncode": process.returncode, "stderr": process.stderr.read()})

# --- Main App ---

st.title('Bulk Video Editor')

# --- Sidebar and UI ---
st.sidebar.header("File Upload")
uploaded_files = st.sidebar.file_uploader("Upload video files", type=['mp4', 'mov', 'avi', 'mkv'], accept_multiple_files=True, key="video_uploader")

image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif', 'webp']
# RE-ENABLED: Outro file uploader
outro_file = st.sidebar.file_uploader("Upload an Outro Image (Optional)", type=image_formats, key="outro_uploader")

st.sidebar.header("Preview Source (Optional)")
preview_file = st.sidebar.file_uploader("Upload an image for the preview", type=image_formats, key="preview_uploader")

if 'frame_image' not in st.session_state: st.session_state.frame_image = None

try:
    if preview_file:
        st.session_state.frame_image = Image.open(preview_file)
    elif uploaded_files and not st.session_state.get('preview_generated_for', []) == [f.name for f in uploaded_files]:
        with st.spinner('Generating preview...'):
            with tempfile.TemporaryDirectory() as td:
                p_path = os.path.join(td, uploaded_files[0].name)
                with open(p_path, 'wb') as f: f.write(uploaded_files[0].getbuffer())
                st.session_state.frame_image = get_frame_at_time(p_path)
        st.session_state.preview_generated_for = [f.name for f in uploaded_files]
except Exception as e:
    st.error(f"Could not generate preview: {e}")
    st.session_state.frame_image = None

max_w, max_h = (st.session_state.frame_image.size if st.session_state.frame_image else (1920, 1080))
if st.session_state.frame_image: st.sidebar.info(f"Sliders max values: {max_w}x{max_h}px.")
else: st.sidebar.info("Upload a file for precise controls.")

b1 = st.sidebar.expander("Box 1 Properties", expanded=False)
b1c, b1x, b1y, b1w, b1h, b1r = b1.color_picker('B1 Color', "#FFFFFF", key="c1"), b1.slider('B1 X', 0, max_w, min(1100, max_w), key="x1"), b1.slider('B1 Y', 0, max_h, min(650, max_h), key="y1"), b1.slider('B1 W', 1, max_w, min(150, max_w), key="w1"), b1.slider('B1 H', 1, max_h, min(30, max_h), key="h1"), b1.slider('B1 R', 0, 100, 10, key="r1")
intro_b = st.sidebar.expander("Intro Box Properties", expanded=False)
intro_bc, intro_bx, intro_by, intro_bw, intro_bh, intro_br = intro_b.color_picker('Intro Box Color', "#FFFFFF", key="c2"), intro_b.slider('Intro Box X', 0, max_w, min(540, max_w), key="x2"), intro_b.slider('Intro Box Y', 0, max_h, min(50, max_h), key="y2"), intro_b.slider('Intro Box W', 1, max_w, min(190, max_w), key="w2"), intro_b.slider('Intro Box H', 1, max_h, min(25, max_h), key="h2"), intro_b.slider('Intro Box R', 0, 100, 25, key="r2")

# MODIFIED: Description clarifies behavior with uploaded image
outro_b = st.sidebar.expander("Outro Box Properties", expanded=True)
enable_outro = outro_b.checkbox("Enable full-screen white outro (fades in for 1s, stays for 3s)", value=True, key="enable_outro")
outro_b.info("If an outro image is uploaded, it will be centered on this white screen for the final 3 seconds.")


box1_props = {'x': b1x, 'y': b1y, 'w': b1w, 'h': b1h, 'rad': b1r, 'color': b1c}
intro_box_props = {'x': intro_bx, 'y': intro_by, 'w': intro_bw, 'h': intro_bh, 'rad': intro_br, 'color': intro_bc}

if not uploaded_files: st.info('Upload video files to begin.')
if st.session_state.frame_image:
    st.subheader("Live Preview")
    st.image(draw_preview(st.session_state.frame_image, box1_props, intro_box_props), use_container_width=True)

if uploaded_files and st.button('Render All Videos'):
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info(f'Starting parallel rendering for {len(uploaded_files)} videos...')
        progress_queue, processes, threads, process_data = queue.Queue(), [], [], []

        total_progress_container = st.container()
        total_bar = total_progress_container.progress(0)
        total_text = total_progress_container.empty()

        progress_table_placeholder = st.empty()

        grand_total_frames = 0
        batch_start_time = time.time()

        # ADDED: Handle the uploaded outro file once before the loop
        outro_image_path = None
        if outro_file:
            outro_image_path = os.path.join(temp_dir, "user_outro_image" + os.path.splitext(outro_file.name)[1])
            with open(outro_image_path, 'wb') as f:
                f.write(outro_file.getbuffer())
            st.info(f"Using uploaded outro image: {outro_file.name}")


        # 1. Setup UI and start all background jobs
        for idx, uploaded_file in enumerate(uploaded_files):
            input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_path, 'wb') as f: f.write(uploaded_file.getbuffer())

            width, height, total_frames, duration = get_video_properties(input_path)
            if not all([width, height, total_frames is not None, duration is not None]):
                # Error is already displayed inside the function
                continue

            grand_total_frames += total_frames

            process_data.append({
                '#': idx + 1, 'File Name': uploaded_file.name, 'ETR': 'N/A',
                'Progress': 0.0, 'Status': 'Queued', 'total_frames': total_frames,
                'current_frames': 0, 'start_time': 0, 'final_status': None, 'stderr': ''
            })

            output_path = os.path.join(temp_dir, f"{os.path.splitext(uploaded_file.name)[0]}_edited{os.path.splitext(uploaded_file.name)[1]}")
            # MODIFIED: Call start_ffmpeg_process with new parameters including the outro image path
            process, temp_imgs = start_ffmpeg_process(
                input_path,
                output_path,
                box1_props,
                intro_box_props,
                duration,
                enable_outro_box=enable_outro,
                outro_image_path=outro_image_path, # Pass the path here
                video_width=width,
                video_height=height
            )


            processes.append({'process': process, 'file_name': uploaded_file.name, 'output_path': output_path, 'temp_imgs': temp_imgs})

            thread = threading.Thread(target=monitor_process_and_queue_updates, args=(process, uploaded_file.name, progress_queue))
            threads.append(thread)
            thread.start()

        # 2. Main thread listens to the queue and updates the UI
        active_threads = len(threads)
        while active_threads > 0:
            try:
                update = progress_queue.get(timeout=0.2)
                file_name = update['file_name']

                p_data = next((p for p in process_data if p['File Name'] == file_name), None)
                if not p_data: continue

                if "status" in update and update["status"] == "done":
                    active_threads -= 1
                    p_data['final_status'], p_data['stderr'] = update['returncode'], update['stderr']
                    if update['returncode'] == 0:
                        p_data['Status'], p_data['Progress'], p_data['current_frames'], p_data['ETR'] = "✅ Success", 1.0, p_data['total_frames'], "Done"
                    else:
                        p_data['Status'], p_data['ETR'] = "❌ Failed", "Failed"

                elif "progress" in update:
                    if p_data['start_time'] == 0: p_data['start_time'] = time.time()
                    progress_val = update['progress']
                    p_data['current_frames'] = progress_val
                    total_frames = p_data['total_frames']
                    p_data['Progress'] = min(1.0, progress_val / total_frames) if total_frames > 0 else 0

                    elapsed_time = time.time() - p_data['start_time']
                    etr_str = "Calculating..."
                    if progress_val > 5:
                        speed = elapsed_time / progress_val
                        etr_seconds = speed * (total_frames - progress_val)
                        etr_str = time.strftime('%H:%M:%S', time.gmtime(etr_seconds))

                    p_data['ETR'], p_data['Status'] = etr_str, "Encoding..."

                total_frames_processed = sum(p['current_frames'] for p in process_data)
                total_percentage = min(1.0, total_frames_processed / grand_total_frames) if grand_total_frames > 0 else 0
                total_bar.progress(total_percentage)

                total_etr_str = "Calculating..."
                if total_frames_processed > 10:
                    total_elapsed_time = time.time() - batch_start_time
                    speed = total_elapsed_time / total_frames_processed
                    total_etr_seconds = speed * (grand_total_frames - total_frames_processed)
                    total_etr_str = time.strftime('%H:%M:%S', time.gmtime(total_etr_seconds))

                total_text.text(f"Overall Progress: {total_frames_processed}/{grand_total_frames} frames ({total_percentage:.1%}) | Total Estimated Time Remaining: {total_etr_str}")

                df = pd.DataFrame(process_data)[['#', 'File Name', 'ETR', 'Progress', 'Status']]
                progress_table_placeholder.dataframe(
                    df,
                    column_config={"Progress": st.column_config.ProgressColumn("Progress", format="%.2f", min_value=0, max_value=1)},
                    hide_index=True, use_container_width=True
                )

            except queue.Empty:
                continue

        # 3. Finalize results
        total_text.info("All rendering tasks complete. Finalizing...")
        success_count, successful_paths = 0, []

        for p_data in process_data:
            p_info = next((p for p in processes if p['file_name'] == p_data['File Name']), None)
            if p_info and p_data['final_status'] == 0:
                success_count += 1
                successful_paths.append(p_info['output_path'])
            elif p_info:
                st.error(f"Failed: {p_data['File Name']} (code: {p_data['final_status']})")
                if p_data['stderr']: st.expander(f"Show ffmpeg error for {p_data['File Name']}").code(p_data['stderr'])

            if p_info:
                for path in p_info['temp_imgs']:
                    if os.path.exists(path): os.remove(path)

        if success_count > 0:
            st.balloons()
            st.success(f'Finished! {success_count}/{len(uploaded_files)} successful.')
            zip_path = os.path.join(temp_dir, "edited_videos.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in successful_paths: zipf.write(file_path, os.path.basename(file_path))
            with open(zip_path, "rb") as f:
                st.download_button("Download All as ZIP", f, "edited_videos.zip", "application/zip")
            st.subheader("Edited Video Previews:")
            for file_path in successful_paths:
                with open(file_path, 'rb') as f: st.video(f.read())
                with open(file_path, 'rb') as f: st.download_button(f"Download {os.path.basename(file_path)}", f, os.path.basename(file_path))
