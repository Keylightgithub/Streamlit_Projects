'''
streamlit run /Users/isaachenry/Documents/VS_Code_intern_projects/Bulk_Video_Editor/main.py
'''

import streamlit as st
import os
import subprocess
import tempfile
import zipfile
from PIL import Image, ImageDraw

def get_video_duration(video_path):
    """Gets the duration of a video file in seconds."""
    try:
        command = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(process.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error getting duration for {os.path.basename(video_path)}: {e.stderr}")
        return None
    except FileNotFoundError:
        # This error is better handled once globally
        return None
    except ValueError:
        st.error(f"Could not parse duration for {os.path.basename(video_path)}.")
        return None

def get_frame_at_time(video_path, time_in_seconds=1):
    """Extracts a single frame from a video file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame_file:
        frame_path = temp_frame_file.name
        try:
            command = [
                'ffmpeg', '-ss', str(time_in_seconds), '-i', video_path,
                '-vframes', '1', '-y', frame_path # Added -y to overwrite temp file without asking
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            return Image.open(frame_path)
        except subprocess.CalledProcessError as e:
            st.error(f"Error extracting frame: {e.stderr}")
            return None
        except FileNotFoundError:
            st.error("ffmpeg not found. Please make sure it's installed and in your PATH.")
            return None
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)

def draw_preview(frame_image, box1_props, box2_props):
    """Draws two rounded rectangles on a copy of the image for preview."""
    img_with_box = frame_image.copy()
    draw = ImageDraw.Draw(img_with_box, "RGBA")

    # Draw Box 1
    b1 = box1_props
    if b1['w'] > 0 and b1['h'] > 0:
        draw.rounded_rectangle(
            [b1['x'], b1['y'], b1['x'] + b1['w'], b1['y'] + b1['h']],
            radius=b1['rad'], fill=b1['color']
        )

    # Draw Box 2
    b2 = box2_props
    if b2['w'] > 0 and b2['h'] > 0:
        draw.rounded_rectangle(
            [b2['x'], b2['y'], b2['x'] + b2['w'], b2['y'] + b2['h']],
            radius=b2['rad'], fill=b2['color']
        )
    
    return img_with_box

def create_rounded_rect_image(width, height, radius, color):
    """Creates a transparent PNG image of a rounded rectangle."""
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=color)
    
    temp_img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp_img_file.name)
    return temp_img_file.name

def edit_video(input_path, output_path, box1_props, box2_props):
    """
    Adds two rounded boxes and a fade-out effect to a video using ffmpeg.
    Corrected for QuickTime compatibility.
    """
    box1_img_path = create_rounded_rect_image(box1_props['w'], box1_props['h'], box1_props['rad'], box1_props['color'])
    box2_img_path = create_rounded_rect_image(box2_props['w'], box2_props['h'], box2_props['rad'], box2_props['color'])

    duration = get_video_duration(input_path)
    fade_filter_str = ""
    if duration:
        if duration > 3:
            fade_start_time = duration - 3
            fade_filter_str = f",fade=t=out:st={fade_start_time}:d=1:color=white"
        else:
            st.warning(f"'{os.path.basename(input_path)}' is shorter than 3 seconds. Skipping fade-out effect.")

    try:
        filter_complex = (
            f"[0:v][1:v]overlay=x={box1_props['x']}:y={box1_props['y']}[bg1];"
            f"[bg1][2:v]overlay=x={box2_props['x']}:y={box2_props['y']}"
            f"{fade_filter_str}"
        )

        command = [
            'ffmpeg', '-i', input_path, '-i', box1_img_path, '-i', box2_img_path,
            '-filter_complex', filter_complex,
            '-c:v', 'libx264',           # Set video codec for compatibility
            '-pix_fmt', 'yuv420p',       # Set pixel format for compatibility
            '-c:a', 'copy',              # Copy audio stream from source
            '-y', output_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error editing {os.path.basename(input_path)}: {e.stderr}")
        return False
    except FileNotFoundError:
        st.error("ffmpeg not found. Please make sure it's installed and in your PATH.")
        return False
    finally:
        os.remove(box1_img_path)
        os.remove(box2_img_path)

st.title('Bulk Video Editor')

# --- Sidebar: File Uploaders ---
st.sidebar.header("File Upload")
uploaded_files = st.file_uploader("Upload your video files", type=['mp4', 'mov', 'avi', 'mkv'], accept_multiple_files=True, key="video_uploader")

st.sidebar.header("Preview Source (Optional)")
st.sidebar.caption("Upload an image or use the first video for the preview.")
preview_file = st.file_uploader("Upload an image for the preview", type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'], key="preview_uploader")

# --- Determine Preview Source and Generate Image ---
frame_image = None
# Prioritize the dedicated image uploader for the preview
preview_source = preview_file if preview_file else uploaded_files[0] if uploaded_files else None

if preview_source:
    if preview_source.type.startswith('image/'):
        frame_image = Image.open(preview_source)
    elif preview_source.type.startswith('video/'):
        # This is the slow part, so we show a spinner
        with st.spinner('Generating preview from video...'):
            with tempfile.TemporaryDirectory() as temp_dir:
                preview_path = os.path.join(temp_dir, preview_source.name)
                with open(preview_path, 'wb') as f:
                    f.write(preview_source.getbuffer())
                frame_image = get_frame_at_time(preview_path)
else:
    # This block runs when no files are uploaded yet
    pass


# --- Sidebar: Box Property Controls ---
if frame_image:
    max_w, max_h = frame_image.size
    st.sidebar.info(f"Sliders max values set to preview dimensions: {max_w}x{max_h}px.")
else:
    max_w, max_h = 1920, 1080
    st.sidebar.info("Upload a file to adjust sliders to its dimensions. (Default max: 1920x1080)")

# --- Box 1 ---
st.sidebar.subheader('Box 1 Properties')
color1 = st.sidebar.color_picker('Box 1 Color', "#FFFFFF", key="color1")
x_pos1 = st.sidebar.slider('X 1', 0, max_w, min(1100, max_w), key="x1")
y_pos1 = st.sidebar.slider('Y 1', 0, max_h, min(650, max_h), key="y1")
width1 = st.sidebar.slider('Width 1', 1, max_w, min(150, max_w), key="w1")
height1 = st.sidebar.slider('Height 1', 1, max_h, min(30, max_h), key="h1")
radius1 = st.sidebar.slider('Corner Radius 1', 0, 100, 10, key="rad1")

# --- Box 2 ---
st.sidebar.subheader('Box 2 Properties')
color2 = st.sidebar.color_picker('Box 2 Color', "#FFFFFF", key="color2")
x_pos2 = st.sidebar.slider('X 2', 0, max_w, min(540, max_w), key="x2")
y_pos2 = st.sidebar.slider('Y 2', 0, max_h, min(50, max_h), key="y2")
width2 = st.sidebar.slider('Width 2', 1, max_w, min(190, max_w), key="w2")
height2 = st.sidebar.slider('Height 2', 1, max_h, min(25, max_h), key="h2")
radius2 = st.sidebar.slider('Corner Radius 2', 0, 100, 25, key="rad2")

# --- Main Panel ---
box1_props = {'x': x_pos1, 'y': y_pos1, 'w': width1, 'h': height1, 'rad': radius1, 'color': color1}
box2_props = {'x': x_pos2, 'y': y_pos2, 'w': width2, 'h': height2, 'rad': radius2, 'color': color2}

# --- Logic for Display ---
if not uploaded_files:
    st.info('Upload video files to begin.')

# Show the preview if a frame_image has been generated
if frame_image:
    st.subheader("Live Preview")
    preview_image = draw_preview(frame_image, box1_props, box2_props)
    st.image(preview_image, use_container_width=True)
elif uploaded_files and not preview_file:
    # This message shows while the video preview is being generated
    st.info("Generating preview from the first video...")

# Always show the Render button and its logic if videos are uploaded
if uploaded_files:
    if st.button('Render Videos'):
        with tempfile.TemporaryDirectory() as temp_dir:
            st.info(f'Found {len(uploaded_files)} video files. Starting the rendering process...')
            progress_bar = st.progress(0)
            success_count = 0
            edited_files_paths = []

            for i, uploaded_file in enumerate(uploaded_files):
                input_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                file_name, file_ext = os.path.splitext(uploaded_file.name)
                output_filename = f"{file_name}_edited{file_ext}"
                output_path = os.path.join(temp_dir, output_filename)

                with st.spinner(f'Rendering {uploaded_file.name}...'):
                    if edit_video(input_path, output_path, box1_props, box2_props):
                        st.success(f'Successfully edited {uploaded_file.name} and saved as {output_filename}')
                        success_count += 1
                        edited_files_paths.append(output_path)
                progress_bar.progress((i + 1) / len(uploaded_files))

            if success_count > 0:
                st.balloons()
                st.success(f'Finished rendering. {success_count}/{len(uploaded_files)} successful.')
                
                # Create ZIP file for download
                zip_path = os.path.join(temp_dir, "edited_videos.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_path in edited_files_paths:
                        zipf.write(file_path, os.path.basename(file_path))
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="Download All as ZIP",
                        data=f,
                        file_name="edited_videos.zip",
                        mime="application/zip"
                    )
                
                # Display individual videos and download buttons
                st.subheader("Edited Video Previews:")
                for file_path in edited_files_paths:
                    try:
                        with open(file_path, 'rb') as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.download_button(
                            f"Download {os.path.basename(file_path)}", 
                            video_bytes, 
                            os.path.basename(file_path)
                        )
                    except FileNotFoundError:
                        st.error(f"Could not find {os.path.basename(file_path)} to display.")