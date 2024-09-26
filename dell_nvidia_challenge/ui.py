import streamlit as st
import gc
import torch
import os
from read_transcribe import process_prescription
from speech_avatar import process_transcription_to_avatar
import asyncio
import glob

# At the beginning of your script, initialize the session state if it doesn't exist
if 'prescription_result' not in st.session_state:
    st.session_state.prescription_result = None

def free_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

async def process_prescription_ui():
    st.subheader("Prescription Processing")
    uploaded_file = st.file_uploader("Choose a prescription image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Prescription", use_column_width=True)

        if st.button("Process Prescription"):
            with st.spinner("Processing..."):
                result = await process_prescription(uploaded_file)
                st.session_state.prescription_result = result  # Store the result

    # Display the result, whether it's from a new upload or from the session state
    if st.session_state.prescription_result:
        result = st.session_state.prescription_result
        st.subheader("Processed Information:")
        if 'medications' in result:
            for medication in result['medications']:
                st.write(medication)
        elif 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            st.write("No processed information available.")

        if 'disclaimer' in result:
            st.warning(result['disclaimer'])

def get_latest_video(output_dir):
    list_of_files = glob.glob(os.path.join(output_dir, '*.mp4'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def generate_avatar_video():
    st.subheader("Avatar Video Generation")
    
    if st.session_state.prescription_result is None:
        st.warning("Please process a prescription first.")
        return

    result = st.session_state.prescription_result

    if st.button("Generate Avatar Video"):
        st.warning("Creating avatar video may take some time and require significant computational resources.")
        with st.spinner("Generating Avatar Video..."):
            free_memory()
            
            # Define the base directory (where your code is)
            base_dir = os.path.dirname(os.path.abspath(__file__))

            # Define the output directory relative to the base directory
            output_dir = os.path.join(base_dir, "output")
            image_loc = os.path.join(base_dir, "/src/doctor1.jpeg")

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            avatar_image_path = image_loc
            
            if not os.path.exists(avatar_image_path):
                st.error(f"Avatar image not found at {avatar_image_path}")
                return

            try:
                medications_text = ""
                for medication in result.get('medications', []):
                    medications_text += f"Medication: {medication['drug_name']}\n"
                    # ... other medication details ...
                
                process_transcription_to_avatar(medications_text, avatar_image_path, output_dir)
                
                # Get the path of the latest created video
                latest_video_path = get_latest_video(output_dir)
                
                if latest_video_path and os.path.exists(latest_video_path):
                    st.success(f"Video created successfully at {latest_video_path}")
                    st.video(latest_video_path)
                else:
                    st.error("No video file found in the output directory.")
            except Exception as e:
                st.error(f"Error during video creation: {str(e)}")
                st.error("The avatar video could not be created. Please check the logs for more information.")

async def main():
    st.title("Prescription Processing App")

    col1, col2 = st.columns(2)

    with col1:
        await process_prescription_ui()

    with col2:
        generate_avatar_video()

if __name__ == "__main__":
    asyncio.run(main())
