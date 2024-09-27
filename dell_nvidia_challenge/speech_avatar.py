import os
import tempfile
import subprocess
from gtts import gTTS
import torch

def text_to_speech(text, output_path):
    """Convert text to speech using gTTS."""
    tts = gTTS(text, lang='en', tld='com.au')  # Using Australian English for a deeper voice
    tts.save(output_path)
    print(f"Audio saved to {output_path}")

def create_avatar_video(image_path, audio_path, output_path):
    """Create a lip-synced avatar video using SadTalker's inference script."""
    print(f"Creating avatar video with image: {image_path} and audio: {audio_path}")
    
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.abspath(image_path)
    audio_path = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)
    print(output_path)
    # Construct the command to run inference.py
    command = [
        "python", os.path.join(current_dir, "src", "SadTalker", "inference.py"),
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_path,
        "--still",
        "--preprocess", "crop",  # Changed from "full" to "crop"
        "--checkpoint_dir", os.path.join(current_dir, "checkpoints"),
        "--batch_size", "1",
        "--size", "256", # Added to reduce output size if not already specified
        # "--" + ("cpu" if not torch.cuda.is_available() else "gpu")
        "--cpu"
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env["GFPGAN_DIR"] = os.path.join(current_dir, "gfpgan")
    env["PYTHONPATH"] = os.path.join(current_dir, "src", "SadTalker") + os.pathsep + env.get("PYTHONPATH", "")
    # Remove the following line:
    # env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Run the command
    try:
        subprocess.run(command, check=True, env=env)
        print("Avatar video created successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error creating avatar video: {e}")

def process_transcription_to_avatar(transcription, avatar_image_path, output_video_path):
    """Process transcription to create an avatar video."""
    print(f"Processing transcription: {transcription}")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        # Generate audio from transcription
        text_to_speech(transcription, temp_audio.name)
        
        # Create avatar video
        create_avatar_video(avatar_image_path, temp_audio.name, output_video_path)
    
    # Clean up temporary audio file
    os.unlink(temp_audio.name)
    print("Process completed successfully")

# Example usage
if __name__ == "__main__":
    transcription = "This"
    avatar_image_path = "src/doctor1.jpeg"
    output_video_path = "output/"
    
    process_transcription_to_avatar(transcription, avatar_image_path, output_video_path)
