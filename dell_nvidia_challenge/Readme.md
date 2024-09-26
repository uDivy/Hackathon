# Prescription Transcription and Structured Data Extraction System and Avatar Generation

This project provides a web-based interface for transcribing prescription images and extracting structured data from them and create digital avatars. It utilizes advanced AI models for image processing, optical character recognition (OCR), and natural language understanding and creates avatars.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Key Components](#key-components)
- [Setup and Installation](#setup-and-installation)
  - [Phase I](#phase-i)
  - [Phase II](#phase-ii)
- [Usage](#usage)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)

## Features

- Image preprocessing to optimize for OCR
- Prescription transcription using the Qwen2-VL model
- Structured data extraction using Google's Gemini API
- SadTalker to create talking digital avatars
- Text to Audio using gTTS
- User-friendly web interface built with Streamlit

## Technology Stack

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- Google Generative AI
- SadTalker
- Google Text to Speech
- Streamlit

## Key Components

1. **Image Preprocessing**: Converts images to grayscale and resizes them for optimal OCR performance.
2. **OCR with Qwen2-VL**: Uses the Qwen2-VL-2B-Instruct model to transcribe text from prescription images.
3. **Structured Data Extraction**: Employs Google's Gemini API to extract structured information about medications from the transcribed text.
4. **Digital Avatar Generation**: Used locally installed SadTalker model and google Text-To-Speech to generate Talking digital avatars.
5. **Web Interface**: Provides a simple, intuitive interface for users to upload prescription images and view extracted data.

## Setup and Installation

### Phase I

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install git+https://github.com/huggingface/transformers
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Set up environment variables:
   - Create `.env` file and add the following:
     - `GEMINI_API_KEY`: Your Google Gemini API key

### Phase II

Final directory structure:
```
project_root/
│
├── ui.py
├── speech_avatar.py
├── read_transcribe.py
│
├── src/
│   ├── SadTalker/
│   │   └── ... (SadTalker contents)
│   │
│   └── doctor1.jpg
│
├── output/
│   └── ... (generated videos)
│
├── checkpoints/
│   └── ... (model checkpoints)
│
├── gfpgan/
│   └── ... (GFPGAN contents)
│
└── README.md
```

Set up SadTalker:
1. Change to the `src` directory.
2. Clone the SadTalker repository:
   ```
   git clone https://github.com/OpenTalker/SadTalker.git
   cd SadTalker
   ```
3. Download the pre-trained model:
   - For Windows users, modify the download script as specified:
   - Edit the file with this:
      #!/bin/bash

      mkdir -p ../../checkpoints

      curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -o ../../checkpoints/mapping_00109-model.pth.tar \
      curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -o ../../checkpoints/mapping_00229-model.pth.tar \
      curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -o ../../checkpoints/SadTalker_V0.0.2_256.safetensors \
      curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors -o ../../checkpoints/SadTalker_V0.0.2_512.safetensors 

      # Create gfpgan/weights directory
      mkdir -p ../../gfpgan/weights

      curl -L https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth -o ../../gfpgan/weights/alignment_WFLW_4HG.pth \
      curl -L https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -o ../../gfpgan/weights/detection_Resnet50_Final.pth \
      curl -L https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -o ../../gfpgan/weights/GFPGANv1.4.pth \
      curl -L https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -o ../../gfpgan/weights/parsing_parsenet.pth 

   - Run the bash script: `bash download_models.sh` on git bash.

4. Install ffmpeg on Windows:

   a. Go to the ffmpeg download page: https://ffmpeg.org/download.html \
   b. Under the "Get packages & executable files" section, click on the Windows logo. \
   c. On the next page, click on "Windows builds by BtbN" link. \
   d. Scroll down and find the latest release. Look for a file named like "ffmpeg-master-latest-win64-gpl.zip". \
   e. Download this zip file. \
   f. Extract the contents of the zip file to a location on your computer, for example: "C:\ffmpeg" \
   g. Now we need to add ffmpeg to your system PATH:
      - Press Win + X and select "System"
      - Click on "Advanced system settings" on the right
      - Click on "Environment Variables" at the bottom
      - Under "System variables", find the "Path" variable and click "Edit"
      - Click "New" and add the path to the ffmpeg "bin" folder (e.g., "C:\ffmpeg\bin")
      - Click "OK" on all windows to save the changes

   h. Verify ffmpeg installation:
   - Open a new Command Prompt window
   - Type `ffmpeg -version` and press Enter
   - If you see version information, ffmpeg is successfully installed

5. Ensure you have a default avatar image:

   Place a front-facing image named `doctor1.jpg` in your project directory under `src`.

6. This error will occur due torch version mismatch between Qwen2VL and SadTalker: 
   Based on the error message, it seems that the issue is related to an outdated version of `torchvision` or a mismatch between `torch` and `torchvision` versions. Since you've requested a solution that doesn't involve uninstalling or installing dependencies, we can try to work around this issue by modifying the code that's causing the error.

   Here's a potential solution:

   **1. Locate the file `basicsr/data/degradations.py` in your environment.**
   2. Open this file in a text editor.
   3. Find the line that says:

   ```python
   from torchvision.transforms.functional_tensor import rgb_to_grayscale
   ```

   4. Replace this line with:

   ```python
   from torchvision.transforms.functional import rgb_to_grayscale
   ```

   This change reflects an update in the `torchvision` library where `functional_tensor` was merged into `functional`.

## Usage

Run the Streamlit app:
```
streamlit run ui.py
```

Then, follow these steps:
1. Upload a prescription image
2. Click "Transcribe Prescription"
3. View the extracted medication information and disclaimer
4. Once the step ran successfully, click `Generate Avatar Video`.

## Disclaimer

This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for interpretation of prescription details and medical guidance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]