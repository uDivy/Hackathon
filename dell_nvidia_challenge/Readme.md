# [Copharmacists AI](https://youtu.be/PxHCiFPKNX8)
# Prescription Transcription and Structured Data Extraction System and Avatar Generation

This project presents a web-based system for transcribing prescription images, extracting structured data, and generating digital avatars. Leveraging advanced AI models for image preprocessing, optical character recognition (OCR), and natural language understanding. The project utilizes the Qwen2-VL model for accurate text transcription and Google's Gemini API for structured data extraction, particularly for medical prescriptions. Additionally, the SadTalker model and Google Text-to-Speech (gTTS) create dynamic, talking avatars based on the extracted data. The user-friendly interface, built with Streamlit, enables seamless interaction, making the process of digitizing prescriptions both efficient and accessible.

By integrating AI-driven solutions for OCR, data extraction, and avatar generation into a single platform, this project offers an efficient method for digitizing and interacting with prescription data, making it particularly valuable for healthcare applications, patient education, and medical data management.

## From prescription ---> Talking Avatars (Let the image do the talking!!!)

<div style="display: flex;">
  <img src="https://github.com/uDivy/Hackathon/blob/development/dell_nvidia_challenge/sample//1.jpg?raw=true" alt="Sample Image 1" width="300"/>
</div>

## Video Demo
<p>
  <a href="https://github.com/uDivy/Hackathon/blob/development/dell_nvidia_challenge/sample/final_demo.mp4">Watch the demo video</a>
</p>



## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Key Components](#key-components)
- [Setup and Installation](#setup-and-installation)
  - [Local Setup](#local-setup)
    - [Phase I](#phase-i)
    - [Phase II](#phase-ii)
- [Running the Project in Docker](#running-the-project-in-docker)
- [Usage](#usage)
- [Target Systems](#target-systems)
- [Restrictions](#restrictions)
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

- Python 3.10.14
- PyTorch with Cuda enabled
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
5. **Web Interface**: Provides a simple, intuitive interface for users to upload prescription images and view extracted data and generates video.

## Setup and Installation

### Local Setup [Windows 11+]

#### Phase I

1. **Clone the repository**
2. **Install dependencies**:
       ```
       pip install -r requirements.txt # pip install -r requirementslinuxdocker.txt for Linux distros
       pip install git+https://github.com/huggingface/transformers
       pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
       ```
3. **Set up environment variables:**
   - Create `.env` file and add the following:
   - `GEMINI_API_KEY`: Your Google Gemini API key

#### Phase II

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
  ├── .env
  └── README.md
  ```

Set up SadTalker:
1. **Change to the `src` directory**
2. **Clone the SadTalker repository**:
   ```
   git clone https://github.com/OpenTalker/SadTalker.git
   cd SadTalker
   ```
3. **Download the pre-trained model**:
   - For Windows users, modify the download script as specified:
   - Edit the file with this:
        #!/bin/bash

        mkdir -p ../../checkpoints
        
        curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -o ../../checkpoints/mapping_00109-model.pth.tar \
        curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -o ../../checkpoints/mapping_00229-model.pth.tar \
        curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -o ../../checkpoints/SadTalker_V0.0.2_256.safetensors \
        curl -L https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors -o ../../checkpoints/SadTalker_V0.0.2_512.safetensors
        
        mkdir -p ../../gfpgan/weights
        
        curl -L https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth -o ../../gfpgan/weights/alignment_WFLW_4HG.pth \
        curl -L https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -o ../../gfpgan/weights/detection_Resnet50_Final.pth \
        curl -L https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -o ../../gfpgan/weights/GFPGANv1.4.pth \
        curl -L https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -o ../../gfpgan/weights/parsing_parsenet.pth


   - Run the bash script: `bash download_models.sh` on git bash.

4. **Install ffmpeg on Windows**:

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

5. **Ensure you have a default avatar image and python in command for speect_avatar.py**:

   Place a front-facing image named `doctor1.jpg` in your project directory under `src`.
   If you are using Linux distibution python cmd in `speech_avatar.py` will change to `<"python3.10", os.path.join(current_dir, "src", "SadTalker", "inference.py"),>`, OR you could use the `dockerized` branch.

6. **This error will occur due torch version mismatch between Qwen2VL and SadTalker**:
   
   Based on the error message, it seems that the issue is related to an outdated version of `torchvision` or a mismatch between `torch` and `torchvision` versions. Since you've requested a solution that doesn't involve uninstalling or installing dependencies, we can try to work around this issue by modifying the code that's causing the error.

   Here's a potential solution:

     1. **Locate the file `basicsr/data/degradations.py` in your environment.**
     2. **Open this file in a text editor.**
     3. **Find the line that says:**
  
           ```python
           from torchvision.transforms.functional_tensor import rgb_to_grayscale
  
      4. **Replace this line with:**
  
           ```python
           from torchvision.transforms.functional import rgb_to_grayscale
           ```
  
     This change reflects an update in the `torchvision` library where `functional_tensor` was merged into `functional`.
   
## Running the Project in Docker

To run this project using Docker Desktop or Docker Daemon, follow the steps below:

### Using the Prebuilt Docker Image from Docker Hub:
   ```bash
     docker pull divya291/nvdia_ai_wb_python_31014_cuda12:latest
    ```
           

    To run the container, use the following command:
   
   ```bash
   docker run -it --gpus all -p 8501:8501 -p 8888:8888 -e GEMINI_API_KEY=YOUR_API_KEY --name my_nvidia_container -w /workspace divya291/nvdia_ai_wb_python_31014_cuda12:latest /bin/bash 

   streamlit run ui.py 

   Access the app by navigating to http://localhost:8501 in your browser.
   ```

    This will make the web interface available at http://localhost:8501.

1. **Build the Docker Image**:
   Ensure that you are in the root directory of the project, where the Dockerfile is 
   located.  
   
   If you are using Linux distibution python cmd in `speech_avatar.py` will change to `<"python3.10", os.path.join(current_dir, "src", "SadTalker", "inference.py"),>`, OR you could use the `dockerized` branch.
   
   Run the following command to build the image:

     ```bash
     docker build -f cuda_python_dockerfile -t <your-image-name>:latest .

   This will build the Docker image and tag it with <your-image-name>:latest.

2. **Test the Image Locally**:
   Run the image locally to make sure everything works as expected. Use the following 
   command:

    ```bash
    docker run -it --gpus all -p 8501:8501 -p 8888:8888 -e GEMINI_API_KEY=YOUR_API_KEY --name <container-name> -w /workspace <your-dockerhub-username>/<your-image-name>:latest /bin/bash 

    streamlit run ui.py 

    Access the app by navigating to http://localhost:8501 in your browser.
    ```

3. **Push the Image to a Docker Registry:**:
   First, log in to the Docker registry you want to use (Docker Hub or NVIDIA NGC):

    **For Docker Hub**:
      
     ```bash
       docker login
     ```
         
      After logging in, push the image to Docker Hub: 
    
     ```bash
     docker tag <your-image-name>:latest <your-dockerhub-username>/<your-image-name>:latest      
     docker push <your-dockerhub-username>/<your-image-name>:latest
     ```
           
    
    **For NVIDIA NGC**:
      
      Log in to the NVIDIA NGC registry:
      
      ```bash
      docker login nvcr.io
      ```
        
    
      Then, push the image to the NVIDIA NGC registry:
        
      ```bash
      docker tag <your-image-name>:latest nvcr.io/<your-ngc-username>/<your-image-name>:latest
      docker push nvcr.io/<your-ngc-username>/<your-image-name>:latest
      ```
    
4. **Run the Image from Docker Registry**:
   
   Once the image is pushed, others can pull it from the registry and run it using:
   
    ```bash
     docker pull <your-dockerhub-username>/<your-image-name>:latest
    ```
           

    Then, run the container with:
   
   ```bash
   docker run -it --gpus all -p 8501:8501 -p 8888:8888 -e GEMINI_API_KEY=YOUR_API_KEY --name <container-name> -w /workspace <your-dockerhub-username>/<your-image-name>:latest /bin/bash 

   streamlit run ui.py 

    Access the app by navigating to http://localhost:8501 in your browser.
   ```

    This will expose the web interface at http://localhost:8501.
  
This setup allows you to push the prebuilt Docker image to a registry so you can use it without needing to rebuild the entire environment.


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

## Target Systems
This project has been designed to work in the **NVIDIA AI Workbench** environment. It requires the following setup:

- **CUDA** support for models that leverage GPU acceleration (NVIDIA GPUs).
- **Docker** and **NVIDIA NGC** for container management.
- **Python 3.10** environment.

Follow the instructions in the [Readme.md](https://github.com/uDivy/Hackathon/tree/development/dell_nvidia_challenge) from the development branch to set up and run the application locally (on Docker/Windows). \
You can use the provided sample image (uploaded in the sample directory) or your own image through the web app, accessible at http://localhost:8501/. \
Since the application leverages deep learning models, it requires a GPU with at least 6GB of memory and a CPU with at least 16GB of RAM. \
The application has been tested on a Windows machine with 16GB of CPU RAM and a 4GB Nvidia MX130 GPU ([specs](https://www.notebookcheck.net/NVIDIA-GeForce-MX130-GPU-Benchmarks-and-Specs.258054.0.html)). \
Performance is slower on older GPUs like the MX130, but you can expect significant improvements with newer, more powerful GPUs. \

This project should work on both **Linux** and **Windows** environments where Docker or native installations are supported.

## Restrictions
- The project uses the **Qwen2-VL** and **SadTalker** models, which require significant computational power. While they are GPU-accelerated, a **CUDA-compatible GPU** is recommended for optimal performance.
- The **SadTalker** model may have dependencies and limitations based on the **PyTorch** version being used. Ensure compatibility between **PyTorch** and **Torchvision** versions to avoid model loading errors.

## Future work

This project serves as a prototype for advanced medical solutions. The scope of this project can be expanded by supporting to a broader range of medical documents beyond prescriptions, such as patient reports, lab results, and diagnostic images.  By integrating advanced Natural Language Processing (NLP) models, such as BERT or GPT, we could improve the understanding and extraction of more complex medical terms and relationships from the transcribed text. Enhancing the structured data extraction by linking it to medical knowledge graphs or databases, could provide richer context and more precise drug or dosage information. Additionally, integrating multi-language support would broaden the system's applicability to non-English-speaking regions. 

The ability of the project in transcribing images, extracting structured data, and generate talking avatars can be adapted for various other fields, such as marketing and advertising. For example, it could be used to create personalized, dynamic ads where product images are converted into interactive digital avatars. This will improve customer engagement and can help in branding of the companies. Expanding avatar capabilities to support personalized ad content could offer significant value in industries such as e-commerce and media. 


## Disclaimer

This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for interpretation of prescription details and medical guidance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Creative Commons Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND)]
