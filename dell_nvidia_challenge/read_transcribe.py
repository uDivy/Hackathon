import os
import asyncio
import json
import logging
from typing import List, Dict, Any
from PIL import Image
import io
from dotenv import load_dotenv
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import google.generativeai as genai
import re
from transformers import BitsAndBytesConfig
from speech_avatar import process_transcription_to_avatar


"""
Prescription Transcription and Structured Data Extraction System

This script processes prescription images to transcribe their contents and extract structured data.
It uses a combination of the Qwen2-VL model for image transcription and Google's Gemini API for
structured data extraction.

Key components:
1. Image preprocessing
2. OCR using Qwen2-VL model
3. Structured data extraction using Gemini API
4. Concurrent processing of multiple prescriptions

Dependencies:
- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- python-dotenv
- google-generativeai

Environment variables required:
- GEMINI_API_KEY: API key for Google's Gemini API
"""

# Load environment variables
load_dotenv()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Confirm if the model is 8-bit quantized
def is_8bit_quantized(model):
    return any(param.dtype == torch.int8 for param in model.parameters())


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Qwen2-VL model and tokenizer
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto",
    offload_folder="offload",
    attn_implementation="sdpa",
    quantization_config=quantization_config,
    offload_state_dict=True
)
# print(f"Is model 8-bit quantized: {is_8bit_quantized(model)}")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Get the API key from environment variables
api_key = os.environ.get("GEMINI_API_KEY", "")

# If the API key is empty, try to read it from os.getenv
if api_key == "":
    api_key = os.getenv('GEMINI_API_KEY', '')

# Configure Gemini API
genai.configure(api_key=api_key)

async def preprocess_image(image_file: io.BytesIO) -> bytes:
    """
    Preprocess the prescription image.
    
    This function opens an image, converts it to grayscale, resizes it to a standard size,
    and returns it as bytes for further processing. Additional preprocessing steps like
    denoising or deskewing could be added in the future.
    
    Args:
        image_file (io.BytesIO): File-like object containing the image data.
    
    Returns:
        bytes: Preprocessed image as bytes.
    
    Raises:
        Exception: If there's an error during image preprocessing.
    """
    try:
        with Image.open(image_file) as img:
            # Convert to grayscale for better OCR results
            img = img.convert('L')
            
            # Calculate the target size while maintaining aspect ratio
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            
            # Get current dimensions
            width, height = img.size
            current_pixels = width * height
            
            if current_pixels < min_pixels:
                scale = (min_pixels / current_pixels) ** 0.5
            elif current_pixels > max_pixels:
                scale = (max_pixels / current_pixels) ** 0.5
            else:
                scale = 1
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize the image
            img = img.resize((new_width, new_height))
            
            # TODO: Add more preprocessing steps (e.g., denoising, deskewing)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
        return buffered.getvalue()
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

async def transcribe_prescription(image_data: bytes) -> Dict[str, Any]:
    """
    Use Qwen2-VL-2B-Instruct model to transcribe the prescription image.
    
    This function sends the preprocessed image to the Qwen2-VL model for optical character
    recognition (OCR). It prepares the input, processes it through the model, and decodes
    the output to obtain the transcribed text.
    
    Args:
        image_data (bytes): Preprocessed image as bytes.
    
    Returns:
        Dict[str, Any]: A dictionary containing the transcribed text under the key 'text'.
    
    Raises:
        Exception: If there's an error during the transcription process.
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Prepare the input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Transcribe all the text in this prescription image."},
                ],
            }
        ]
        
        # Process the input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to appropriate device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
       
        # Generate the output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        # Debug print
        # print("Generated IDs:", generated_ids)
        
        # Move generated_ids to CPU for post-processing
        generated_ids = generated_ids.cpu()
        
        # Debug print
        # print("Inputs before moving to CPU:", inputs)
        
        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Debug print
        # print("Inputs after moving to CPU:", inputs_cpu)
        
        # Debug print
        print("Input IDs shape:", inputs_cpu['input_ids'].shape if 'input_ids' in inputs_cpu else "No input_ids found")
        print("Generated IDs shape:", generated_ids.shape)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_cpu['input_ids'], generated_ids)]
        transcribed_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return {"text": transcribed_text}
    except Exception as e:
        logger.error(f"Error transcribing prescription: {str(e)}")
        raise

async def extract_structured_data(transcribed_text: str) -> Dict[str, Any]:
    """
    Extract structured data from transcribed prescription text using Google's Gemini API.
    
    This function sends the transcribed text to the Gemini API with a prompt to extract
    structured information about medications. It processes the API response and returns
    the extracted data in a structured format.
    
    Args:
        transcribed_text (str): The OCR text from the prescription image.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'medications': A list of dictionaries, each representing a medication with fields
              for drug name, dosage, frequency, duration, and additional instructions.
            - 'disclaimer': A medical disclaimer string.
            - 'error': An error message if the extraction process fails.
    """
    disclaimer = (
        "DISCLAIMER: This system is for informational purposes only and should not "
        "be used as a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for interpretation of "
        "prescription details and medical guidance."
    )
    
    prompt = f"""
    {disclaimer}

    Given the following OCR text from a prescription, extract the structured information:

    {transcribed_text}

    Please format the extracted information as a JSON array of objects, where each object represents a medication with the following fields:
    - drug_name: The name of the prescribed medication
    - dosage: The dosage of the medication
    - frequency: How often the medication should be taken
    - duration: How long the medication should be taken
    - additional_instructions: Any additional instructions or notes

    If any field cannot be confidently extracted, use not provided for its value.
    If there are multiple medications, create a separate object for each one.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    
    try:
        safety_settings = {
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
        }
        
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        if response.parts:
            content = response.parts[0].text
            # Extract JSON objects from the content
            json_objects = re.findall(r'\{[^{}]*\}', content)
            
            if json_objects:
                parsed_data = []
                for json_obj in json_objects:
                    try:
                        parsed_obj = json.loads(json_obj)
                        parsed_data.append(parsed_obj)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON object: {json_obj}")
                
                return {
                    "medications": parsed_data,
                    "disclaimer": disclaimer
                }
            else:
                logger.error("No valid JSON objects found in Gemini API response")
                return {
                    "error": "No valid JSON objects found in Gemini API response",
                    "content": content,
                    "disclaimer": disclaimer
                }
        else:
            logger.warning("Empty response from Gemini API")
            return {
                "error": "Empty response from Gemini API",
                "disclaimer": disclaimer
            }
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return {
            "error": str(e),
            "disclaimer": disclaimer
        }

async def process_prescription(image_file: io.BytesIO) -> Dict[str, Any]:
    """
    Process a single prescription image end-to-end, including avatar video creation.
    """
    try:
        # Preprocess the image
        image_data = await preprocess_image(image_file)
        
        # Transcribe the prescription using Qwen2-VL-7B-Instruct model
        transcribed_result = await transcribe_prescription(image_data)
        print("transcribed_result: ", transcribed_result)
        
        # Extract structured data using Gemini's LLM
        structured_data = await extract_structured_data(transcribed_result['text'])
        return structured_data
    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")
        return {"error": str(e)}

# Remove or comment out the main function in prescription_transcription.py
# as it will now be handled by the Streamlit app