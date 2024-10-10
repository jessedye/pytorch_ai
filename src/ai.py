import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BlipProcessor
from diffusers import StableDiffusionPipeline
import os
import uuid
import time
from huggingface_hub import login

# Fetch the token from the environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if hf_token:
    login(hf_token)
else:
    raise EnvironmentError("HUGGINGFACE_TOKEN is not set in the environment.")


app = FastAPI()

# Model names
model_name = "meta-llama/Llama-3.2-11B-Vision"
stable_diffusion_model_name = "CompVis/stable-diffusion-v1-4"

# Device configuration - Use 'bfloat16' or 'float16' for efficiency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models and tokenizer variables
model = None
tokenizer = None
pipe = None

# Function to load LLaMA-3.2-11B-Vision model and tokenizer
def load_llama_vision_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading LLaMA-3.2-11B-Vision model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,  # Use bfloat16 for reduced memory usage
            device_map="auto"  # Automatically handle multi-GPU setup
        ).to(device)

        # Add a proper pad token if it's not available
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

# Function to load Stable Diffusion pipeline
def load_stable_diffusion():
    global pipe
    if pipe is None:
        print("Loading Stable Diffusion pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            stable_diffusion_model_name, 
            torch_dtype=torch.float16  # Use FP16 for speed and memory efficiency
        ).to(device)

# Define input model for FastAPI
class PromptRequest(BaseModel):
    prompt: str

# Event handler to load models on startup
@app.on_event("startup")
async def startup_event():
    load_llama_vision_model()     # Load LLaMA-3.2-11B-Vision model and tokenizer at startup
    load_stable_diffusion()       # Load Stable Diffusion model at startup
    print("Models loaded successfully at startup.")

# Function to generate response following OpenAI JSON format
def format_openai_response(generated_text, model_name):
    return {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ]
    }

# Define the generate endpoint for text generation
@app.post("/direct/")
async def generate_text(request: PromptRequest):
    # Tokenize input with padding and truncation
    inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=20,      # Generate up to 20 new tokens
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.3,         # Randomness for creative outputs
        top_k=50,                # Consider top 50 tokens
        top_p=0.9,               # Sample from 90% of the probability distribution
        repetition_penalty=1.2   # Penalize repetition for more diverse outputs
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_openai_response(response_text, model_name)

# POST endpoint for more creative responses
@app.post("/creative/")
async def generate_creative_text(request: PromptRequest):
    # Tokenize input with padding and truncation
    inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=200,     # Generate up to 200 new tokens
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,        # Higher temperature for creative output
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_openai_response(response_text, model_name)

# Define the endpoint for image generation using Stable Diffusion
@app.post("/generate_image/")
async def generate_image(request: PromptRequest):
    # Generate image from the prompt
    image = pipe(request.prompt).images[0]

    # Create a unique filename using uuid
    image_filename = f"{uuid.uuid4()}.png"
    image_path = f"/app/{image_filename}"

    # Save the image to the generated path
    image.save(image_path)

    return {
        "id": str(uuid.uuid4()),
        "object": "image_generation",
        "created": int(time.time()),
        "model": stable_diffusion_model_name,
        "image_path": image_path
    }

# Define the endpoint for image-to-text generation using LLaMA Vision
@app.post("/generate_image_to_text/")
async def generate_image_to_text(request: PromptRequest):
    processor = BlipProcessor.from_pretrained(model_name)
    inputs = processor(request.prompt, return_tensors="pt").to(device)

    # Generate text from the image input
    outputs = model.generate(**inputs)
    response_text = processor.decode(outputs[0], skip_special_tokens=True)

    return format_openai_response(response_text, model_name)
