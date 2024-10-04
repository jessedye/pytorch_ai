import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from reportlab.pdfgen import canvas
import os

app = FastAPI()

# Load the model and tokenizer for GPT-J
model_name = "EleutherAI/gpt-j-6B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a proper pad token if it's not available
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load Stable Diffusion model for AI art generation
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Define input model for FastAPI
class PromptRequest(BaseModel):
    prompt: str

# Define the generate endpoint for text generation
@app.post("/generate/")
async def generate_text(request: PromptRequest):
    # Tokenize input with padding and truncation
    inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        inputs['input_ids'],
        max_length=300,         # Increased max_length for detailed responses
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,         # Enable sampling for diverse responses
        temperature=0.7,        # Slightly higher randomness for creative outputs
        top_k=50,               # Consider top 50 tokens
        top_p=0.9,              # Sample from 90% of the probability distribution
        repetition_penalty=1.2  # Penalize repetition for more diverse long outputs
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response_text}

# Define the generate art endpoint for AI art creation using Stable Diffusion
@app.post("/create-art/")
async def create_art(prompt: str):
    # Generate an image from a text prompt using Stable Diffusion
    image = pipe(prompt).images[0]
    
    # Define the path where the image will be saved
    image_path = "/path/to/save/art.png"  # Change this path to your desired location
    
    # Save the image to a file
    image.save(image_path)

    return {"message": f"Art created and saved as {image_path}"}

# Define the create text file endpoint
@app.post("/create-text-file/")
async def create_text_file(content: str):
    # Define the path where the text file will be saved
    text_file_path = "/path/to/save/output.txt"  # Change this path to your desired location

    # Create a text file and write content to it
    with open(text_file_path, "w") as f:
        f.write(content)

    return {"message": f"Text file created and saved as {text_file_path}"}

# Define the create PDF file endpoint
@app.post("/create-pdf/")
async def create_pdf(content: str):
    # Define the path where the PDF will be saved
    pdf_file_path = "/path/to/save/output.pdf"  # Change this path to your desired location

    # Create a PDF file with the provided content
    pdf = canvas.Canvas(pdf_file_path)
    pdf.drawString(100, 750, content)
    pdf.save()

    return {"message": f"PDF created and saved as {pdf_file_path}"}
