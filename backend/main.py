from fastapi import FastAPI
import torch
from diffusers import StableDiffusion3Pipeline,StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,EulerAncestralDiscreteScheduler
from huggingface_hub import login
from uuid import uuid4

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Initialize FastAPI
app = FastAPI()
# Attach a static directory to serve files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the local path to store the model
xl_local_model_path = "./models/stable-diffusion-xl-base-1.0"
refiner_local_model_path = "./models/stable-diffusion-xl-refiner-1.0"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
static_dir = "./static"  # Directory to store generated images

pipe = None  # Global variable for the pipeline
refiner_pipe = None  # Global variable for the pipeline


# Ensure the static directory exists
os.makedirs(static_dir, exist_ok=True)

@app.get('/')
def index():
    try:
        
        print("Torch version:", torch.__version__)
        if torch.cuda.is_available(): 
            print("avail")
       
    except Exception as s:
        print(s)
    print("Indexing")
    return {'data': {"name":'Mohammed', 'age':27}}

@app.on_event("startup")
def load_pipeline():
    """
    This function runs when the FastAPI app starts. It downloads the model if not found locally,
    and loads it into memory for use.
    """
    global pipe
    global refiner_pipe
    try:
        # Check if the model exists locally
        if not os.path.exists(xl_local_model_path):
            print("Model not found locally. Downloading...")
            login(token=huggingface_token)

            pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id,  torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
            pipe.save_pretrained(xl_local_model_path)  # Save the model locally for future use
            print(f"Model downloaded and saved at {xl_local_model_path}")
        else:
            print(f"Loading model from local path: {xl_local_model_path}")
            pipe = StableDiffusionXLPipeline.from_pretrained(xl_local_model_path, torch_dtype=torch.bfloat16)
        # refiner checking
        if not os.path.exists(refiner_local_model_path):
            print("Refiner Model not found locally. Downloading...")

            refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_model_id,  torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
            refiner_pipe.save_pretrained(refiner_local_model_path)  # Save the model locally for future use
            print(f"Refiner Model downloaded and saved at {refiner_local_model_path}")
        else:
            print(f"Loading Refiner model from local path: {refiner_local_model_path}")
            refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_local_model_path, torch_dtype=torch.bfloat16)

        # Move the model to GPU if available, otherwise use CPU
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        refiner_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Pipelines loaded successfully")
    except Exception as e:
        print(f"Error loading the pipeline: {e}")

@app.post("/generate")
async def generate_image(prompt: str):
    """
    API endpoint to generate an image from a given prompt.
    """
    global pipe
    global refiner_pipe

    try:
        if pipe is None or refiner_pipe is None:
            return {"error": "Model pipeline not loaded. Please check the server startup logs."}

        scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler

        # Generate the image
        image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5, height=512, width=512).images[0]
        refined_image = refiner_pipe(prompt=prompt, image=image, num_inference_steps=50,guidance_scale=7.5, height=512, width=512).images[0]
        output_filename = f"{uuid4().hex}.png"  # Generate a unique filename
        output_path = os.path.join(static_dir, output_filename)
        refined_image.save(output_path)
        print(f"Image generated and saved at {output_path}")

        # Return the URL to access the image
        image_url = f"/view-image/{output_filename}"
        return {"message": "Image generated successfully", "image_url": image_url}
    except Exception as e:
        print(f"Error during image generation: {e}")
        return {"error": str(e)}
    
@app.get("/view-image/{image_name}")
def view_image(image_name: str):
    """
    API endpoint to serve a specific image.
    """

    print('image viewed')
    file_path = os.path.join(static_dir, image_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "Image not found"}
    
@app.on_event("shutdown")
def unload_pipeline():
    """
    Clean up resources when the application shuts down.
    """
    global pipe
    pipe = None
    print("Pipeline unloaded successfully.")
