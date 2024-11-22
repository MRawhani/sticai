from fastapi import FastAPI
import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login


import os

# Initialize FastAPI
app = FastAPI()

# Define the local path to store the model
turbo_local_model_path = "./models/stable-diffusion-3.5-large-turbo"
xl_local_model_path = "./models/stable-diffusion-3.5-large-turbo"
pipe = None  # Global variable for the pipeline

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
    try:
        # Check if the model exists locally
        if not os.path.exists(turbo_local_model_path):
            print("Model not found locally. Downloading...")
            login(token="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")

            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16)
            pipe.save_pretrained(turbo_local_model_path)  # Save the model locally for future use
            print(f"Model downloaded and saved at {turbo_local_model_path}")
        else:
            print(f"Loading model from local path: {turbo_local_model_path}")
            pipe = StableDiffusionPipeline.from_pretrained(turbo_local_model_path, torch_dtype=torch.bfloat16)

        # Move the model to GPU if available, otherwise use CPU
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Pipeline loaded successfully")
    except Exception as e:
        print(f"Error loading the pipeline: {e}")

@app.post("/generate")
async def generate_image(prompt: str):
    """
    API endpoint to generate an image from a given prompt.
    """
    global pipe
    try:
        if pipe is None:
            return {"error": "Model pipeline not loaded. Please check the server startup logs."}

        # Generate the image
        image = pipe(prompt).images[0]
        output_path = "output.png"
        image.save(output_path)
        print(f"Image generated and saved at {output_path}")
        return {"message": "Image generated successfully", "image_path": output_path}
    except Exception as e:
        print(f"Error during image generation: {e}")
        return {"error": str(e)}

@app.on_event("shutdown")
def unload_pipeline():
    """
    Clean up resources when the application shuts down.
    """
    global pipe
    pipe = None
    print("Pipeline unloaded successfully.")
