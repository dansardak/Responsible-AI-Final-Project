# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import torch
from diffusers import StableDiffusionPipeline
import requests
from PIL import Image
import io
import os
import base64
import matplotlib.pyplot as plt
import numpy as np
from google import genai 

from PIL import Image
from io import BytesIO
import base64
# For open source models
from huggingface_hub import login
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load environment variables from .env file
from dotenv import load_dotenv
# from calcmetrics import get_metrics
# Load the environment variables
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# genai.configure(api_key=GOOGLE_API_KEY)
GOOGLE_MODEL_NAME = 'gemini-2.0-flash-exp-image-generation'


# google_model = genai.GenerativeModel(GOOGLE_MODEL_NAME)

client = genai.Client()
from google.genai import types




from google.genai import types


# Function to display images
def display_images(images, titles):
    """Display multiple images with titles."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Initialize models globally
import openai
import torch
from diffusers import StableDiffusionPipeline, KandinskyV22Pipeline, KandinskyV22PriorPipeline

# DALL-E setup
openai.api_key = OPENAI_API_KEY

# Stable Diffusion setup
sd_device = "cuda" if torch.cuda.is_available() else "cpu"
sd_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=sd_dtype
).to(sd_device)
if sd_device == "cuda":
    sd_pipe.enable_attention_slicing()

# Kandinsky setup
kandinsky_device = "cuda" if torch.cuda.is_available() else "cpu"
kandinsky_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", 
    torch_dtype=kandinsky_dtype
).to(kandinsky_device)

decoder_pipe = KandinskyV22Pipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", 
    torch_dtype=kandinsky_dtype
).to(kandinsky_device)

# Google Imagen setup
imagen_client = genai.Client()
# 1. DALL-E (OpenAI) - Requires API
def generate_dalle(prompt, count=1, size="1024x1024", save_path=None):
    """Generate image using DALL-E 2 via OpenAI API."""
    response = openai.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size=size,
        n=count,
    )
    
    images = []
    for i, img_data in enumerate(response.data):
        # Download the image
        image_url = img_data.url
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))

        if save_path:
            save_name = save_path.replace('.png', f'_{i}.png')
            image.save(save_name)
        
        images.append(image)

    return images[0] if count == 1 else images

# 2. Stable Diffusion - Open Source, run locally
def generate_stable_diffusion(prompt, count=1, save_path=None):
    """Generate image using Stable Diffusion locally."""
    # Generate images
    images = []
    with torch.no_grad():
        for i in range(count):
            image = sd_pipe(prompt, guidance_scale=7.5,).images[0]
            if save_path:
                save_name = save_path.replace('.png', f'_{i}.png')
                image.save(save_name)
            images.append(image)
    
    return images[0] if count == 1 else images

# 3. Gemini2 (Google) - Requires API
def generate_imagen(prompt, count=1, save_path=None):
    """Generate image using Google's Imagen via API."""
    images = []
    for i in range(count):
        response = imagen_client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))
                if save_path:
                    save_name = save_path.replace('.png', f'_{i}.png')
                    image.save(save_name)
                images.append(image)
                break
    
    return images[0] if count == 1 else images

def generate_kandinsky(prompt, count=1, save_path=None):
    """Generate image using Kandinsky 2.2 (Prior + Decoder)."""
    negative_prompt = None # Or provide a negative prompt if desired
    images = []
    
    for i in range(count):
        # Generate image embeddings
        image_embeds, negative_image_embeds = prior_pipe(
            prompt, negative_prompt=negative_prompt
        ).to_tuple()

        # Generate final image
        with torch.no_grad():
            image = decoder_pipe(
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                height=768,
                width=768,
                num_inference_steps=50,
            ).images[0]

        if save_path:
            save_name = save_path.replace('.png', f'_{i}.png')
            image.save(save_name)
            
        images.append(image)

    return images[0] if count == 1 else images

#TODO: FIGURE OUT
# def generate_midjourney(prompt, api_key):
#     """Generate image using Midjourney via third-party API."""
#     # Note: Midjourney doesn't have an official API, this uses a hypothetical third-party service
#     response = requests.post(
#         "https://api.third-party-midjourney-service.com/generate",
#         json={"prompt": prompt},
#         headers={"Authorization": f"Bearer {api_key}"}
#     )
    
#     image = Image.open(io.BytesIO(response.content))
#     return image


def genImages(prompt, savedir="images", debug=False, count=5):
    os.makedirs(savedir, exist_ok=True)

    # if debug:
    #     print(f"Generating {count} images for prompt: {prompt}")
    
    generated_images = []
    titles = []
    

    # DALL-E
    # try:
    #     if debug:
    #         print("Starting DALLE image generation...")
    #     dalle_images = generate_dalle(prompt, count=count, save_path=f"{savedir}/dalle_image.png")
    # except Exception as e:
    #     print(f"DALL-E generation failed: {e}")

    # Imagen
    try:
        if debug:
            print('Starting google generation')
        imagen_images = generate_imagen(prompt, count=count, save_path=f"{savedir}/imagen_image.png")
    except Exception as e:
        print(f"Google Imagen generation failed: {e}")

    # Kandinsky
    try:
        if debug:
            print('Starting kandinsky generation')
        kandinsky_images = generate_kandinsky(prompt, count=count, save_path=f"{savedir}/kandinsky_image.png")
    except Exception as e:
        print(f"Kandinsky generation failed: {e}")

    # Generate with Stable Diffusion (open source, local)
    try:
        if debug:
            print(f"Generating with Stable Diffusion locally...")
        sd_images = generate_stable_diffusion(prompt, count=count, save_path=f"{savedir}/sd_image.png")
    except Exception as e:
        print(f"Stable Diffusion generation failed: {e}")




locations = ['North America', 'South America', 'Europe', 'Africa', 'Asia']



# prompts = ['George chasing detoasty']

# for prompt in prompts:
#     os.makedirs(f"images/{prompt}", exist_ok=True)
#     genImages(prompt, savedir=f"images/{prompt}", debug=True, count=1)   


import os
with open('second_paper_prompts.txt', 'r') as f:
    prompts = f.readlines()

prompts = [prompt.replace('/', '-').strip() for prompt in prompts]
regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', ]

count = 10
# prompts = prompts[:1]

ct = len(prompts)
#make prompt dirs
# for i, prompt in enumerate(prompts):
#     os.makedirs(f"images/{prompt}", exist_ok=True)
#     genImages(prompt, savedir=f"images/{prompt}", debug=False, count=count)
#     print(f"Generated for prompt {i+1}/{ct}")


import time
print('Starting DALLE generation')
# run dalle 
for i, prompt in enumerate(prompts):
    print(f"Starting DALLE generation for prompt {i+1}/{ct}")
    
    try:
        for i in range(0, count, 5):
            batch_size = min(5, count - i)
            # print(f"Generating batch {i//5 + 1} of {(count + 4)//5}")
            dalle_images = generate_dalle(prompt, count=batch_size, save_path=f"images/{prompt}/dalle_image_{i}.png")
            time.sleep(61)
    # Sleep for 1 minute between batches to avoid rate limits
    except Exception as e:
        print(f"DALLE generation failed: {e}")

    

# Calculate metrics for generated images
 

