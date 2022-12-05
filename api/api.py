from auth_token import auth_token
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline, DPMSolverMultistepScheduler
from io import BytesIO
import base64 

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = "cuda"
model_id = "openvoyage/voyage-v1"
negative_prompt = ""

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

#pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to(device)

@app.get("/")
def generate(prompt: str):
    with autocast(device):
        # image = pipe(prompt, guidance_scale=8.5).images[0]
        image = pipe(prompt=f'<voyage> style {prompt}', negative_prompt=negative_prompt, num_inference_steps=25, width=512, height=512, guidance_scale=10).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
