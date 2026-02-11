import base64
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import os
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model - Set to CPU explicitly to save memory
print("Loading CLIP Model (CPU Optimized)...")
try:
    model = SentenceTransformer('clip-ViT-B-32', device='cpu')
    model.eval() # Set to evaluation mode
except Exception as e:
    print(f"FAILED TO LOAD MODEL: {e}")

class ImageRequest(BaseModel):
    image_url: str = None
    image_base64: str = None

def smart_center_crop(img):
    img = img.convert("RGB")
    width, height = img.size
    crop_percent = 0.70
    crop_width = width * crop_percent
    crop_height = height * crop_percent
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height
    cropped = img.crop((int(left), int(top), int(right), int(bottom)))
    # Ensure output is always exactly 224x224 to avoid tensor dimension issues
    resized = cropped.resize((224, 224), Image.Resampling.LANCZOS)
    return resized

@app.post("/vectorize")
async def vectorize_image(request: ImageRequest):
    try:
        img = None
        if request.image_base64:
            encoded = request.image_base64
            if "," in encoded:
                encoded = encoded.split(",")[1]
            img_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_data))
        
        elif request.image_url:
            response = requests.get(request.image_url, timeout=10)
            img = Image.open(BytesIO(response.content))
        
        if not img:
            raise HTTPException(status_code=400, detail="No image provided")

        processed_img = smart_center_crop(img)

        # Use torch.no_grad() to minimize RAM usage during inference
        with torch.no_grad():
            # Encode with batch_size to handle padding properly
            embeddings = model.encode([processed_img], batch_size=1, convert_to_numpy=True)
            embedding = embeddings[0].tolist()
        
        return {"embedding": embedding}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)