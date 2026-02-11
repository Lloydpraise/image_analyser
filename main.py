from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import uvicorn

app = FastAPI()

# --- LOAD CLIP MODEL (Hugging Face Version) ---
# This is the exact same CLIP model but loads via Sentence Transformers
print("Loading CLIP model...")
model = SentenceTransformer('clip-ViT-B-32')

class ImageRequest(BaseModel):
    image_url: str

def smart_center_crop(img):
    """Applies the 70% Center Crop."""
    width, height = img.size
    crop_percent = 0.70
    
    new_width = width * crop_percent
    new_height = height * crop_percent
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    
    return img.crop((left, top, right, bottom))

@app.post("/vectorize")
async def vectorize_image(request: ImageRequest):
    try:
        # 1. Download image
        response = requests.get(request.image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # 2. Apply Smart Crop
        cropped_img = smart_center_crop(img)
        
        # 3. Generate Embedding
        # We ensure the image is in the correct format and compute the vector
        import numpy as np
        
        
        feat = model.encode([cropped_img]) # Passing as a list helps stability
        
        # Convert from numpy array to a standard Python list
        embedding = feat[0].tolist()
        
        return {"embedding": embedding}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)