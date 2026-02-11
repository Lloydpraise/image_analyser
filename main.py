import base64
import requests
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# --- NEW: FIX CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your HTML Lab and any other source
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = SentenceTransformer('clip-ViT-B-32')

class ImageRequest(BaseModel):
    image_url: str = None
    image_base64: str = None

def smart_center_crop(img):
    width, height = img.size
    crop_percent = 0.70
    left = (width - width * crop_percent) / 2
    top = (height - height * crop_percent) / 2
    right = (width + width * crop_percent) / 2
    bottom = (height + height * crop_percent) / 2
    return img.crop((left, top, right, bottom))

@app.post("/vectorize")
async def vectorize_image(request: ImageRequest):
    try:
        img = None
        
        # Logic to handle both sources
        if request.image_base64:
            # Handle Base64 (Data URI or raw string)
            header, encoded = request.image_base64.split(",", 1) if "," in request.image_base64 else (None, request.image_base64)
            img_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_data)).convert("RGB")
        
        elif request.image_url:
            # Handle URL
            response = requests.get(request.image_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        
        if not img:
            raise HTTPException(status_code=400, detail="No image source provided")

        cropped_img = smart_center_crop(img)
        embedding = model.encode(cropped_img).tolist()
        
        return {"embedding": embedding}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)