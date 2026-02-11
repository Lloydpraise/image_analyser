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

app = FastAPI()

# --- 1. STRICT CORS CONFIG ---
# This must be defined before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (essential for local testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, etc.
    allow_headers=["*"],
)

# Load Model
print("Loading Model...")
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
        if request.image_base64:
            # Check if it has the data:image/jpeg;base64, prefix
            encoded = request.image_base64
            if "," in encoded:
                encoded = encoded.split(",")[1]
            img_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_data)).convert("RGB")
        
        elif request.image_url:
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

# For local testing vs Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)