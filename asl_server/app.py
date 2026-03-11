import os
import pickle
import urllib.request
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALPHA_MODEL = None
model_load_error = None

# Fallback: Download from GitHub if Render didn't pull the LFS object
MODEL_URL = "https://media.githubusercontent.com/media/anandzeus/s-meet/main/asl_server/sign_language_model.p"
MODEL_PATH = "sign_language_model.p"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    print(f"Downloading model directly from LFS to bypass Render limitations...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")

try:
    with open(MODEL_PATH, 'rb') as f:
        ALPHA_MODEL = pickle.load(f)
except Exception as exc:
    print(f"Failed to load alpha model: {exc}")
    model_load_error = str(exc)

@app.get("/health")
async def health():
    return {
        "ok": True,
        "alpha_model_loaded": ALPHA_MODEL is not None,
        "error": model_load_error
    }

class AlphaPredictRequest(BaseModel):
    landmarks: List[Dict[str, float]]

@app.post("/asl/predict_alpha")
async def predict_alpha(req: AlphaPredictRequest):
    if ALPHA_MODEL is None:
        raise HTTPException(status_code=503, detail="Alpha model unavailable")
    
    if not req.landmarks or len(req.landmarks) < 21:
        raise HTTPException(status_code=400, detail="Not enough landmarks provided")
    
    # Extract x, y coordinates
    coords = []
    for lm in req.landmarks:
        coords.extend([lm.get('x', 0.0), lm.get('y', 0.0)])
        
    # Standardize to base landmark (first one)
    base_x, base_y = coords[0], coords[1]
    norm_coords = []
    for i in range(0, len(coords), 2):
        norm_coords.append(coords[i] - base_x)
        norm_coords.append(coords[i+1] - base_y)

    try:
        # Run prediction on the scikit-learn model
        prediction = ALPHA_MODEL.predict([norm_coords])[0]
        label = str(prediction)
        print(f"Prediction: {label}")
        return {"label": label, "score": 1.0}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
