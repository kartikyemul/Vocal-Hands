import csv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .features import normalize
from .model import Classifier
from .nlp import refine_text

DATA = Path(__file__).resolve().parents[1] / "data" / "samples.csv"
app = FastAPI(title="Vocal Hands API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])
clf = Classifier()

class Landmarks(BaseModel):
    points: list[list[float]] = Field(..., min_length=21, max_length=21)

class Sample(Landmarks):
    label: str
    signer: str = "anon"

class Text(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": clf.model is not None}

@app.post("/predict")
def predict(body: Landmarks):
    return clf.predict(body.points) or {"label": None, "confidence": 0.0}

@app.post("/collect")
def collect(s: Sample):
    new = not DATA.exists()
    with DATA.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "signer"] + [f"f{i}" for i in range(63)])
        w.writerow([s.label.upper(), s.signer, *normalize(s.points)])
    return {"saved": True}

@app.post("/refine")
def refine(t: Text):
    return {"text": refine_text(t.text)}

@app.post("/reload-model")
def reload_model():
    global clf
    clf = Classifier()
    return health()
