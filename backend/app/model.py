from pathlib import Path
import joblib
from .features import normalize

PATH = Path(__file__).resolve().parents[1] / "model.joblib"

class Classifier:
    def __init__(self):
        self.model = joblib.load(PATH) if PATH.exists() else None

    def predict(self, points):
        if self.model is None:
            return None
        probs = self.model.predict_proba([normalize(points)])[0]
        i = probs.argmax()
        return {"label": str(self.model.classes_[i]), "confidence": float(probs[i])}
