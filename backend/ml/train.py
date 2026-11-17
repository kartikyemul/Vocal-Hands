"""Train on data/samples.csv. Evaluates by signer when >1 signer (no leakage)."""
from pathlib import Path
import joblib, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "samples.csv")
X, y = df.filter(regex=r"^f\d+$").values, df["label"].values
print(f"{len(df)} samples, {df.label.nunique()} classes, {df.signer.nunique()} signers")

if df.signer.nunique() > 1:
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=0).split(X, y, df.signer))
    print("Split: held-out SIGNERS (honest estimate)")
else:
    tr, te = train_test_split(range(len(df)), test_size=0.25, stratify=y, random_state=0)
    print("WARNING: one signer only - accuracy will be optimistic")

m = RandomForestClassifier(n_estimators=200, random_state=0).fit(X[tr], y[tr])
pred = m.predict(X[te])
print(classification_report(y[te], pred, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y[te], pred, xticks_rotation=90)
plt.tight_layout(); plt.savefig(ROOT / "confusion_matrix.png", dpi=150)

joblib.dump(RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y), ROOT / "model.joblib")
print("Saved model.joblib - POST /reload-model or restart the API.")
