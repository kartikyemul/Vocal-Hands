# 🤟 Vocal Hands

Assistive ASL tool: webcam → hand landmarks (in the browser, **no video leaves the device**) → classifier (FastAPI + scikit-learn) → text → AI cleanup (Groq LLM) → speech (browser TTS).

```
Webcam → React+TS → MediaPipe HandLandmarker → 21 landmarks
       → POST /predict → FastAPI → normalize → RandomForest → label+confidence
       → stability filter → raw text → POST /refine → Groq → sentence → Text-to-Speech
```

## Step 0 - Prerequisites
Python 3.10+, Node 18+, a webcam, a free Groq key from https://console.groq.com

## Step 1 - Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # paste GROQ_API_KEY
uvicorn app.main:app --reload --port 8000
```
Check http://localhost:8000/health -> `{"ok":true,"model_loaded":false}` and http://localhost:8000/docs.

## Step 2 - Frontend
```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```
Allow camera. You should see yellow dots on your hand. Detected shows "?" or "API offline" until a model exists - expected.

## Step 3 - Collect data (Collect tab)
1. Type label `A`, signer name, press Record, hold the sign, move your hand slightly (angle, distance, tilt) for ~10 s, Stop. Aim for **150-300 samples per sign**.
2. Repeat for your starter vocabulary. Suggested start: A-Z static letters (skip J, Z - they need motion) + words `HELP`, `WATER`, `PAIN`, `YES`, `NO`.
3. **Ask 3-5 other people to record as different signer names.** This is what makes your evaluation honest.
4. Data lands in `backend/data/samples.csv`. Learn real signs from ASL University / Handspeak / ASL Citizen; don't guess.

## Step 4 - Train (data science step)
```bash
cd backend && python ml/train.py
```
Prints a per-class report, saves `confusion_matrix.png` and `model.joblib`. With 2+ signers it tests on **unseen signers**; put that number on your poster. Look at the confusion matrix for look-alike signs (M/N/S/T, etc.) and collect more data for them.
Then `curl -X POST localhost:8000/reload-model` (or restart uvicorn).

## Step 5 - Translate
Translate tab: hold a sign steady ~1 s; it commits to the text box. Then **Refine + speak** sends the raw string to Groq running OpenAI's open-weight `openai/gpt-oss-120b` (swap to `openai/gpt-oss-20b` in `.env` for speed), which fixes spelling errors and (for words) turns gloss order into English, then reads it aloud.

## Lightweight by design
Backend: 8 packages (pydantic/joblib come bundled). Frontend: React + one MediaPipe package. Inference is a ~1 MB Random Forest on 63 numbers per frame; the only heavy model (GPT-OSS) runs on Groq's servers, only when you press Refine.

## Tuning knobs (frontend/src/App.tsx)
`STABLE_FRAMES` (higher = fewer false letters), `MIN_CONF`, `INTERVAL_MS`.

## Roadmap (what to build next, in order)
1. **Evaluate**: leave-one-signer-out CV; log accuracy per signer and per lighting condition.
2. **Motion signs (J, Z, most words)**: buffer 30 frames of landmarks, train a small 1D-CNN/LSTM in PyTorch; add `/predict-sequence`. Use the Kaggle Google ISLR or ASL Citizen datasets for pretraining.
3. **Two hands / pose**: set `numHands: 2`, add MediaPipe Pose for arm position.
4. **Two-way mode**: browser SpeechRecognition shows live captions of a hearing person's reply.
5. **Practice mode**: show a target sign and give feedback from confidence.
6. **Product polish**: PWA install, offline model via ONNX in-browser, Docker compose, tests, CI.
7. **Community validation**: get feedback from Deaf/ASL users; add it to your expo slides.

## Limitations (say these out loud at the expo)
Static-pose model only; no facial grammar; ASL regional variation; accuracy depends on data diversity. This is an assistive and learning aid, not an interpreter replacement. Check the license of any dataset you use.
