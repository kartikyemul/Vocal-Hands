import { useRef, useState } from "react";
import { collect, predict, refine, type Point } from "./api";
import { useHandTracker } from "./useHandTracker";

const STABLE_FRAMES = 6, MIN_CONF = 0.7, INTERVAL_MS = 150;

export default function App() {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<"translate" | "collect">("translate");
  const [live, setLive] = useState("-");
  const [text, setText] = useState("");
  const [sentence, setSentence] = useState("");
  const [label, setLabel] = useState("A");
  const [signer, setSigner] = useState("me");
  const [recording, setRecording] = useState(false);
  const [count, setCount] = useState(0);
  const [busy, setBusy] = useState(false);

  const last = useRef(0), streak = useRef({ l: "", n: 0 }), committed = useRef("");
  const rec = useRef(recording); rec.current = recording;
  const modeRef = useRef(mode); modeRef.current = mode;
  const labelRef = useRef(label); labelRef.current = label;
  const signerRef = useRef(signer); signerRef.current = signer;

  useHandTracker(video, canvas, async (pts: Point[] | null) => {
    if (!pts) { committed.current = ""; streak.current = { l: "", n: 0 }; setLive("-"); return; }
    const now = performance.now();
    if (now - last.current < INTERVAL_MS) return;
    last.current = now;
    try {
      if (modeRef.current === "collect") {
        if (rec.current && labelRef.current) {
          await collect(pts, labelRef.current, signerRef.current);
          setCount((c) => c + 1);
        }
        return;
      }
      const r = await predict(pts);
      if (!r.label || r.confidence < MIN_CONF) { setLive("?"); return; }
      setLive(`${r.label} (${Math.round(r.confidence * 100)}%)`);
      const s = streak.current;
      s.n = s.l === r.label ? s.n + 1 : 1; s.l = r.label;
      if (s.n >= STABLE_FRAMES && committed.current !== r.label) {
        committed.current = r.label;
        setText((t) => t + (r.label!.length === 1 ? r.label : ` ${r.label} `));
      }
    } catch { setLive("API offline"); }
  });

  const speak = (t: string) => {
    if (!t) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(t));
  };
  const doRefine = async () => {
    setBusy(true);
    try { const r = await refine(text); setSentence(r.text); speak(r.text); }
    finally { setBusy(false); }
  };

  return (
    <main>
      <h1>🤟 Vocal Hands</h1>
      <div className="tabs" role="tablist">
        {(["translate", "collect"] as const).map((m) => (
          <button key={m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>{m}</button>
        ))}
      </div>
      <div className="stage">
        <video ref={video} muted playsInline />
        <canvas ref={canvas} />
      </div>

      {mode === "translate" ? (
        <section>
          <p className="live" aria-live="polite">Detected: {live}</p>
          <p className="out">{text || "Start signing…"}</p>
          <div className="row">
            <button onClick={() => speak(text)}>🔊 Speak raw</button>
            <button onClick={doRefine} disabled={busy || !text}>{busy ? "…" : "✨ Refine + speak"}</button>
            <button onClick={() => { setText(""); setSentence(""); }}>Clear</button>
          </div>
          {sentence && <p className="out refined">{sentence}</p>}
        </section>
      ) : (
        <section>
          <p>Record labelled samples. Vary angle and distance; get friends to record as different signers.</p>
          <div className="row">
            <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Label (A, HELP…)" />
            <input value={signer} onChange={(e) => setSigner(e.target.value)} placeholder="Signer name" />
            <button className={recording ? "rec" : ""} onClick={() => setRecording(!recording)}>
              {recording ? "⏹ Stop" : "⏺ Record"}
            </button>
          </div>
          <p className="live">Saved this session: {count}</p>
        </section>
      )}
    </main>
  );
}
