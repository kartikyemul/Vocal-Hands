import { useEffect, useRef } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { Point } from "./api";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export function useHandTracker(
  video: React.RefObject<HTMLVideoElement>,
  canvas: React.RefObject<HTMLCanvasElement>,
  onHand: (points: Point[] | null) => void
) {
  const cb = useRef(onHand);
  cb.current = onHand;

  useEffect(() => {
    let raf = 0, stream: MediaStream | undefined, stopped = false;
    (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM);
      const lm = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL },
        runningMode: "VIDEO",
        numHands: 1,
      });
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      const v = video.current!;
      v.srcObject = stream;
      await v.play();
      const loop = () => {
        if (stopped) return;
        const res = lm.detectForVideo(v, performance.now());
        const c = canvas.current!, ctx = c.getContext("2d")!;
        c.width = v.videoWidth; c.height = v.videoHeight;
        ctx.clearRect(0, 0, c.width, c.height);
        const hand = res.landmarks[0];
        if (hand) {
          ctx.fillStyle = "#ffd400";
          hand.forEach((p) => {
            ctx.beginPath();
            ctx.arc(p.x * c.width, p.y * c.height, 5, 0, 7);
            ctx.fill();
          });
          cb.current(hand.map((p) => [p.x, p.y, p.z] as Point));
        } else cb.current(null);
        raf = requestAnimationFrame(loop);
      };
      loop();
    })().catch((e) => alert("Camera/model error: " + e.message));
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [video, canvas]);
}
