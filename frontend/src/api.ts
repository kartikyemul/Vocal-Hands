export type Point = [number, number, number];
const BASE = "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return r.json();
}

export const predict = (points: Point[]) =>
  post<{ label: string | null; confidence: number }>("/predict", { points });
export const collect = (points: Point[], label: string, signer: string) =>
  post("/collect", { points, label, signer });
export const refine = (text: string) => post<{ text: string }>("/refine", { text });
