import os
from dotenv import load_dotenv
load_dotenv()

SYSTEM = (
    "You clean up raw output from a sign-language recognizer. The input may "
    "contain fingerspelled letters with errors and ASL gloss words in ASL "
    "order. Return ONLY one natural, correct English sentence. Do not add "
    "facts that are not implied. If it is unintelligible, return it unchanged."
)
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # or openai/gpt-oss-20b (faster)

def refine_text(raw: str) -> str:
    raw = raw.strip()
    key = os.getenv("GROQ_API_KEY")
    if not raw or not key:
        return raw
    try:
        from groq import Groq
        r = Groq(api_key=key).chat.completions.create(
            model=MODEL,
            temperature=0.2,
            max_completion_tokens=1024,
            extra_body={"reasoning_effort": "low"},  # short task: keep latency low
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": raw}],
        )
        return (r.choices[0].message.content or raw).strip()
    except Exception as e:  # never break the live demo on an API hiccup
        print("Groq error:", e)
        return raw
