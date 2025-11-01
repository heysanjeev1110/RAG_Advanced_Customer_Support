from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from google import genai
import re
import json

# Optional: import log from agent if shared, else define locally
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class PerceptionResult(BaseModel):
    user_input: str
    intent: Optional[str]
    entities: List[str] = []
    tool_hint: Optional[str] = None


def extract_perception(user_input: str) -> PerceptionResult:
    """Extracts intent, entities, and tool hints using LLM"""

    prompt = f"""
You are an AI that extracts structured facts from user input.

Input: "{user_input}"

Return the response as a Python dictionary with keys:
- intent: (brief phrase about what the user wants)
- entities: a list of strings representing keywords or values (e.g., ["INDIA", "ASCII"])
- tool_hint: (name of the MCP tool that might be useful, if any)

Output only the dictionary on a single line. Do NOT wrap it in ```json or other formatting. Ensure `entities` is a list of strings, not a dictionary.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        log("perception", f"LLM output: {raw}")

        # Strip Markdown backticks if present
        clean = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        try:
            # Try to parse as JSON first
            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError:
                # If that fails, try replacing single quotes with double quotes
                try:
                    parsed = json.loads(clean.replace("'", '"'))
                except json.JSONDecodeError:
                    # Last resort: use ast.literal_eval (safer than eval)
                    import ast
                    clean_python = clean.replace("null", "None")
                    parsed = ast.literal_eval(clean_python)
        except Exception as e:
            log("perception", f"⚠️ Failed to parse cleaned output: {e}")
            raise

        # Fix common issues
        if isinstance(parsed.get("entities"), dict):
            parsed["entities"] = list(parsed["entities"].values())


        return PerceptionResult(user_input=user_input, **parsed)

    except Exception as e:
        log("perception", f"⚠️ Extraction failed: {e}")
        return PerceptionResult(user_input=user_input)
