# server.py
import os
import time
import json
import re
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
import base64

load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)


def extract_structured_dom(page_content):
    """
    Return a simplified DOM: list of form controls with attributes and surrounding label/placeholder text.
    """
    soup = BeautifulSoup(page_content, "lxml")
    controls = []
    for control in soup.select("input, textarea, select, [role='combobox']"):
        try:
            tag = control.name
            attrs = dict(control.attrs)
            label_text = ""

            # try label[for=id]
            if "id" in attrs:
                lab = soup.find("label", {"for": attrs["id"]})
                if lab and lab.get_text(strip=True):
                    label_text = lab.get_text(" ", strip=True)

            # try parent label
            if not label_text:
                parent_label = control.find_parent("label")
                if parent_label and parent_label.get_text(strip=True):
                    label_text = parent_label.get_text(" ", strip=True)

            # fallback: previous sibling text
            if not label_text:
                prev = control.previous_sibling
                if prev:
                    try:
                        label_text = prev.get_text(" ", strip=True)
                    except Exception:
                        label_text = str(prev).strip()

            control_info = {
                "tag": tag,
                "type": attrs.get("type", ""),
                "name": attrs.get("name", ""),
                "id": attrs.get("id", ""),
                "placeholder": attrs.get("placeholder", ""),
                "aria_label": attrs.get("aria-label", ""),
                "label": label_text,
                "outer_html": str(control)[:800]
            }
            controls.append(control_info)
        except Exception:
            continue
    return controls


# NOTE: braces for the JSON example must be escaped (double braces) so .format() works.
PROMPT_TEMPLATE = """
You are a smart form auto-filler LLM.

User data (plaintext):
{user_details}

Page metadata:
Title: {title}
URL: {url}

Structured form controls (JSON array):
{controls_json}

Task:
Build a JSON object that maps the user's data to the best matching form controls.
Return ONLY parseable JSON in the exact structure below.

Expected JSON structure:
{{
  "mapping": {{ "<control_selector>": "<value_to_type>", ... }},
  "notes": "<short notes about CAPTCHAs/ambiguous fields or special instructions>"
}}

Rules:
- Use selector conventions: prefer "id:THE_ID" when element has an id, otherwise "name:THE_NAME" if element has a name, otherwise provide a valid CSS selector.
- Do NOT attempt to bypass CAPTCHAs, 2FA, email or SMS verification. If present, mention in notes.
- If unsure, provide best-effort mapping and include ambiguous fields in notes.
- Return JSON only, no explanatory text.
"""


@app.post("/autofill")
def autofill():
    req = request.get_json(force=True)
    url = req.get("url")
    user_details = req.get("details", "")
    if not url or not user_details:
        return jsonify({"error": "url and details are mandatory"}), 400

    # Start Playwright (do not use context manager so browser can stay open)
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=False, args=["--start-maximized"])
    context = browser.new_context()
    page = context.new_page()

    try:
        page.goto(url, timeout=40000)
    except PWTimeout:
        # continue even if initial navigation timed out (page may still have loaded partially)
        pass
    except Exception as e:
        # if navigation fails entirely, return error
        return jsonify({"error": "Failed to navigate to URL", "details": str(e)}), 500

    # small wait for JS to render dynamic forms
    time.sleep(2)
    # attempt scroll to trigger lazy loads
    try:
        page.evaluate("() => { window.scrollBy(0, window.innerHeight); }")
        time.sleep(0.5)
    except Exception:
        pass

    html = page.content()
    title = page.title()

    controls = extract_structured_dom(html)
    controls_json = json.dumps(controls, ensure_ascii=False, indent=2)

    prompt = PROMPT_TEMPLATE.format(
        user_details=user_details,
        title=title,
        url=url,
        controls_json=controls_json
    )

    # Call LLM
    try:
        ai_resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0
        )
        ai_text = ai_resp.choices[0].message.content.strip()
    except Exception as e:
        # keep browser open for debugging, return error to frontend
        return jsonify({"error": "LLM request failed", "detail": str(e)}), 500

    # Parse LLM output (attempt JSON, fallback to extract first JSON block)
    ai_json = None
    try:
        ai_json = json.loads(ai_text)
    except Exception:
        m = re.search(r"(\{.*\})", ai_text, re.S)
        if m:
            try:
                ai_json = json.loads(m.group(1))
            except Exception:
                return jsonify({"error": "Could not parse LLM JSON output", "raw": ai_text}), 500
        else:
            return jsonify({"error": "LLM did not return JSON", "raw": ai_text}), 500

    mapping = ai_json.get("mapping", {})
    notes = ai_json.get("notes", "")

    # Apply mapping: fill fields according to selectors returned by LLM
    mapped_results = []
    errors = []

    for selector, value in mapping.items():
        try:
            # normalize selector convention
            sel = selector
            if selector.startswith("id:"):
                sel = "#" + selector.split(":", 1)[1]
            elif selector.startswith("name:"):
                name_val = selector.split(":", 1)[1]
                sel = f'[name="{name_val}"]'

            el = None
            try:
                el = page.query_selector(sel)
            except Exception:
                el = None

            if not el:
                # some sites use input fields inside iframes; try searching all frames
                for frame in page.frames:
                    try:
                        if frame.query_selector(sel):
                            el = frame.query_selector(sel)
                            break
                    except Exception:
                        continue

            if el:
                try:
                    el.scroll_into_view_if_needed()
                except Exception:
                    pass
                try:
                    # clear then type
                    el.fill("")
                except Exception:
                    pass
                try:
                    el.type(str(value), delay=10)
                except Exception:
                    # fallback: set value via JS
                    try:
                        page.eval_on_selector(sel, "(el, val) => el.value = val", str(value))
                    except Exception:
                        # try frame eval if element was inside a frame
                        for frame in page.frames:
                            try:
                                frame.eval_on_selector(sel, "(el, val) => el.value = val", str(value))
                                break
                            except Exception:
                                continue
                mapped_results.append({"selector": selector, "applied_selector": sel, "value": value})
            else:
                errors.append({"selector": selector, "error": "Element not found"})
        except Exception as ex:
            errors.append({"selector": selector, "error": str(ex)})



    # IMPORTANT: do NOT close browser or stop playwright here — leave it open so user can review & submit manually.
    # (If you want the server not to leak processes on repeated calls, you should manage browser lifecycle externally.)

    response = {
        "status": "filled",
        "message": "AI filled the form. Please review the visible browser and click Submit manually.",
        "url": url,
        "title": title,
        "mapped_fields": mapped_results,
        "errors": errors,
        "notes": notes,
        "llm_raw": ai_text,
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
