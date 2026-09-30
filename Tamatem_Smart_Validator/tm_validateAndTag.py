"""VIP Baloot frame validator.
AI vision judges the 'vibes' (luxury, artifacts); coverage percentages are
computed with exact pixel math (no AI involved)."""
import asyncio
import io
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

BASE = Path(__file__).resolve().parent
TEMPLATES = BASE / "templates"
AVATAR_TEMPLATE = TEMPLATES / "user_image.jpg"
NAME_TEMPLATE = TEMPLATES / "user_name.png"
REPORT_PATH = BASE / "output" / "tm_validation_report.json"

load_dotenv(BASE / ".env")
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3-flash-preview")
ALPHA_THRESHOLD = 32          # frame pixel counts as "covering" if alpha > this (0-255)
AVATAR_FAIL_PCT = 50.0
NAME_FAIL_PCT = 40.0
AVATAR_BG_THRESHOLD = 250     # user_image.jpg pixels darker than this belong to the avatar circle
# How to read "the user_image.jpg area":
#   "full"   -> the whole user_image.jpg canvas (literal reading of the brief)
#   "circle" -> only the visible avatar silhouette inside it
AVATAR_AREA_MODE = "full"
VISION_RUNS = 3               # independent AI opinions, combined by majority vote

VISION_PROMPT = """You are an art director reviewing avatar frames for a premium card game (VIP Baloot).
You get two views of one frame: (1) the frame on a dark background, (2) the frame over a sample avatar.

1. luxury_check: true if the frame clearly looks premium: polished gold/metal textures, gems, intricate
   ornament, depth and bevels. false only if it looks cheap, flat, basic or unfinished.

2. ai_artifacts: list AI-generation flaws. Be calibrated:
   - Frames come from a professional pipeline; most are fine. Only report a flaw you are CONFIDENT is a
     real, clearly visible defect: blurry or melted gems, garbled or nonsensical patterns, broken or
     disconnected shapes, jagged or smeared edges, stray pixels.
   - Intentional themed content (different objects, characters, vehicles, props on each side) and normal
     stylistic variation are NOT artifacts. Asymmetry is a flaw only when an element that is clearly meant
     to be symmetric (borders, corner ornaments) is visibly mismatched.
   - severity "major" = a player would notice it at normal viewing size. "minor" = only visible on close
     inspection or arguable. When unsure, use "minor".
   - If there are no clear flaws, return an empty list. Do not invent flaws.
   Give each note as a short, specific sentence stating where the flaw is."""

SCHEMA = {
    "type": "object",
    "properties": {
        "luxury_check": {"type": "boolean"},
        "ai_artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "severity": {"type": "string", "enum": ["minor", "major"]},
                },
                "required": ["note", "severity"],
            },
        },
    },
    "required": ["luxury_check", "ai_artifacts"],
}


def _coverage(frame: Image.Image) -> tuple[float, float]:
    """Exact pixel-math coverage, computed on the templates' own pixel grid.

    avatar_pct : % of the user_image.jpg area (see AVATAR_AREA_MODE) covered by the frame.
    name_pct   : % of the red name zone (user_name.png) covered by the frame.
    """
    avatar_gray = Image.open(AVATAR_TEMPLATE).convert("L")
    size = avatar_gray.size
    if AVATAR_AREA_MODE == "circle":
        avatar_mask = np.array(avatar_gray) < AVATAR_BG_THRESHOLD   # circle vs. white background
    else:
        avatar_mask = np.ones((size[1], size[0]), dtype=bool)         # whole template canvas
    if not avatar_mask.any():
        raise ValueError("Could not find the avatar area in user_image.jpg")

    name = Image.open(NAME_TEMPLATE).convert("RGBA")
    if name.size != size:
        name = name.resize(size, Image.NEAREST)
    r, g, b, a = (np.array(name)[..., i].astype(int) for i in range(4))
    red_mask = (r > 150) & (g < 100) & (b < 100) & (a > 0)
    if not red_mask.any():
        raise ValueError("No red zone found in user_name.png")

    # Bring the frame onto the same grid as the templates (as it would be laid over the avatar).
    alpha = frame.split()[3].resize(size, Image.BILINEAR)
    frame_mask = np.array(alpha) > ALPHA_THRESHOLD

    avatar_pct = 100.0 * (frame_mask & avatar_mask).sum() / avatar_mask.sum()
    name_pct = 100.0 * (frame_mask & red_mask).sum() / red_mask.sum()
    return round(float(avatar_pct), 2), round(float(name_pct), 2)


def _png_part(img: Image.Image, max_side=1536) -> types.Part:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


async def _vision_once(client, parts) -> dict:
    resp = await client.aio.models.generate_content(
        model=VISION_MODEL,
        contents=[VISION_PROMPT, *parts],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return json.loads(resp.text)


async def _vision(frame: Image.Image) -> dict:
    """Run several independent AI reviews and combine them by majority vote."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY (see .env.example)")
    dark = Image.new("RGBA", frame.size, (30, 30, 34, 255))
    dark.alpha_composite(frame)
    sample = Image.open(AVATAR_TEMPLATE).convert("RGBA").resize(frame.size)
    sample.alpha_composite(frame)
    parts = [_png_part(dark.convert("RGB")), _png_part(sample.convert("RGB"))]

    client = genai.Client(api_key=api_key)
    runs = await asyncio.gather(*[_vision_once(client, parts) for _ in range(VISION_RUNS)])

    need = VISION_RUNS // 2 + 1
    luxury = sum(r["luxury_check"] for r in runs) >= need

    # Only MAJOR artifacts count, and only if most runs agree there are some.
    major_lists = [[a["note"] for a in r["ai_artifacts"] if a["severity"] == "major"] for r in runs]
    if sum(1 for m in major_lists if m) >= need:
        artifacts = max(major_lists, key=len)
    else:
        artifacts = []
    return {"luxury_check": luxury, "ai_artifacts": artifacts}


async def tm_validateAndTag(framePath):
    """Validate one frame and save ./output/tm_validation_report.json."""
    frame = Image.open(framePath).convert("RGBA")
    avatar_pct, name_pct = _coverage(frame)
    vision = await _vision(frame)

    luxury = bool(vision["luxury_check"])
    artifacts = [str(x) for x in vision["ai_artifacts"]]
    passed = luxury and not artifacts and avatar_pct <= AVATAR_FAIL_PCT and name_pct <= NAME_FAIL_PCT

    report = {
        "luxury_check": luxury,
        "ai_artifacts": artifacts,
        "avatar_coverage_pct": avatar_pct,
        "username_blockage_pct": name_pct,
        "final_verdict": "PASS" if passed else "FAIL",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report