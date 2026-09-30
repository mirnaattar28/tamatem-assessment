"""War of Sparta asset generator: simple prompt in, on-style hero art out.

Everything that shapes the look (text direction AND how the Style_Guides_Sparta
images are used) lives in tm_prompt_config.json.
"""

import asyncio
import io
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "tm_prompt_config.json"
OUTPUT_PATH = BASE / "output" / "tm_new_hero.png"

VALID_RATIOS = {
    "1:1", "2:3", "3:2", "3:4", "4:3",
    "4:5", "5:4", "9:16", "16:9", "21:9"
}

load_dotenv(BASE / ".env")


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _mentions(prompt: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\b", prompt, flags=re.IGNORECASE))


def _select_references(prompt: str, cfg: dict) -> tuple[list, list]:
    """Decide which guide images act as character refs and which as style refs."""
    sr = cfg["style_references"]
    cr = cfg["character_references"]

    folder = BASE / sr["folder"]
    exts = {e.lower() for e in sr["extensions"]}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No style guide images found in {folder}")
    by_name = {p.name.lower(): p for p in files}

    # 1) Character references: hero named in the prompt -> that hero's image.
    characters = []
    if cr.get("enabled"):
        for entry in cr.get("characters", []):
            path = by_name.get(entry["file"].lower())
            if path and any(_mentions(prompt, n) for n in entry["names"]):
                characters.append(path)
        if cr.get("match_by_filename"):
            for path in files:
                name = re.sub(r"[_\-]+", " ", path.stem).strip()
                if path not in characters and _mentions(prompt, name):
                    characters.append(path)
        characters = characters[: cr.get("max_characters", 3)]

    # 2) Style references: pinned images first, then the rest spread evenly.
    pool = [p for p in files if p not in characters]
    n = sr["max_images"]
    style = [by_name[f.lower()] for f in sr.get("pinned", [])
             if f.lower() in by_name and by_name[f.lower()] not in characters]
    rest = [p for p in pool if p not in style]
    slots = max(n - len(style), 0)
    if sr.get("selection") == "all" or len(rest) <= slots:
        style += rest[:slots] if slots else []
    elif slots:
        step = len(rest) / slots
        style += [rest[int(i * step)] for i in range(slots)]
    style = style[:n] or characters
    return characters, style


def _build_prompt(user_prompt: str, cfg: dict, has_character: bool) -> str:
    """Wrap the user's simple idea with the War of Sparta art direction."""
    text = cfg["template"].format(
        user_prompt=user_prompt.strip(),
        style_reference_instruction=cfg["style_references"]["instruction"],
        **cfg["art_direction"],
    )
    if has_character:
        text += "\n\nCHARACTER CONSISTENCY: " + cfg["character_references"]["instruction"]
    return text


def _image_part(path: Path, max_side: int) -> types.Part:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


async def tm_generateAsset(prompt, aspectRatio="1:1"):
    """Generate a War of Sparta styled image and save it to ./output/tm_new_hero.png."""

    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if aspectRatio not in VALID_RATIOS:
        raise ValueError(f"aspectRatio must be one of {sorted(VALID_RATIOS)}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY (see .env.example)")

    cfg = _load_config()
    sr, cr = cfg["style_references"], cfg["character_references"]
    characters, style = _select_references(prompt, cfg)

    contents = [_build_prompt(prompt, cfg, bool(characters))]
    for path in characters:
        contents.append(cr["label"].format(name=path.stem))
        contents.append(_image_part(path, sr["max_side"]))
    contents.append(sr["label"])
    contents.extend(_image_part(p, sr["max_side"]) for p in style)

    image_kwargs = {"aspect_ratio": aspectRatio}
    if cfg.get("generation", {}).get("image_size"):
        image_kwargs["image_size"] = cfg["generation"]["image_size"]

    client = genai.Client(api_key=api_key)
    last_text = ""

    for attempt in range(3):
        response = await client.aio.models.generate_content(
            model=cfg["model"],
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**image_kwargs),
            ),
        )

        for cand in response.candidates or []:
            for part in (cand.content.parts if cand.content else None) or []:
                if part.inline_data and part.inline_data.data:
                    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(io.BytesIO(part.inline_data.data)).save(OUTPUT_PATH, "PNG")
                    return str(OUTPUT_PATH)
                if part.text:
                    last_text = part.text

        await asyncio.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"No image returned after 3 attempts. Model said: {last_text!r}"
    )