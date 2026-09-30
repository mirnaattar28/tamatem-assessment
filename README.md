# Tamatem Assessment

Both parts are in Python (3.10 or newer).

## Setup (same for both parts)

1. Go into the folder and install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Edit `.env` file in that folder and put in your Gemini key:
   ```
   GEMINI_API_KEY=your_key_here
   ```

## Part 1: Tamatem_Asset_Generator

```
cd Tamatem_Asset_Generator
python main.py "Sparta warrior fighting a lion"
```

The image is saved to `output/tm_new_hero.png`. Style images are in `Style_Guides_Sparta/`
and all the prompt and style settings are in `tm_prompt_config.json`.



## Part 2: Tamatem_Smart_Validator

```
cd Tamatem_Smart_Validator
python main.py "frames/frame (1).png"
```

The report is saved to `output/tm_validation_report.json`. The frames are in `frames/` and
the two templates are in `templates/`.

I measure avatar coverage over the whole `user_image.jpg` image. There is a setting at the
top of `tm_validateAndTag.py` (`AVATAR_AREA_MODE`) if you want it to use only the avatar circle.