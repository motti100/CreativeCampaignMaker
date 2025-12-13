# Creative Campaign Maker

Storyboard generator for marketing campaigns using Bria FIBO VLM.

## What it does

Enter a campaign brief with multiple scenes, and the app generates professional images for each scene. Bria's FIBO VLM expands each scene description into detailed structured prompts, then generates the images.

**Example:**

Input this campaign brief:
```
Scene 1: Wide establishing shot of artisan coffee roastery at golden hour
Scene 2: Medium shot of barista's hands pouring espresso
Scene 3: Intimate close-up of coffee cup on rustic table
Scene 4: Lifestyle shot of person enjoying coffee by window
```

Output: 4 professional marketing images (one per scene)

## Requirements

**System:**
- Python 3.8 or higher
- 2GB RAM
- Internet connection

**API Key:**
- Bria API key (free at https://bria.ai/)

**Check Python version:**
```bash
python3 --version
# Should show Python 3.8.0 or higher
```

**Don't have Python 3.8+?**
- macOS: `brew install python3` or https://www.python.org/downloads/
- Linux: `sudo apt install python3` or `sudo yum install python3`
- Windows: https://www.python.org/downloads/

## Setup

```bash
git clone https://github.com/motti100/CreativeCampaignMaker.git
cd CreativeCampaignMaker
pip3 install -r requirements.txt
cp .env.example .env
# Edit .env and add your Bria API key
python3 CreativeCampaignMaker.py
```

Opens at: http://localhost:7861

**Get Bria API key:**
1. Go to https://bria.ai/
2. Sign up (free)
3. Navigate to API section
4. Copy your API key
5. Paste into `.env` file (replace `your_bria_api_key_here`)

## How to Use

### Storyboard Tab

1. Enter your campaign brief - multiple scenes, one per line or paragraph
2. Choose aspect ratio (16:9, 1:1, 9:16, etc.)
3. Set seed (optional - leave blank for random)
4. Select negative prompt preset
5. Click "Generate Storyboard"
6. Wait 2-3 minutes for all scenes to generate

**Built-in Templates:**
- Premium Coffee (4 scenes)
- Tech Launch (4 scenes)
- Corporate (4 scenes)

Click any template to load it, then click Generate.

### Scene Variation Tab

1. Load scenes from your storyboard
2. Select a scene
3. Choose what to vary: Lighting / Camera Angles / Style
4. Generate 5 variations
5. Or use batch mode to generate multiple versions with different seeds

### Refinement Tab

1. Select a scene from storyboard or variations
2. Enter what you want to change (e.g., "add dramatic lighting")
3. Click Refine
4. Changes build on previous refinements

### Prompt Library Tab

Save your best structured prompts for reuse.

### Video Export Tab

Convert static frames to 4-second videos using Stable Video Diffusion.

## Bria FIBO Integration

The app uses Bria's FIBO API in two ways:

**1. Structured Prompt Generation** - `/v2/structured_prompt/generate`

For each scene in your brief, the app calls:

```python
response = requests.post(
    "https://engine.prod.bria-api.com/v2/structured_prompt/generate",
    headers={"api_token": BRIA_API_KEY},
    json={"prompt": scene_text, "sync": True, "ip_signal": True}
)
structured = json.loads(response.json()["result"]["structured_prompt"])
```

This converts:
```
"Wide shot of coffee roastery at golden hour"
```

Into detailed JSON with:
- Object descriptions and placement
- Lighting setup (direction, shadows, conditions)
- Camera settings (focal length, angle, depth of field)
- Composition rules
- Color schemes and mood

**2. Image Generation** - `/v2/image/generate`

Then generates the actual image:

```python
response = requests.post(
    "https://engine.prod.bria-api.com/v2/image/generate",
    headers={"api_token": BRIA_API_KEY},
    json={
        "structured_prompt": json.dumps(structured),
        "negative_prompt": "blurry, low quality, distorted",
        "guidance_scale": 5,
        "steps_num": 50,
        "seed": 123456,
        "aspect_ratio": "16:9",
        "sync": True
    }
)
```

**3. Refinements**

When you refine a scene, the app sends the existing structured prompt back to FIBO VLM with your instruction:

```python
# You have a scene already generated
existing_structured = {...}  # Previous structured prompt

# You want to change something
refined = client.generate_structured_prompt(
    prompt="change lighting to dramatic side lighting",
    original_structured=existing_structured  # Preserves everything else
)

# Only lighting changes, all other details stay the same
```

This lets you make targeted changes without regenerating from scratch.

## Features

**Storyboard Generation**
- Enter 2-6 scene descriptions in a campaign brief
- Each scene gets expanded via FIBO VLM
- All scenes generated with consistent settings
- Optional reference image for style consistency
- Export brief as TXT or structured prompts as JSON

**Parameter Exploration**
- Take any generated scene
- Vary one parameter: Lighting / Camera Angles / Style & Mood
- Generates 5 variations showing only that parameter changing
- Same seed = pure comparison
- Batch mode generates 2-8 variations with different seeds

**Refinement Studio**
- Select any scene
- Make changes with natural language
- Each refinement builds on previous changes
- Visual diff shows what changed
- Export refinement workflow

**Prompt Library**
- Save successful structured prompts
- Load and reuse for similar projects

**Video Export**
- Convert any frame to 4-second video
- Uses Stable Video Diffusion
- Motion control slider

**Analytics**
- Track scenes generated
- Monitor API usage
- Session metrics

## Configuration

Your `.env` file:
```
BRIA_API_KEY=your_key_here
BRIA_BASE_URL=https://engine.prod.bria-api.com
REPLICATE_API_TOKEN=your_token_here
```

The `REPLICATE_API_TOKEN` is optional - only needed for video generation.

## Troubleshooting

**Invalid API Key**
```bash
cat .env | grep BRIA_API_KEY
# Make sure no quotes, no spaces
```

**Gateway Timeout (504)**
Normal on first request. The app auto-retries. Takes 20-30 seconds.

**Module errors**
```bash
pip3 install -r requirements.txt --force-reinstall
```

**SSL Warning (macOS)**
The urllib3 LibreSSL warning is normal. App works fine.

**Port already in use**
```bash
lsof -i :7861
kill -9 <PID>
```

**Generation is slow**
First scene takes 30-60 seconds (model loads on server). After that, each scene takes 10-20 seconds.

## Why Bria FIBO

- **Commercial-safe**: All generated images licensed for commercial use
- **VLM intelligence**: Understands photography and marketing context
- **Structured control**: Detailed control over lighting, camera, composition
- **Iterative refinement**: Make changes without regenerating everything
- **Professional quality**: Marketing-ready output

## License

MIT

Generated images subject to Bria AI license (commercial use allowed).

---

**Bria FIBO Hackathon 2025**

Developer: Martin Tobias
