# Creative Campaign Maker

Multi-scene storyboard generator for marketing campaigns using Bria FIBO VLM.

## What it does

Enter a campaign brief with multiple scenes and generate professional images for each scene. Bria's FIBO VLM automatically expands each scene description into detailed structured prompts with professional photography specifications (lighting, camera angles, composition), then generates the images.

**Example Input:**
```
Scene 1: Wide establishing shot of artisan coffee roastery at golden hour, warm amber light streaming through industrial windows, 24mm lens
Scene 2: Medium shot of barista's hands pouring espresso, soft studio lighting with dramatic side rim light, 50mm lens
Scene 3: Intimate close-up of coffee cup on rustic table, steam rising with golden backlight, 85mm macro lens
Scene 4: Lifestyle shot of person enjoying coffee by window, natural morning light, 35mm lens
```

**Output:** 4 professional marketing images matching each scene description.

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

The app has 6 tabs:

### 1. Storyboard Tab - Generate Multi-Scene Campaigns

**Quick start with templates:**
1. Click "Premium Coffee" or "Tech Launch" or "Corporate" to load a template
2. Choose aspect ratio (16:9, 1:1, 9:16, etc.)
3. Set seed (optional - leave blank for random)
4. Select negative prompt preset
5. Click "Generate Storyboard"
6. Wait 2-3 minutes for all scenes

**Custom campaign:**
1. Enter your scene descriptions in the Campaign Brief box
2. One scene per line or paragraph
3. Include details like lighting, camera angles, lens focal length
4. Set aspect ratio and seed
5. Click "Generate Storyboard"

**Reference Image (optional):**
- Upload a reference image to influence all scenes with a specific style
- Useful for maintaining brand consistency across scenes

**Export options:**
- Export Brief (TXT) - saves your scene descriptions
- Export Structured (JSON) - saves the FIBO VLM generated structured prompts

### 2. Scene Variation Tab - Explore Parameters

Systematically explore lighting, camera angles, or style variations while keeping everything else constant.

**How to use:**
1. Click "Load Storyboard Scenes"
2. Select a scene from dropdown
3. Choose one parameter to vary:
   - Lighting Conditions
   - Camera Angles
   - Style & Mood
4. Click "Generate 5 Variations"
5. System locks all other parameters using same seed
6. Get 5 versions showing only that parameter changing

**Batch Variations:**
- Generate 2-8 variations with same prompt, different seeds
- Set number of variations
- Set base seed (optional)
- Click "Generate Batch"

### 3. Refinement Tab - Targeted Editing

Make specific changes without regenerating. Chained refinements preserve previous edits.

**How to use:**
1. Select image source: Storyboard Scenes or Parameter Variations
2. Click "Load Images"
3. Choose a scene from dropdown
4. Enter what you want to change (e.g., "remove people")
5. Set strength (0.1-1.0)
6. Keep same seed checkbox (recommended)
7. Click "Refine Image"
8. See Before & After comparison

**Chaining refinements:**
- Each refinement builds on the previous one
- Example: "remove people" → "add dramatic lighting" → "change to sunset"
- All changes are preserved

**Export options:**
- Marketing Document (HTML) - shows all refinements with before/after
- Export Workflow (JSON) - saves complete refinement chain
- Clear History - start fresh

### 4. Prompt Library Tab - Save & Reuse

Save successful structured prompts for reuse.

1. Enter a name for your prompt
2. Paste structured prompt JSON
3. Click "Save to Library"
4. Later: select from dropdown and click "Load from Library"

### 5. Video Export Tab - Animate Frames

Convert static frames to 4-second videos using Stable Video Diffusion.

1. Click "Load Storyboard"
2. Select a scene
3. Adjust motion amount slider (1-255)
4. Click "Generate AI Video"
5. Wait 1-2 minutes
6. Download video

### 6. Analytics Tab - Track Usage

View session metrics:
- Scenes Generated
- Videos Created
- AI Frames
- API Calls
- Session time
- Deterministic rate

Click "Refresh Metrics" to update.

## Bria FIBO Integration

The app uses Bria's FIBO API:

**1. Structured Prompt Generation** - `/v2/structured_prompt/generate`

For each scene, converts simple text into detailed structured prompts:

```python
response = requests.post(
    "https://engine.prod.bria-api.com/v2/structured_prompt/generate",
    headers={"api_token": BRIA_API_KEY},
    json={
        "prompt": "Wide shot of coffee roastery at golden hour",
        "sync": True,
        "ip_signal": True
    }
)
structured = json.loads(response.json()["result"]["structured_prompt"])
```

This generates:
- Object descriptions and placement
- Lighting setup (conditions, direction, shadows)
- Camera settings (focal length, angle, depth of field)
- Composition and framing rules
- Color schemes and mood
- Aesthetic preferences

**2. Image Generation** - `/v2/image/generate`

Generates images from structured prompts:

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
image_url = response.json()["result"]["image_url"]
```

**3. Refinements (Chained)**

Makes targeted changes while preserving everything else:

```python
# Original scene
base_structured = {...}  # Generated from "coffee roastery"

# First refinement
refined_1 = client.generate_structured_prompt(
    prompt="remove people",
    original_structured=base_structured  # Keeps all other details
)

# Second refinement builds on first
refined_2 = client.generate_structured_prompt(
    prompt="add dramatic lighting",
    original_structured=refined_1  # Keeps "no people" + other details
)

# Result: Coffee roastery without people, with dramatic lighting
```

**4. Reference Images (IP Adapter)**

Maintain consistent style across all scenes:

```python
# Convert reference image to base64
ref_img_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Generate with reference
payload = {
    "structured_prompt": json.dumps(structured),
    "images": [ref_img_base64],
    "image_weight": 0.8,  # Reference strength
    "ip_signal": True,
    ...
}
```

## Configuration

Your `.env` file:
```
BRIA_API_KEY=your_key_here
BRIA_BASE_URL=https://engine.prod.bria-api.com
REPLICATE_API_TOKEN=your_token_here
```

`REPLICATE_API_TOKEN` is optional - only needed for video generation.

## Built-in Templates

**Premium Coffee** (4 scenes)
- Scene 1: Wide shot of artisan coffee roastery at golden hour
- Scene 2: Medium shot of barista's hands pouring espresso
- Scene 3: Intimate close-up of coffee cup on rustic table
- Scene 4: Lifestyle shot of person enjoying coffee by window

**Tech Launch** (4 scenes)
- Scene 1: Dramatic low-angle of smartphone emerging from shadow
- Scene 2: Dynamic overhead view on minimalist surface
- Scene 3: Extreme macro of screen showing vibrant interface
- Scene 4: Lifestyle shot of hand holding device against urban bokeh

**Corporate** (4 scenes)
- Scene 1: Wide cinematic shot of modern glass skyscraper at blue hour
- Scene 2: Medium shot of diverse team collaborating in conference room
- Scene 3: Close-up of hands over laptop showing data
- Scene 4: Environmental portrait in modern office

## Troubleshooting

**Invalid API Key**
```bash
cat .env | grep BRIA_API_KEY
# Make sure no quotes, no spaces
```

**Gateway Timeout (504)**
Normal on first request. App auto-retries. Takes 20-30 seconds.

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
First scene: 30-60 seconds (model loads on server)
Subsequent scenes: 10-20 seconds each

## Why Bria FIBO

- **Commercial-safe**: All generated images licensed for commercial use
- **VLM intelligence**: Understands photography and marketing context
- **Structured control**: Fine control over lighting, camera, composition
- **Iterative refinement**: Make targeted changes without regenerating
- **Professional quality**: Marketing-ready output
- **Consistent branding**: Reference images maintain style across scenes

## License

MIT

Generated images subject to Bria AI license (commercial use allowed).

---

**Bria FIBO Hackathon 2025**

Developer: Martin Tobias
