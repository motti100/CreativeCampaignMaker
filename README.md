# Creative Campaign Maker

Marketing campaign generator using Bria FIBO VLM.

## What it does

Takes text descriptions and generates professional marketing images. Uses Bria's FIBO API to expand simple prompts into detailed structured prompts, then generates images.

```
Input:  "coffee shop, morning light, 4 scenes"
Output: 4 professional marketing images
```

## Setup

```bash
git clone https://github.com/motti100/creative-campaign-maker.git
cd creative-campaign-maker
pip install -r requirements.txt
cp .env.example .env
# Add your Bria API key to .env
python CreativeCampaignMaker.py
```

Get API key: https://bria.ai/

## Bria FIBO Integration

Uses two endpoints:

**1. Structured Prompt Generation** - `/v2/structured_prompt/generate`

```python
response = requests.post(
    "https://engine.prod.bria-api.com/v2/structured_prompt/generate",
    headers={"api_token": BRIA_API_KEY},
    json={"prompt": "robot in park", "sync": True}
)
structured = json.loads(response.json()["result"]["structured_prompt"])
```

Converts "robot in park" into 500+ lines of:
- Object descriptions
- Lighting specs (direction, shadows, conditions)
- Camera settings (focal length, angle, DOF)
- Composition rules
- Color schemes

**2. Image Generation** - `/v2/image/generate`

```python
response = requests.post(
    "https://engine.prod.bria-api.com/v2/image/generate",
    headers={"api_token": BRIA_API_KEY},
    json={
        "structured_prompt": json.dumps(structured),
        "guidance_scale": 5,
        "steps_num": 50,
        "seed": 123456,
        "aspect_ratio": "16:9",
        "sync": True
    }
)
```

**3. Refinements**

```python
# Original
base = generate_structured("smartphone on desk")

# Refinement - only changes what you ask for
refined = generate_structured(
    "change desk to marble",
    original_structured=base  # keeps phone details
)

# Chain them
refined2 = generate_structured(
    "add dramatic lighting",
    original_structured=refined  # keeps phone + marble
)
```

## Features

**Storyboard Generation**
- Enter 2-6 scene descriptions
- Each scene gets expanded via FIBO VLM
- Optional reference image for consistent style
- Export as TXT or JSON

**Parameter Exploration**
- Vary lighting, camera angles, or style
- Same seed = only selected parameter changes
- Batch generation with different seeds

**Refinement Studio**
- Select any generated scene
- Make changes with natural language
- Chain multiple refinements
- Each refinement builds on previous

**Prompt Library**
- Save structured prompts
- Reuse successful templates

**Video Export**
- Convert frames to 4-second videos
- Uses Stable Video Diffusion

## Requirements

- Python 3.8+
- Bria API key
- 2GB RAM

## Configuration

`.env` file:
```
BRIA_API_KEY=your_key
BRIA_BASE_URL=https://engine.prod.bria-api.com
REPLICATE_API_TOKEN=your_token  # optional, for video
```

## Project Structure

```
creative-campaign-maker/
├── CreativeCampaignMaker.py    # Main app
├── requirements.txt
├── .env.example
├── README.md
├── docs/
│   └── API_INTEGRATION.md
└── examples/
    └── structured_prompts/
```

## How to Use

**Quick start:**
1. Load "Premium Coffee" template
2. Click Generate Storyboard
3. Wait 2-3 minutes

**Custom campaign:**
```
Scene 1: Office building at sunset
Scene 2: Laptop on desk
Scene 3: Team meeting
Scene 4: Handshake
```
Set aspect ratio to 16:9, seed to 123456, generate.

**Refinement:**
1. Generate "smartphone on desk"
2. Refine: "marble surface" 
3. Refine: "dramatic lighting"
4. Done - all changes preserved

## Troubleshooting

**Invalid API Key**
```bash
cat .env | grep BRIA_API_KEY
# No quotes, no spaces
```

**Gateway Timeout**
Normal on first request. Auto-retries. Takes 20-30 seconds.

**Module errors**
```bash
pip install -r requirements.txt --force-reinstall
```

## Documentation

See `docs/API_INTEGRATION.md` for detailed API docs.

## Why Bria FIBO

- Commercial-safe (licensed for commercial use)
- VLM understands marketing context
- Iterative refinements without regenerating
- Professional output quality

## License

MIT - See LICENSE

Images subject to Bria AI license (commercial use allowed).

---

**Bria FIBO Hackathon 2025**

Developer: Martin Tobias
