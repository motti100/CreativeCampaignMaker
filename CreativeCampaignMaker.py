#!/usr/bin/env python3
"""
Creative Campaign Maker - Complete FIBO VLM Integration
================================================

"""

import os, re
import io
import json
import time
import tempfile
import subprocess
import copy
import base64
import re
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, List
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import requests
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

# Gradio client boolean-schema guard
try:
    import gradio_client.utils as _gcu
    _orig_get_type = _gcu.get_type
    _orig_json_to_py = _gcu._json_schema_to_python_type

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return None
        return _orig_get_type(schema)

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "object" if schema else "never"
        if isinstance(schema, dict) and isinstance(schema.get("additionalProperties"), bool):
            ap = schema["additionalProperties"]
            return "Dict[str, Any]" if ap else "Dict[str, None]"
        return _orig_json_to_py(schema, defs)

    _gcu.get_type = _safe_get_type
    _gcu._json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_VERSION = "3.0.0"
APP_NAME = "Creative Campaign Maker"

# CUSTOM THEME
agency_theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
    button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #5568d3 0%, #63408a 100%)",
    button_primary_text_color="white",
    button_shadow="0 10px 25px -5px rgba(102, 126, 234, 0.4)",
    button_large_radius="12px",
    input_background_fill="white",
    input_border_color="#e2e8f0",
    input_border_width="2px",
    input_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
    input_radius="10px",
    block_background_fill="white",
    block_shadow="0 10px 30px -5px rgba(0, 0, 0, 0.1)",
    block_radius="16px",
)

# CUSTOM CSS
custom_css = """
/* Dropdown Fix */
.dropdown, .dropdown-container, .gr-dropdown, .gr-box:has(.gr-dropdown), .block:has(.gr-dropdown) {
    transform: none !important;
    filter: none !important;
    perspective: none !important;
    overflow: visible !important;
    contain: none !important;
    will-change: auto !important;
}

.block:has(.gr-dropdown):hover {
    transform: none !important;
}

/* Global Enhancements */
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Card Hover */
.block {
    transition: box-shadow 0.3s ease, border-color 0.3s ease !important;
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
}

.block:hover {
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.15) !important;
    border-color: rgba(102, 126, 234, 0.3) !important;
}

/* Buttons */
button[value*="Load"], button[value*="Generate"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35) !important;
}

button[value*="Load"]:hover, button[value*="Generate"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(102, 126, 234, 0.45) !important;
}

/* Hide Gradio Footer */
footer {
    display: none !important;
}
"""

# Session metrics
session_metrics = {
    "scenes_generated": 0,
    "videos_created": 0,
    "frames_generated": 0,
    "api_calls": 0,
    "start_time": time.time(),
    "deterministic_rate": 100.0,
}

# Global state
storyboard_images: List[Image.Image] = []
storyboard_scenes: List[Dict[str, Any]] = []
last_brief_text: str = ""
disentanglement_images: List[Image.Image] = [] 
disentanglement_metadata: Dict[str, Any] = {}
prompt_library: Dict[str, Dict] = {}

# NEGATIVE PROMPT PRESETS
NEGATIVE_PRESETS = {
    "Clean (Default)": "deformed, extra limbs, oversaturated, artifacts, watermark, text",
    "Photography Only": "painting, drawing, illustration, 3d render, cartoon, anime, sketch",
    "Product Shot": "people, hands, text, watermark, low quality, blurry, background clutter",
    "Portrait": "extra limbs, deformed face, asymmetric eyes, poor anatomy, distorted features",
    "Architectural": "people, vehicles, distortion, unrealistic lighting, poor perspective",
    "Minimal": "artifacts, watermark, text"
}

DEFAULT_NEGATIVE_PROMPT = NEGATIVE_PRESETS["Clean (Default)"]

# Campaign templates
CAMPAIGN_PRESETS = {
    "Premium Coffee": {
        "brief": """Scene 1: Wide establishing shot of artisan coffee roastery at golden hour, warm amber light streaming through industrial windows, 24mm lens
Scene 2: Medium shot of barista's hands pouring espresso, soft studio lighting with dramatic side rim light, 50mm lens
Scene 3: Intimate close-up of coffee cup on rustic table, steam rising with golden backlight, 85mm macro lens
Scene 4: Lifestyle shot of person enjoying coffee by window, natural morning light, 35mm lens""",
        "aspect": "16:9",
        "seed": 123456,
        "animation_rec": "camera_push",
        "industry": "Food & Beverage"
    },
    "Tech Launch": {
        "brief": """Scene 1: Dramatic low-angle of smartphone emerging from shadow, sharp side lighting, 35mm lens
Scene 2: Dynamic overhead view on minimalist surface, controlled studio lighting, 50mm lens
Scene 3: Extreme macro of screen showing vibrant interface, soft diffused lighting, 85mm lens
Scene 4: Lifestyle shot of hand holding device against urban bokeh, natural daylight, 50mm lens""",
        "aspect": "9:16",
        "seed": 234567,
        "animation_rec": "composite",
        "industry": "Technology"
    },
    "Corporate": {
        "brief": """Scene 1: Wide cinematic shot of modern glass skyscraper at blue hour, 24mm lens
Scene 2: Medium shot of diverse team collaborating in conference room, natural light, 35mm lens
Scene 3: Close-up of hands over laptop showing data, soft natural light, 50mm lens
Scene 4: Environmental portrait in modern office, large windows with city view, 35mm lens""",
        "aspect": "16:9",
        "seed": 345678,
        "animation_rec": "lighting_transition",
        "industry": "Corporate"
    },
}

# Default structured prompt schema
DEFAULT_STRUCTURED_SCHEMA = {
    "short_description": "",
    "objects": [],
    "background_setting": "",
    "lighting": {"conditions": "", "direction": "", "shadows": ""},
    "aesthetics": {
        "composition": "",
        "color_scheme": "",
        "mood_atmosphere": "",
    },
    "photographic_characteristics": {
        "depth_of_field": "",
        "focus": "",
        "camera_angle": "Eye-level",
        "lens_focal_length": "35mm",
    },
    "style_medium": "photograph",
    "artistic_style": "realistic, detailed, vibrant",
}

# ============================================================================
# VIDEO GENERATION
# ============================================================================

def generate_video_from_frame_hf(image: Image.Image, motion_bucket_id: int = 127, fps: int = 6) -> Optional[str]:
    """
    Generate video using Stable Video Diffusion via Hugging Face Inference API (FREE)
    Tries multiple models in sequence until one succeeds.
    
    Args:
        image: PIL Image to animate
        motion_bucket_id: Motion amount (1-255, default 127)
        fps: Frames per second (default 6)
    
    Returns:
        Path to generated video file or None
    """
    
    # List of models to try (in order of preference)
    models_to_try = [
       "https://router.huggingface.co/hf-inference/models/damo-vilab/text-to-video-ms-1.7b"
    ]
    
    # Get HF token from environment (optional - works without it but may be slower)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")
    
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    # Convert image to base64
    buffered = io.BytesIO()
    # Resize to 1024x576 (SVD's preferred resolution)
    img_resized = image.resize((1024, 576), Image.Resampling.LANCZOS)
    img_resized.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Try each model in sequence
    for API_URL in models_to_try:
        model_name = API_URL.split('/')[-1]
        
        try:
            print(f"🎬 Generating video with {model_name} (Hugging Face - FREE)")
            
            # Call API
            response = requests.post(
                API_URL,
                headers=headers,
                json={
                    "inputs": img_base64,
                    "parameters": {
                        "motion_bucket_id": motion_bucket_id,
                        "fps": fps,
                        "num_frames": 25  # 25 frames at 6fps = ~4 seconds
                    }
                },
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 503:
                # Model is loading, wait and retry once
                print(f"⏳ Model {model_name} loading, waiting 20 seconds...")
                time.sleep(20)
                response = requests.post(
                    API_URL,
                    headers=headers,
                    json={
                        "inputs": img_base64,
                        "parameters": {
                            "motion_bucket_id": motion_bucket_id,
                            "fps": fps,
                            "num_frames": 25
                        }
                    },
                    timeout=300
                )
            
            if response.status_code == 200:
                # Save video
                temp_file = Path(tempfile.gettempdir()) / f"svd_video_{int(time.time())}.mp4"
                with open(temp_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅Video generated with {model_name}: {temp_file}")
                return str(temp_file)
            else:
                print(f"{model_name} error: {response.status_code} - {response.text}")
                continue  # Try next model
                
        except Exception as e:
            print(f" Error with {model_name}: {e}")
            continue  # Try next model
    
    # If all models failed
    print("All Hugging Face models failed")
    return None

def generate_video_from_frame_replicate(image: Image.Image, motion_amount: int = 127) -> Optional[str]:
    """
    Generate video using Stable Video Diffusion via Replicate API (FREE CREDITS)
    
    Requires: pip install replicate
    Set environment: REPLICATE_API_TOKEN
    """
    try:
        import replicate
        
        print("🎬 Generating video with Stable Video Diffusion (Replicate)")
        
        # Convert image to base64 data URL
        buffered = io.BytesIO()
        img_resized = image.resize((1024, 576), Image.Resampling.LANCZOS)
        img_resized.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{img_base64}"
        
        # Run Stable Video Diffusion
        output = replicate.run(
            "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
            input={
                "input_image": data_url,
                "motion_bucket_id": motion_amount,
                "fps": 6,
                "frames_per_second": 6
            }
        )
        
        if output:
            # Download the video - convert FileOutput to string URL
            video_url = str(output)  # ✅ This is the fix
            video_response = requests.get(video_url)
            
            temp_file = Path(tempfile.gettempdir()) / f"svd_video_{int(time.time())}.mp4"
            with open(temp_file, 'wb') as f:
                f.write(video_response.content)
            
            print(f"✅ Video generated: {temp_file}")
            return str(temp_file)
        else:
            print("❌ Replicate returned no output")
            return None
            
    except ImportError:
        print("❌ Replicate not installed. Run: pip install replicate")
        return None
    except Exception as e:
        print(f"❌ Error generating video with Replicate: {e}")
        return None

def generate_video_smart(image: Image.Image, motion_amount: int = 127, method: str = "replicate") -> Optional[str]:
    """
    Smart video generation - tries free methods first
    
    Methods:
    - "huggingface": Use HF Inference API (free, no signup needed but may be slow)
    - "replicate": Use Replicate (requires API key, has free credits)
    - "auto": Try HF first, fallback to Replicate
    """
    
    
    if method == "replicate" or method == "auto":
        print(" Trying Replicate API...")
        result = generate_video_from_frame_replicate(image, motion_amount=motion_amount)
        if result:
            return result
    
    return None



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def update_metrics(scenes=0, videos=0, frames=0, api_calls=0):
 
    session_metrics["scenes_generated"] += scenes
    session_metrics["videos_created"] += videos
    session_metrics["frames_generated"] += frames
    session_metrics["api_calls"] += api_calls

def get_metrics_html():

    elapsed = time.time() - session_metrics["start_time"]
    elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
    
    return f"""
    <div style='background: linear-gradient(135deg, #1a1f2e, #2a3f5f); padding: 32px; border-radius: 16px; 
                box-shadow: 0 8px 32px rgba(0,0,0,0.3); border: 2px solid rgba(76, 175, 80, 0.3);'>
        <h2 style='margin: 0 0 24px; text-align: center; color: #4CAF50;'>Session Analytics</h2>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;'>
            <div style='text-align: center; padding: 20px; background: rgba(76, 175, 80, 0.1); border-radius: 12px;'>
                <div style='font-size: 42px; font-weight: 800; color: #4CAF50;'>{session_metrics["scenes_generated"]}</div>
                <div style='font-size: 13px; color: #b4b8c0; margin-top: 6px;'>Scenes Generated</div>
            </div>
            <div style='text-align: center; padding: 20px; background: rgba(33, 150, 243, 0.1); border-radius: 12px;'>
                <div style='font-size: 42px; font-weight: 800; color: #2196F3;'>{session_metrics["videos_created"]}</div>
                <div style='font-size: 13px; color: #b4b8c0; margin-top: 6px;'>Videos Created</div>
            </div>
            <div style='text-align: center; padding: 20px; background: rgba(156, 39, 176, 0.1); border-radius: 12px;'>
                <div style='font-size: 42px; font-weight: 800; color: #9C27B0;'>{session_metrics["frames_generated"]}</div>
                <div style='font-size: 13px; color: #b4b8c0; margin-top: 6px;'>AI Frames</div>
            </div>
            <div style='text-align: center; padding: 20px; background: rgba(255, 152, 0, 0.1); border-radius: 12px;'>
                <div style='font-size: 42px; font-weight: 800; color: #FF9800;'>{session_metrics["api_calls"]}</div>
                <div style='font-size: 13px; color: #b4b8c0; margin-top: 6px;'>API Calls</div>
            </div>
        </div>
        <div style='margin-top: 16px; padding: 16px; background: rgba(76, 175, 80, 0.05); border-radius: 12px; text-align: center;'>
            <div style='font-size: 16px; color: #4CAF50; font-weight: 600;'>
                {session_metrics["deterministic_rate"]:.0f}% Deterministic | {elapsed_str} session time
            </div>
        </div>
    </div>
    """

def dimensions_from_aspect(aspect: Optional[str]) -> Tuple[int, int]:
  
    mapping = {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1024, 768),
        "3:4": (768, 1024),
        "21:9": (1280, 548),
        "9:21": (548, 1280),
    }
    return mapping.get(aspect or "1:1", (1024, 1024))

def parse_brief_to_scenes(brief: str) -> List[Dict[str, Any]]:

    lines = [ln.strip(" •-\t") for ln in brief.splitlines() if ln.strip()]
    scenes = []
    for ln in lines:
        if ln.lower().startswith("a ") and "scene" in ln.lower():
            continue
        cleaned = ln.lstrip("0123456789. )").strip()
        if cleaned:
            scenes.append(cleaned)
    return [{"text": s} for s in scenes[:6]]

def scene_to_structured(scene_text: str) -> Dict[str, Any]:
   
    txt = scene_text.lower()
    
    # Determine lens and camera settings
    if any(k in txt for k in ["close-up", "macro", "intimate"]):
        focal = "Portrait lens (e.g., 85mm-100mm)"
        dof = "Shallow, with the subject in sharp focus and background softly blurred (bokeh)."
        angle = "Eye-level or slightly elevated"
    elif any(k in txt for k in ["wide", "establishing", "aerial", "panoramic"]):
        focal = "Wide-angle lens (e.g., 24mm-35mm)"
        dof = "Deep, with most elements in focus from foreground to background."
        angle = "Eye-level or elevated perspective"
    else:
        focal = "Standard lens (e.g., 50mm)"
        dof = "Medium depth of field, with primary subject in focus."
        angle = "Eye-level"
    
    # Determine lighting
    if any(k in txt for k in ["studio", "controlled"]):
        light_cond = "Studio lighting with softboxes and controlled environment."
        light_dir = "Diffused from multiple angles with key light from front-left."
        shadows = "Soft, controlled shadows that add dimension without harshness."
    elif any(k in txt for k in ["golden hour", "sunset", "sunrise"]):
        light_cond = "Golden hour natural light, warm and soft."
        light_dir = "Low-angle sunlight from the side, creating warm highlights."
        shadows = "Long, soft shadows with warm tones."
    elif any(k in txt for k in ["night", "evening", "moonlight"]):
        light_cond = "Cool moonlight or artificial night lighting."
        light_dir = "Softly diffused from above or ambient sources."
        shadows = "Deep shadows with high contrast in darker areas."
    else:
        light_cond = "Natural daylight, bright and even."
        light_dir = "Diffused natural light from above and sides."
        shadows = "Soft shadows that add depth without obscuring details."
    
    # Determine composition and mood
    if "dramatic" in txt or "dynamic" in txt:
        comp = "Dynamic composition with strong leading lines and visual tension."
        mood = "Dramatic, energetic, and bold."
    elif "minimalist" in txt or "clean" in txt:
        comp = "Minimalist composition with negative space and clear focal point."
        mood = "Clean, modern, and serene."
    else:
        comp = "Balanced composition with clear visual hierarchy."
        mood = "Professional, polished, and engaging."
    
    # Build structured prompt
    structured = {
        "short_description": scene_text.strip(),
        "objects": [
            {
                "description": "Primary subject of the scene, detailed and prominent.",
                "location": "center",
                "relationship": "Main focal point of the composition.",
                "relative_size": "large within frame",
                "shape_and_color": "Detailed with natural or vibrant colors as appropriate.",
                "texture": "High-quality texture appropriate to the subject.",
                "appearance_details": "Sharp, detailed, and visually appealing.",
                "orientation": "Optimally positioned for visual impact."
            }
        ],
        "background_setting": "Appropriate background that complements the subject without distraction, with appropriate depth and context.",
        "lighting": {
            "conditions": light_cond,
            "direction": light_dir,
            "shadows": shadows
        },
        "aesthetics": {
            "composition": comp,
            "color_scheme": "Natural and harmonious color palette appropriate to the scene.",
            "mood_atmosphere": mood,
            "preference_score": "very high",
            "aesthetic_score": "very high"
        },
        "photographic_characteristics": {
            "depth_of_field": dof,
            "focus": "Sharp focus on primary subject with appropriate falloff.",
            "camera_angle": angle,
            "lens_focal_length": focal
        },
        "style_medium": "photograph",
        "context": "Professional commercial photography for marketing and advertising purposes.",
        "artistic_style": "realistic, detailed, vibrant, professional"
    }
    
    return structured

# ============================================================================
# FIBO ENHANCEMENTS
# ============================================================================

def upsample_simple_prompt(simple_prompt: str) -> Dict:

    try:
        client = get_bria_client(verbose=True)
        print(f" Upsampling prompt: {simple_prompt}")
        
        structured = client.generate_structured_prompt(
            prompt=simple_prompt,
            original_structured=None
        )
        
        if structured:
            print(f"✓ VLM expanded prompt to {len(json.dumps(structured))} characters")
            return structured
        else:
            print("x VLM failed to expand prompt")
            return scene_to_structured(simple_prompt)
    except Exception as e:
        print(f"Error upsampling prompt: {e}")
        return scene_to_structured(simple_prompt)

def generate_batch_variations(structured: Dict, seed: int, num_images: int, width: int, height: int) -> List[Image.Image]:

    images = []
    aspect = "16:9" if width > height else "9:16" if height > width else "1:1"
    
    for i in range(num_images):
        print(f"Generating variation {i+1}/{num_images} with seed {seed + i}")
        
        payload = build_payload(
            structured_prompt=structured,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            guidance_scale=5,
            steps_num=50,
            seed=seed + i,
            aspect_ratio=aspect,
            sync=True
        )
        
        client = get_bria_client(verbose=False)
        img, _ = client.generate(**payload)
        
        if img:
            images.append(img)
            update_metrics(scenes=1)
    
    return images

def show_structured_diff(original: Dict, refined: Dict) -> str:
  
    import difflib
    
    orig_str = json.dumps(original, indent=2, ensure_ascii=False)
    ref_str = json.dumps(refined, indent=2, ensure_ascii=False)
    
    diff = list(difflib.unified_diff(
        orig_str.splitlines(keepends=True),
        ref_str.splitlines(keepends=True),
        fromfile='Original',
        tofile='Refined',
        lineterm=''
    ))
    
    if not diff:
        return "<div style='padding: 20px; background: #f5f5f5; border-radius: 8px;'>No changes detected</div>"
    
    # Convert to HTML with color coding
    html = "<pre style='background: #f5f5f5; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; line-height: 1.6;'>"
    for line in diff[:100]:  # Limit to first 100 lines
        if line.startswith('+') and not line.startswith('+++'):
            html += f"<span style='color: #2e7d32; background: #c8e6c9; display: block;'>{line}</span>"
        elif line.startswith('-') and not line.startswith('---'):
            html += f"<span style='color: #c62828; background: #ffcdd2; display: block;'>{line}</span>"
        elif line.startswith('@@'):
            html += f"<span style='color: #1976d2; font-weight: bold; display: block;'>{line}</span>"
        else:
            html += line
    html += "</pre>"
    
    return html

def save_prompt_to_library(name: str, structured_json: str) -> tuple:

    global prompt_library
    
    if not name or not name.strip():
        return gr.update(), "<div style='background: #ffebee; padding: 12px; border-radius: 8px; color: #c62828;'>Please enter a name</div>"
    
    if not structured_json or not structured_json.strip():
        return gr.update(), "<div style='background: #ffebee; padding: 12px; border-radius: 8px; color: #c62828;'>No structured prompt to save</div>"
    
    try:
        structured = json.loads(structured_json)
        prompt_library[name.strip()] = structured
        
        msg = f"<div style='background: #e8f5e9; padding: 12px; border-radius: 8px; color: #2e7d32;'>✅ Saved '{name}' to library ({len(prompt_library)} total)</div>"
        return gr.update(choices=list(prompt_library.keys())), msg
    except json.JSONDecodeError as e:
        return gr.update(), f"<div style='background: #ffebee; padding: 12px; border-radius: 8px; color: #c62828;'>Invalid JSON: {str(e)}</div>"

def load_prompt_from_library(name: str) -> tuple:
    """Load structured prompt from library"""
    global prompt_library
    
    if not name or name not in prompt_library:
        return "", "<div style='background: #ffebee; padding: 12px; border-radius: 8px; color: #c62828;'>Prompt not found</div>"
    
    structured = prompt_library[name]
    json_str = json.dumps(structured, indent=2, ensure_ascii=False)
    msg = f"<div style='background: #e8f5e9; padding: 12px; border-radius: 8px; color: #2e7d32;'>✅ Loaded '{name}'</div>"
    
    return json_str, msg

def export_refinement_workflow(history: List[Dict]) -> tuple:
    """Export refinement workflow as JSON"""
    if not history:
        return None, "<div style='background: #ffebee; padding: 12px; border-radius: 8px;'>No refinements to export</div>"
    
    try:
        workflow = {
            "version": "1.0",
            "app": "Creative Campaign Maker",
            "timestamp": datetime.now().isoformat(),
            "refinements": [
                {
                    "step": i+1,
                    "instruction": h["instruction"],
                    "timestamp": h["timestamp"],
                    "seed": h.get("seed"),
                    "strength": h.get("strength"),
                    "changes": h.get("changes", "")
                }
                for i, h in enumerate(history)
            ]
        }
        
        temp_file = Path(tempfile.gettempdir()) / f"fibo_workflow_{int(time.time())}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        
        msg = f"""
        <div style='background: #e8f5e9; padding: 12px; border-radius: 8px;'>
            <strong>✅ Workflow Exported</strong><br>
            {len(history)} refinement steps<br>
            Download JSON file below
        </div>
        """
        
        return str(temp_file), msg
    except Exception as e:
        return None, f"<div style='background: #ffebee; padding: 12px; border-radius: 8px;'>Error: {str(e)}</div>"

# ============================================================================
# BRIA API CLIENT
# ============================================================================

class BriaClient:
    def __init__(self, base_url: str, api_key: str, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            "api_token": api_key,
            "Content-Type": "application/json",
        })

    def generate_structured_prompt(self, prompt: str, original_structured: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate or refine a structured prompt - completely generic via VLM
        """
        try:
            payload = {
                "prompt": prompt,
                "sync": True,
                "ip_signal": True
            }
            
            if original_structured:
                payload["structured_prompt"] = json.dumps(original_structured, ensure_ascii=False)
            
            if self.verbose:
                print(f"Calling {self.base_url}/v2/structured_prompt/generate")
            
            response = self.session.post(
                f"{self.base_url}/v2/structured_prompt/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            result = data.get("result", {})
            structured_str = result.get("structured_prompt")
            
            if structured_str:
                return json.loads(structured_str)
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"Structured prompt API Error: {e}")
            raise

    def generate(self, **payload) -> Tuple[Optional[Image.Image], Optional[Dict]]:

        max_retries = 3
        base_timeout = 120
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (attempt + 1)  # Increase timeout each retry
                print(f"API call attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)...")
                
                response = self.session.post(
                    f"{self.base_url}/v2/image/generate", 
                    json=payload, 
                    timeout=timeout
                )
                response.raise_for_status()
                data = response.json()
                
                result = data.get("result", {})
                if result.get("image_url"):
                    img_resp = self.session.get(result["image_url"], timeout=60)
                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    return img, result.get("structured_prompt")
                
                return None, None
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)  # Exponential backoff
                    print(f"⏱️ Timeout on attempt {attempt + 1}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"❌ Failed after {max_retries} attempts")
                    raise
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 504:
                    if attempt < max_retries - 1:
                        wait = 5 * (attempt + 1)
                        print(f"⏱️ Gateway timeout. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"❌ Gateway timeout after {max_retries} attempts")
                        raise
                else:
                    raise
                    
            except Exception as e:
                print(f"API Error: {e}")
                raise
        
        return None, None

def get_bria_client(verbose: bool = False):
    """Get configured Bria client"""
    base_url = os.environ.get("BRIA_BASE_URL", "https://engine.prod.bria-api.com")
    api_key = os.environ.get("BRIA_API_KEY") or os.environ.get("BRIA_API_TOKEN")
    if not api_key:
        raise EnvironmentError("Set BRIA_API_KEY environment variable")
    return BriaClient(base_url, api_key, verbose=verbose)

def build_payload(
    text_prompt: Optional[str] = None,
    structured_prompt: Optional[Dict[str, Any]] = None,
    images: Optional[List[str]] = None,
    negative_prompt: Optional[str] = None,
    guidance_scale: Optional[int] = None,
    steps_num: Optional[int] = None,
    seed: Optional[int] = None,
    sync: bool = False,
    ip_signal: bool = False,
    prompt_content_moderation: bool = False,
    visual_input_content_moderation: bool = False,
    visual_output_content_moderation: bool = False,
    aspect_ratio: Optional[str] = None,
    reference_strength: Optional[float] = None,
) -> Dict[str, Any]:
    """Build payload for Bria API"""
    structured_prompt_str = None
    if structured_prompt:
        structured_prompt_str = json.dumps(structured_prompt, ensure_ascii=False)

    payload = {
        "prompt": text_prompt if text_prompt else None,
        "structured_prompt": structured_prompt_str,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "steps_num": steps_num,
        "seed": seed,
        "sync": sync,
        "ip_signal": ip_signal if images else False,
        "prompt_content_moderation": prompt_content_moderation,
        "visual_input_content_moderation": visual_input_content_moderation,
        "visual_output_content_moderation": visual_output_content_moderation,
        "aspect_ratio": aspect_ratio,
    }
    
    if images:
        payload["images"] = images
        if reference_strength is not None:
            payload["image_weight"] = reference_strength

    return {k: v for k, v in payload.items() if v is not None}

def call_bria_and_return_image_and_payload(
    payload: Dict[str, Any],
    verbose: bool = False
) -> Tuple[Optional[Image.Image], str]:
    """Call Bria API and return image with metadata"""
    try:
        client = get_bria_client(verbose=verbose)
        img, result = client.generate(**payload)
        
        if img:
            md = f"Bria call successful"
            return img, md
        else:
            md = "No image returned from API"
            return None, md
            
    except requests.exceptions.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                err_text = e.response.text
            except Exception:
                err_text = "<no response body>"
            md = f"HTTP Error {e.response.status_code}: {err_text}"
        else:
            md = f"Request Error: {e}"
        return None, md
    except Exception as e:
        md = f"Unexpected error: {e}"
        return None, md

def call_bria_api_with_reference(
    scene_text: str, 
    structured: Dict, 
    seed: Optional[int],
    width: int, 
    height: int,
    reference_image: Optional[str] = None,
    reference_strength: Optional[float] = None,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
):
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            update_metrics(api_calls=1)
            aspect = "16:9" if width > height else "9:16" if height > width else "1:1"
            
            # Show progress
            yield f"<div style='background: #e3f2fd; padding: 20px; border-radius: 12px;'>🎨 Generating image... Attempt {attempt + 1}/{max_retries}</div>"

            payload = build_payload(
                text_prompt=scene_text,
                structured_prompt=structured,
                images=[reference_image] if reference_image else None,
                negative_prompt=negative_prompt,
                guidance_scale=5,
                aspect_ratio=aspect,
                steps_num=50,
                seed=seed,
                sync=True,
                ip_signal=True if reference_image else False,
                reference_strength=reference_strength,
            )

            client = get_bria_client(verbose=True)
            
            # Increase timeout with each attempt
            base_timeout = 120 * (attempt + 1)
            
            # Patch the client's session timeout for this call
            original_timeout = getattr(client.session, 'timeout', None)
            
            img, result = client.generate(**payload)

            if img:
                yield f"<div style='background: #e8f5e9; padding: 20px; border-radius: 12px;'>✅ Image generated successfully</div>"
                yield img, structured
                return
            else:
                if attempt < max_retries - 1:
                    yield f"<div style='background: #fff3e0; padding: 20px; border-radius: 12px;'>⚠️ No image returned. Retrying...</div>"
                    time.sleep(2)
                else:
                    yield f"<div style='background: #ffebee; padding: 20px; border-radius: 12px; color: #c62828;'>❌ No image returned after {max_retries} attempts</div>"
                    yield None, None
                    return

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                yield f"<div style='background: #fff3e0; padding: 20px; border-radius: 12px;'>⏱️ Request timeout. Retrying in {wait}s...</div>"
                time.sleep(wait)
            else:
                yield f"<div style='background: #ffebee; padding: 20px; border-radius: 12px; color: #c62828;'>❌ Failed after {max_retries} timeout attempts</div>"
                yield None, None
                return
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 504:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    yield f"<div style='background: #fff3e0; padding: 20px; border-radius: 12px;'>⏱️ Gateway timeout (504). Retrying in {wait}s...</div>"
                    time.sleep(wait)
                else:
                    yield f"<div style='background: #ffebee; padding: 20px; border-radius: 12px; color: #c62828;'>❌ Gateway timeout after {max_retries} attempts</div>"
                    yield None, None
                    return
            else:
                yield f"<div style='background: #ffebee; padding: 20px; border-radius: 12px; color: #c62828;'>❌ HTTP Error {e.response.status_code}: {str(e)}</div>"
                yield None, None
                return
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error in call_bria_api_with_reference: {error_msg}")
            import traceback
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                yield f"<div style='background: #fff3e0; padding: 20px; border-radius: 12px;'>⚠️ Error: {error_msg}. Retrying...</div>"
                time.sleep(2)
            else:
                yield f"<div style='background: #ffebee; padding: 20px; border-radius: 12px; color: #c62828;'>❌ Error after {max_retries} attempts: {error_msg}</div>"
                yield None, None
                return
    
    yield None, None

# ============================================================================
# GRADIO UI
# ============================================================================

def create_demo():
    """Create Gradio demo interface"""
    
    with gr.Blocks(title=f"{APP_NAME}", theme=agency_theme, css=custom_css) as demo:
        
        # Header
        gr.HTML(f"""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                            color: white;
                            padding: 10px;
                            border-radius: 6px;
                            margin-bottom: 15px;
                            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);;">
                    <h1 style="margin: 0; font-size: 48px; font-weight: 800; text-align: center;">{APP_NAME}</h1>
                    <p style="margin: 16px 0 0; font-size: 18px; text-align: center; opacity: 0.95;">
                        Complete FIBO VLM Integration • v{APP_VERSION}
                    </p>
                    <p style="margin: 8px 0 0; font-size: 14px; text-align: center; opacity: 0.85;">
                        Prompt Upsampling • Chained Refinements • Batch Generation • Prompt Library
                    </p>
                </div>
                """)
        
        with gr.Tabs():

# # TAB 0: Quick Start
#             with gr.Tab("⚡ Quick Start"):
#                 gr.HTML("""
#                 <div style='background: white; color: black; padding: 24px; border: 2px solid blpinkack; border-radius: 4px; margin-bottom: 24px;'>
#                     <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>Quick Generate</h2>
#                     <p style='margin: 0; font-size: 16px;'>
#                         Enter a simple prompt and FIBO VLM will expand it into a detailed structured prompt with professional lighting, composition, and camera settings.
#                     </p>
#                 </div>
#                 """)
                
#                 with gr.Row():
#                     with gr.Column():
#                         quick_prompt = gr.Textbox(
#                             label="Simple Prompt",
#                             placeholder="robot in park, golden hour lighting",
#                             lines=3
#                         )
                        
#                         gr.Markdown("### Examples")
#                         gr.Examples(
#                             examples=[
#                                 ["robot on the moon"],
#                                 ["coffee cup on wooden table, morning light"],
#                                 ["sports car in rain, cinematic"],
#                                 ["portrait of woman, soft studio lighting"],
#                             ],
#                             inputs=[quick_prompt]
#                         )
                        
#                         with gr.Row():
#                             quick_aspect = gr.Dropdown(
#                                 ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
#                                 value="1:1",
#                                 label="Aspect Ratio"
#                             )
#                             quick_seed = gr.Number(value=123456, maximum=2147483647, label="Seed (optional else random)", precision=0)
                        
#                         quick_negative = gr.Dropdown(
#                             choices=list(NEGATIVE_PRESETS.keys()),
#                             value="Clean (Default)",
#                             label="Negative Prompt Preset"
#                         )
                        
#                         quick_gen_btn = gr.Button("🚀 Generate with FIBO VLM", variant="primary", size="lg")
                    
#                     with gr.Column():
#                         quick_progress = gr.HTML("<div style='padding: 40px; text-align: center;'>Enter a prompt and click Generate</div>")
#                         quick_output = gr.Image(label="Generated Image", type="pil")
                        
#                         with gr.Accordion("📋 Generated Structured Prompt", open=False):
#                             quick_structured_output = gr.Code(language="json", label="", lines=15)
                
#                 def quick_generate(prompt, aspect, seed, negative_preset):
#                     if not prompt.strip():
#                         return "<div style='background: #ffebee; padding: 20px; border-radius: 12px;'>Please enter a prompt</div>", None, ""
                    
#                     try:
#                         # Generate random seed if not provided
#                         if seed is None or seed == "":
#                             import random
#                             seed = random.randint(100000, 999999)
#                         else:
#                             seed = int(seed)
                        
#                         yield f"<div style='background: #e3f2fd; padding: 20px; border-radius: 12px;'><strong>Step 1/2:</strong> FIBO VLM expanding prompt...</div>", None, ""
                        
#                         # Upsample prompt
#                         structured = upsample_simple_prompt(prompt)
                        
#                         yield f"<div style='background: #fff3e0; padding: 20px; border-radius: 12px;'><strong>Step 2/2:</strong> Generating image with seed {seed}...</div>", None, json.dumps(structured, indent=2)
                        
#                         # Generate image
#                         w, h = dimensions_from_aspect(aspect)
#                         negative = NEGATIVE_PRESETS[negative_preset]
                        
#                         payload = build_payload(
#                             structured_prompt=structured,
#                             negative_prompt=negative,
#                             guidance_scale=5,
#                             steps_num=50,
#                             seed=seed,
#                             aspect_ratio=aspect,
#                             sync=True
#                         )
                        
#                         payload["enhance_image"] = False
                        
#                         client = get_bria_client(verbose=True)
#                         img, _ = client.generate(**payload)
                        
#                         if img:
#                             update_metrics(scenes=1, api_calls=2)
#                             success = f"""
#                             <div style='background: #4CAF50; color: white; padding: 20px; border-radius: 12px;'>
#                                 <strong>✅ Generated successfully</strong><br>
#                                 Seed: {seed}<br>
#                                 VLM expanded your prompt to {len(json.dumps(structured))} characters
#                             </div>
#                             """
#                             yield success, img, json.dumps(structured, indent=2, ensure_ascii=False)
#                         else:
#                             yield "<div style='background: #ffebee; padding: 20px; border-radius: 12px;'>Generation failed</div>", None, json.dumps(structured, indent=2)
                    
#                     except Exception as e:
#                         import traceback
#                         error = f"<div style='background: #ffebee; padding: 20px; border-radius: 12px;'>Error: {str(e)}<br><pre style='font-size: 11px;'>{traceback.format_exc()[:300]}</pre></div>"
#                         yield error, None, ""
                
#                 quick_gen_btn.click(
#                     quick_generate,
#                     inputs=[quick_prompt, quick_aspect, quick_seed, quick_negative],
#                     outputs=[quick_progress, quick_output, quick_structured_output]
#                 )

# TAB 1: Storyboard Generation
            with gr.Tab("📖 Storyboard"):
                gr.HTML("""
                <div style='background: white; color: black; padding: 24px; border: 2px solid black; border-radius: 4px; margin-bottom: 24px;'>
                    <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>Storyboard Generation</h2>
                    <p style='margin: 0; font-size: 16px; color: #333;'>
                        Generate multiple scenes at once. FIBO VLM creates detailed structured prompts for each scene.
                    </p>
                </div>
                """)
                
                with gr.Accordion("❓ Quick Start Guide", open=False):
                    gr.Markdown("""
                    **Three ways to start:**
                    
                    1. **Load a preset** - Click one of the template buttons
                    2. **Import a file** - Upload TXT or JSON with scene descriptions
                    3. **Type directly** - Write scene descriptions
                    
                    **Optional:** Upload reference image for consistent style
                    
                    Click **🎬 GENERATE STORYBOARD** to create all scenes!
                    """)
                
                gr.Markdown("## Campaign Templates")
                
                # Preset buttons
                preset_buttons = {}
                with gr.Row():
                    for name in CAMPAIGN_PRESETS.keys():
                        preset_buttons[name] = gr.Button(f"Load {name}", variant="primary")
                
                gr.Markdown("---")
                gr.Markdown("## Generate Storyboard")
                
                with gr.Row():
                    with gr.Column():
                        brief_input = gr.Textbox(
                            label="Campaign Brief", 
                            lines=10, 
                            placeholder="Scene 1: Wide establishing shot...\nScene 2: Medium shot...\n\nOr import a file below"
                        )

                        with gr.Row():
                            aspect_input = gr.Dropdown(["1:1", "2:3", "3:2", "3:4", "4:3", "4:5","5:4","9:16","16:9", "21:9"], 
                                                      value="1:1", label="Aspect Ratio")
                            seed_input = gr.Number(value=123456, maximum=2147483647,label="Seed (optional else random)", precision=0)
                        
                        negative_preset_storyboard = gr.Dropdown(
                            choices=list(NEGATIVE_PRESETS.keys()),
                            value="Clean (Default)",
                            label="Negative Prompt Preset"
                        )
                        
                        with gr.Row():
                            reference_image = gr.Image(
                                label="Reference Image (Optional)",
                                type="pil",
                                sources=["upload", "clipboard"],
                                height=200
                            )
                        
                        with gr.Row():
                            reference_strength = gr.Slider(
                                0.1, 1.0, 
                                value=1.0, 
                                step=0.1,
                                label="Reference Strength",
                                visible=False
                            )
                        
                        generate_btn = gr.Button("🎬 Generate Storyboard", variant="primary", size="lg")
                        
                        gr.HTML("""
                        <div style='background: #e3f2fd; padding: 12px; border-radius: 8px; font-size: 13px; margin-top: 12px;'>
                            <strong>Reference Image:</strong> Upload to influence all scenes with a specific style
                        </div>
                        """)
                        
                        with gr.Row():
                            import_file = gr.File(
                                label="Import Campaign Brief (TXT or JSON)",
                                file_types=[".txt", ".json"],
                                type="filepath"
                            )
                        
                       # Export buttons
                        with gr.Row():
                            export_brief_btn = gr.Button("Export Brief 2(TXT)", size="lg")
                            export_json_btn = gr.Button("Export Structured 2(JSON)", size="lg")

                        export_brief_file = gr.File(label="Download Brief")
                        export_json_file = gr.File(label="Download JSON")
                        export_status = gr.HTML()

                        def export_brief_txt():
                            global storyboard_scenes
                            
                            if not storyboard_scenes:
                                return None, "<div style='background: #ffebee; padding: 20px; border-radius: 12px; color: #c62828;'>❌ No storyboard generated yet</div>"
                            
                            # Combine all scene texts
                            brief_text = "\n\n".join([f"Scene {i+1}: {scene['text']}" for i, scene in enumerate(storyboard_scenes)])
                            
                            # Write to temp file
                            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix='campaign_brief_')
                            temp_file.write(brief_text)
                            temp_file.close()
                            
                            status = "<div style='background: #e8f5e9; padding: 20px; border-radius: 12px;'>✅ Brief exported! Click download below.</div>"
                            return temp_file.name, status

                        def export_structured_json():
                            global storyboard_scenes
                            
                            if not storyboard_scenes:
                                return None, "<div style='background: #ffebee; padding: 20px; border-radius: 12px; color: #c62828;'>❌ No storyboard generated yet</div>"
                            
                            # Build export data with all structured prompts
                            export_data = {
                                "campaign": "Creative Campaign Maker Export",
                                "total_scenes": len(storyboard_scenes),
                                "scenes": []
                            }
                            
                            for i, scene in enumerate(storyboard_scenes):
                                scene_data = {
                                    "scene_number": i + 1,
                                    "description": scene.get("text", ""),
                                    "seed": scene.get("seed"),
                                    "structured_prompt": scene.get("structured_json", {})
                                }
                                export_data["scenes"].append(scene_data)
                            
                            # Write to temp file
                            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', prefix='structured_prompts_')
                            json.dump(export_data, temp_file, indent=2)
                            temp_file.close()
                            
                            status = "<div style='background: #e8f5e9; padding: 20px; border-radius: 12px;'>✅ JSON exported! Click download below.</div>"
                            return temp_file.name, status

                    # Wire up the buttons
                    export_brief_btn.click(
                        export_brief_txt,
                        outputs=[export_brief_file, export_status]
                    )

                    export_json_btn.click(
                        export_structured_json,
                        outputs=[export_json_file, export_status]
                    )
                                        
                    with gr.Column():
                        progress_html = gr.HTML("<div style='padding: 40px; text-align: center;'>Ready to generate</div>")
                        gallery_output = gr.Gallery(
                            label="Storyboard", 
                            columns=4, 
                            rows=1,
                            height=200,
                            object_fit="cover",
                            preview=True
                        )
                
                def toggle_reference_strength(img):
                    if img is not None:
                        return gr.update(visible=True)
                    return gr.update(visible=False)
                
                reference_image.change(
                    toggle_reference_strength,
                    inputs=[reference_image],
                    outputs=[reference_strength]
                )
                
                # Wire up preset buttons
                def load_preset(name):
                    preset = CAMPAIGN_PRESETS.get(name, {})
                    return preset.get("brief", ""), preset.get("aspect", "16:9"), preset.get("seed")
                
                for name, btn in preset_buttons.items():
                    btn.click(lambda n=name: load_preset(n), 
                             outputs=[brief_input, aspect_input, seed_input])
                
                # Import/Export functions
                imported_structured_prompts = gr.State([])
                
                def import_brief(file_path):
                    if not file_path:
                        return "", []
                    
                    try:
                        file_ext = Path(file_path).suffix.lower()
                        
                        if file_ext in ['.json', '.jsonl']:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            
                            structured_prompts = []
                            scene_descriptions = []
                            
                            # Try JSON array
                            if content.startswith('['):
                                try:
                                    data = json.loads(content)
                                    if isinstance(data, list):
                                        structured_prompts = data
                                except json.JSONDecodeError:
                                    pass
                            
                            # Try JSONL
                            if not structured_prompts:
                                lines = content.split('\n')
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        structured = json.loads(line)
                                        structured_prompts.append(structured)
                                    except json.JSONDecodeError:
                                        continue
                            
                            if structured_prompts:
                                for idx, s in enumerate(structured_prompts, 1):
                                    desc = s.get("short_description", f"Scene {idx}")
                                    scene_descriptions.append(f"Scene {idx}: {desc}")
                                
                                brief_text = "\n\n".join(scene_descriptions)
                                return brief_text, structured_prompts
                            else:
                                return "Could not parse JSON", []
                        
                        else:
                            # Plain text file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            return content, []
                    
                    except Exception as e:
                        return f"Error: {str(e)}", []
                
                import_file.change(
                    import_brief, 
                    inputs=[import_file], 
                    outputs=[brief_input, imported_structured_prompts]
                )

                

                
                  
            
            def generate_storyboard(brief, aspect, seed, imported_structs, ref_img, ref_strength, negative_preset):
                """Generate storyboard with optional reference image"""
                scenes = parse_brief_to_scenes(brief)
                if not scenes:
                    yield [], "<div style='color: red;'>No scenes found in brief</div>"
                    return
                
                global storyboard_images, storyboard_scenes
                images = []
                width, height = dimensions_from_aspect(aspect)
                
                # Generate random seed if not provided
                if seed is None or seed == "":
                    import random
                    base_seed = random.randint(100000, 999999)
                    print(f"🎲 No seed provided, generated random seed: {base_seed}")
                else:
                    base_seed = int(seed)
                    print(f"🎲 Using provided seed: {base_seed}")
                
                # Convert reference image to base64
                ref_img_base64 = None
                if ref_img is not None:
                    try:
                        buffered = io.BytesIO()
                        ref_img.save(buffered, format="PNG")
                        ref_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    except Exception as e:
                        print(f"Error converting reference image: {e}")
                
                use_imported = len(imported_structs) > 0
                negative_prompt = NEGATIVE_PRESETS[negative_preset]
                
                status_msg = f"<div style='background: #e3f2fd; padding: 20px; border-radius: 12px;'><strong>Using seed: {base_seed}</strong><br>"
                if ref_img_base64:
                    status_msg += f"Reference image (strength: {ref_strength})<br>"
                if use_imported:
                    status_msg += f"Using imported prompts for {len(imported_structs)} scenes"
                else:
                    status_msg += f"Generating VLM prompts for {len(scenes)} scenes"
                status_msg += "</div>"
                yield images, status_msg
                
                for idx, scene in enumerate(scenes):
                    status = f"<div style='background: #e8f5e9; padding: 20px; border-radius: 12px;'><strong>Scene {idx+1}/{len(scenes)}: Preparing...</strong></div>"
                    yield images, status
                    
                    scene_seed = base_seed + idx
                    
                    if use_imported and idx < len(imported_structs):
                        structured = imported_structs[idx]
                    else:
                        try:
                            client = get_bria_client(verbose=True)
                            structured = client.generate_structured_prompt(
                                prompt=scene["text"],
                                original_structured=None
                            )
                            if not structured:
                                structured = scene_to_structured(scene["text"])
                        except Exception as e:
                            print(f"Error generating structured prompt: {e}")
                            structured = scene_to_structured(scene["text"])
                    
                    scene["structured_json"] = structured
                    scene["seed"] = scene_seed
                    
                    # Handle yields from API call
                    img = None
                    for update in call_bria_api_with_reference(
                        scene_text=scene["text"],
                        structured=structured,
                        seed=scene_seed,
                        width=width,
                        height=height,
                        reference_image=ref_img_base64,
                        reference_strength=ref_strength if ref_img_base64 else None,
                        negative_prompt=negative_prompt
                    ):
                        if isinstance(update, str):
                            # Progress message from API call
                            yield images, update
                        elif isinstance(update, tuple):
                            # Got the final result (img, structured)
                            img, _ = update
                    
                    if img:
                        images.append(img)
                        update_metrics(scenes=1)
                        # Show updated gallery after each scene
                        yield images, f"<div style='background: #e8f5e9; padding: 20px; border-radius: 12px;'><strong>Scene {idx+1}/{len(scenes)}: Complete ✓</strong></div>"
                
                storyboard_images = images
                storyboard_scenes = scenes
                
                success = f"<div style='background: #4CAF50; color: white; padding: 20px; border-radius: 12px;'><strong>✅ Complete! Generated {len(images)} scenes with base seed {base_seed}</strong></div>"
                yield images, success

            generate_btn.click(
                generate_storyboard, 
                inputs=[
                    brief_input, 
                    aspect_input, 
                    seed_input, 
                    imported_structured_prompts,
                    reference_image,
                    reference_strength,
                    negative_preset_storyboard
                ],
                outputs=[gallery_output, progress_html]
            )

# TAB 2: Parameter Exploration
            with gr.Tab("🔬 Scene Variation"):
                gr.HTML("""
                <div style='background: white; color: black; padding: 24px; border: 2px solid black; border-radius: 4px; margin-bottom: 24px;'>
                    <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>Scene Variants</h2>
                    <p style='margin: 0; font-size: 16px;'>
                        Systematically explore lighting, camera angles, or style variations while keeping everything else constant.
                    </p>
                </div>
                """)
                
                with gr.Accordion("❓ Quick Start Guide", open=False):
                    gr.Markdown("""
                    **How to use:**
                    1. Click "📥 Load Storyboard Scenes"
                    2. Select a scene
                    3. Choose one parameter to vary
                    4. Click "✨ Generate Variations"
                    5. System locks all other parameters using same seed
                    6. Get 5 versions showing only that parameter changing
                    """)
                  
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Step 1: Load Your Storyboard")
                        
                        disent_scene_selector = gr.Dropdown(
                            choices=[],
                            label="Choose Scene from Storyboard",
                            interactive=True
                        )
                        disent_load_btn = gr.Button("📥 Load Storyboard Scenes", variant="secondary", size="lg")
                        
                        disent_base_image = gr.Image(label="Selected Scene", type="pil", interactive=False, height=300)
                        
                        gr.Markdown("### Step 2: Choose Parameter")
                        
                        param_to_vary = gr.Radio(
                            choices=["Lighting Conditions", "Camera Angles", "Style & Mood"],
                            value="Lighting Conditions",
                            label="Parameter to Vary"
                        )
                        
                        generate_variations_btn = gr.Button("✨ Generate 5 Variations", variant="primary", size="lg")
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🎲 Batch Variations")
                        gr.HTML("""
                        <div style='background: #e8f5e9; padding: 16px; border-radius: 8px; margin-bottom: 16px;'>
                            <strong>Generate multiple variations</strong> with same prompt, different seeds
                        </div>
                        """)
                        
                        with gr.Row():
                            batch_count = gr.Slider(2, 8, value=4, step=1, label="Number of Variations")
                            batch_base_seed = gr.Number(value=123456, maximum=2147483647 ,label="Base Seed (optional)", precision=0)
                        
                        batch_gen_btn = gr.Button("Generate Batch", variant="primary")
                    
                    with gr.Column():
                        disent_progress = gr.HTML("""
                        <div style='padding: 60px; text-align: center; background: #fafafa; border-radius: 12px;'>
                            <h3 style='color: #666;'>Ready for Scene Variations</h3>
                        </div>
                        """)
                        
                        gr.Markdown("### Generated Variations")
                        variations_gallery = gr.Gallery(
                            label="Scene Variations",
                            columns=3,
                            rows=2,
                            height=450
                        )
                        
                        gr.Markdown("### Batch Variations")
                        batch_gallery = gr.Gallery(
                            label="Batch Variations", 
                            columns=4, 
                            rows=2, 
                            height=400
                        )
                
                # State
                disent_current_scene = gr.State({})
                
                def load_disent_scenes():
                    global storyboard_images, storyboard_scenes
                    if not storyboard_images:
                        return gr.update(choices=[]), None, "No storyboard found"
                    
                    choices = []
                    for i in range(len(storyboard_images)):
                        text = storyboard_scenes[i].get('text', '')
                        text = re.sub(r'^Scene\s+\d+:\s*', '', text)
                        choices.append(f"Scene {i+1}: {text[:50]}...")
                    
                    return gr.update(choices=choices, value=choices[0] if choices else None), storyboard_images[0] if storyboard_images else None, f"Loaded {len(choices)} scenes"
                
                def select_disent_scene(scene_name):
                    global storyboard_images, storyboard_scenes
                    if not scene_name or not storyboard_images:
                        return None, {}
                    
                    idx = int(scene_name.split(":")[0].split()[1]) - 1
                    
                    if 0 <= idx < len(storyboard_images):
                        scene_info = storyboard_scenes[idx] if idx < len(storyboard_scenes) else {}
                        
                        scene_data = {
                            "image": storyboard_images[idx],
                            "index": idx,
                            "structured": scene_info.get("structured_json"),
                            "original_text": scene_info.get("text", ""),
                            "source": "storyboard",
                            "seed": scene_info.get("seed")
                        }
                        return storyboard_images[idx], scene_data
                    
                    return None, {}
                
                def generate_parameter_variations(base_img, param_type, scene_data):
                    if not base_img or not scene_data:
                        yield "No scene selected", []
                        return
                    
                    try:
                        param_clean = param_type.split()[0]
                        
                        variation_sets = {
                            "Lighting": [
                                ("Studio", "Studio lighting with softboxes", "Diffused from multiple angles", "Soft controlled shadows"),
                                ("Golden Hour", "Golden hour natural light", "Low-angle sunlight", "Long soft shadows"),
                                ("Overcast", "Soft diffused overcast", "Diffused from above", "Minimal shadows"),
                                ("Dramatic", "Dramatic side lighting", "Hard directional 45-degree", "Deep defined shadows"),
                                ("Soft Natural", "Soft natural daylight", "Diffused natural light", "Gentle shadows")
                            ],
                            "Camera": [
                                ("Eye-Level", "Eye-level camera", "Eye-level"),
                                ("Above", "Camera above subject", "Elevated looking down"),
                                ("Below", "Camera below subject", "Low looking upward"),
                                ("Top-Down", "Overhead top-down", "Directly overhead"),
                                ("Low Angle", "Low-angle dramatic", "Dramatic low angle")
                            ],
                            "Style": [
                                ("Photorealistic", "realistic, detailed, photographic"),
                                ("Cinematic", "cinematic, dramatic, movie-like"),
                                ("Editorial", "editorial, fashion magazine"),
                                ("Commercial", "commercial, advertising, polished"),
                                ("Artistic", "artistic, creative, expressive")
                            ]
                        }
                        
                        variations = variation_sets.get(param_clean, [])
                        if not variations:
                            yield f"Unknown parameter: {param_type}", []
                            return
                        
                        seed = scene_data.get("seed", 100000)
                        structured_base = scene_data.get("structured")
                        if not structured_base:
                            yield "No structured prompt", []
                            return
                        
                        variation_images = []
                        
                        for idx, variation_data in enumerate(variations):
                            name = variation_data[0]
                            yield f"Generating {name} ({idx+1}/{len(variations)})...", variation_images
                            
                            modified = copy.deepcopy(structured_base)
                            
                            if param_clean == "Lighting":
                                modified["lighting"]["conditions"] = variation_data[1]
                                if len(variation_data) > 2:
                                    modified["lighting"]["direction"] = variation_data[2]
                                if len(variation_data) > 3:
                                    modified["lighting"]["shadows"] = variation_data[3]
                            elif param_clean == "Camera":
                                modified["photographic_characteristics"]["camera_angle"] = variation_data[1]
                            elif param_clean == "Style":
                                modified["artistic_style"] = variation_data[1]
                            
                            w, h = base_img.size
                            aspect = "16:9" if w > h else "9:16" if h > w else "1:1"
                            
                            payload = build_payload(
                                structured_prompt=modified,
                                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                                guidance_scale=5,
                                steps_num=50,
                                seed=seed,
                                aspect_ratio=aspect,
                                sync=True
                            )
                            payload["enhance_image"] = False
                            
                            client = get_bria_client(verbose=False)
                            img, _ = client.generate(**payload)
                            
                            if img:
                                # Add label
                                draw = ImageDraw.Draw(img)
                                try:
                                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
                                    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
                                except:
                                    font = ImageFont.load_default()
                                    font_small = font
                                
                                draw.rectangle([(12, 12), (300, 60)], fill=(0, 0, 0, 220))
                                draw.text((20, 20), name, fill="white", font=font)
                                
                                draw.rectangle([(12, img.height - 50), (200, img.height - 12)], fill=(76, 175, 80, 200))
                                draw.text((20, img.height - 45), f"Seed: {seed}", fill="white", font=font_small)
                                
                                variation_images.append(img)
                                update_metrics(scenes=1)

                        global disentanglement_images, disentanglement_metadata
                        disentanglement_images = variation_images
                        disentanglement_metadata = {
                            "base_scene": scene_data,
                            "parameter_type": param_clean,
                            "seed": seed,
                            "variation_names": [v[0] for v in variations]
                        }
                        
                        yield f"✅ Complete! Generated {len(variation_images)} {param_clean} variations (seed: {seed})", variation_images
                    
                    except Exception as e:
                        yield f"Error: {str(e)}", []
                
                def generate_batch(base_img, scene_data, count, base_seed):
                    if not base_img or not scene_data:
                        return []
                    
                    if base_seed is None or base_seed == "":
                        import random
                        base_seed = random.randint(100000, 999999)
                    else:
                        base_seed = int(base_seed)
                    
                    structured = scene_data.get("structured")
                    if not structured:
                        return []
                    
                    w, h = base_img.size
                    images = generate_batch_variations(structured, base_seed, int(count), w, h)
                    
                    # Add labels
                    labeled_images = []
                    for i, img in enumerate(images):
                        img_copy = img.copy()
                        draw = ImageDraw.Draw(img_copy)
                        try:
                            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
                        except:
                            font = ImageFont.load_default()
                        
                        draw.rectangle([(10, 10), (200, 50)], fill=(76, 175, 80, 200))
                        draw.text((20, 20), f"Seed: {base_seed + i}", fill="white", font=font)
                        labeled_images.append(img_copy)
                    
                    return labeled_images
                
                # Wire up
                disent_load_btn.click(
                    load_disent_scenes,
                    outputs=[disent_scene_selector, disent_base_image, disent_progress]
                )
                
                disent_scene_selector.change(
                    select_disent_scene,
                    inputs=[disent_scene_selector],
                    outputs=[disent_base_image, disent_current_scene]
                )
                
                generate_variations_btn.click(
                    generate_parameter_variations,
                    inputs=[disent_base_image, param_to_vary, disent_current_scene],
                    outputs=[disent_progress, variations_gallery]
                )
                
                batch_gen_btn.click(
                    generate_batch,
                    inputs=[disent_base_image, disent_current_scene, batch_count, batch_base_seed],
                    outputs=[batch_gallery]
                )

# TAB 3: Refinement
            with gr.Tab("✨ Refinement"):
                gr.HTML("""
                 <div style='background: white; color: black; padding: 24px; border: 2px solid black; border-radius: 4px; margin-bottom: 24px;'>
                    <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>Targeted Editing</h2>
                    <p style='margin: 0; font-size: 16px;'>
                        Make specific changes without regenerating. Chained refinements preserve previous edits.
                    </p>
                </div>
                """)
                
                with gr.Accordion("❓ Quick Start Guide", open=False):
                    gr.Markdown("""
                    1. Select image source (Storyboard or Parameter Variations)
                    2. Choose a scene and click "Load Images"
                    3. Enter refinement instruction
                    4. Click "✨ REFINE IMAGE"
                    5. **Each refinement builds on previous changes!**
                    6. Save to Prompt Library to reuse
                    """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Select Image to Refine")
                        
                        image_source = gr.Radio(
                            choices=["Storyboard Scenes", "Parameter Variations"],
                            value="Storyboard Scenes",
                            label="Image Source"
                        )
                        
                        scene_selector = gr.Dropdown(
                            choices=[],
                            label="Choose Scene",
                            interactive=True
                        )
                        load_scenes_btn = gr.Button("Load Images", variant="secondary")
                        
                        selected_image = gr.Image(label="Current Image", type="pil", interactive=False)
                        
                        gr.Markdown("### Refinement Instructions")
                        refine_prompt = gr.Textbox(
                            label="What would you like to change?",
                            placeholder="Examples:\n• make robot black\n• change earth to moon\n• add golden hour lighting",
                            lines=3
                        )
                        
                        with gr.Row():
                            refine_strength = gr.Slider(0.1, 1.0, value=1.0, step=0.1, label="Strength")
                            keep_seed = gr.Checkbox(label="Keep same seed", value=True)
                        
                        refine_btn = gr.Button("✨ Refine Image", variant="primary", size="lg")
                        
                        save_to_library_btn = gr.Button("💾 Save to Prompt Library", variant="secondary", size="sm")
                    
                    with gr.Column():
                        refine_progress = gr.HTML("<div style='padding: 40px; text-align: center;'>Select source and load images</div>")
                        
                        gr.Markdown("### Before & After")
                        with gr.Row():
                            before_image = gr.Image(label="Before", type="pil", interactive=False)
                            after_image = gr.Image(label="After", type="pil", interactive=False)
                        
                        with gr.Accordion("🐛 Debug & Comparison", open=False):
                            debug_display = gr.HTML("")
                            
                            gr.Markdown("### Structured Prompt Comparison")
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**Original:**")
                                    original_json_display = gr.Code(language="json", label="", lines=10)
                                with gr.Column():
                                    gr.Markdown("**Refined:**")
                                    refined_json_display = gr.Code(language="json", label="", lines=10)
                            
                            gr.Markdown("### Visual Diff")
                            diff_display = gr.HTML("")
                            
                            changes_display = gr.HTML("")
                        
                        gr.Markdown("### Export")
                        with gr.Row():
                            generate_doc_btn = gr.Button("📄 Marketing Document", variant="primary", size="sm")
                            export_workflow_btn = gr.Button("📤 Export Workflow", variant="secondary", size="sm")
                            clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary", size="sm")
                        
                        marketing_doc_output = gr.HTML("")
                        marketing_doc_file = gr.File(label="Download")
                        workflow_export_file = gr.File(label="Download Workflow")
                        workflow_export_msg = gr.HTML("")
                
                # State
                refinement_history = gr.State([])
                current_scene_data = gr.State({})
                prev_structured_prompt = gr.State(None)
                
                def load_refine_scenes(source):
                    global storyboard_images, storyboard_scenes
                    global disentanglement_images, disentanglement_metadata
                    
                    if source == "Storyboard Scenes":
                        if not storyboard_images:
                            return gr.update(choices=[]), None, "No storyboard found"
                        
                        choices = [f"Storyboard Scene {i+1}" for i in range(len(storyboard_images))]
                        return gr.update(choices=choices, value=choices[0] if choices else None), storyboard_images[0] if storyboard_images else None, f"Loaded {len(choices)} scenes"
                    
                    else:
                        if not disentanglement_images:
                            return gr.update(choices=[]), None, "No variations found"
                        
                        var_names = disentanglement_metadata.get("variation_names", [])
                        param_type = disentanglement_metadata.get("parameter_type", "Variation")
                        
                        if var_names and len(var_names) == len(disentanglement_images):
                            choices = [f"{param_type} {i+1}: {var_names[i]}" for i in range(len(disentanglement_images))]
                        else:
                            choices = [f"{param_type} {i+1}" for i in range(len(disentanglement_images))]
                        
                        return gr.update(choices=choices, value=choices[0] if choices else None), disentanglement_images[0] if disentanglement_images else None, f"Loaded {len(choices)} variations"
                
                def select_refine_scene(scene_name, source):
                    global storyboard_images, storyboard_scenes
                    global disentanglement_images, disentanglement_metadata
                    
                    if not scene_name:
                        return None, {}
                    
                    if source == "Storyboard Scenes":
                        if not storyboard_images:
                            return None, {}
                        
                        idx = int(scene_name.split()[2]) - 1
                        if 0 <= idx < len(storyboard_images):
                            scene_data = {
                                "image": storyboard_images[idx],
                                "index": idx,
                                "structured": storyboard_scenes[idx].get("structured_json") if idx < len(storyboard_scenes) else None,
                                "original_text": storyboard_scenes[idx].get("text") if idx < len(storyboard_scenes) else "",
                                "source": "storyboard",
                                "seed": storyboard_scenes[idx].get("seed") if idx < len(storyboard_scenes) else None
                            }
                            return storyboard_images[idx], scene_data
                    
                    else:
                        if not disentanglement_images:
                            return None, {}
                        
                        idx = int(scene_name.split(":")[0].split()[1]) - 1
                        if 0 <= idx < len(disentanglement_images):
                            base_scene = disentanglement_metadata.get("base_scene", {})
                            var_names = disentanglement_metadata.get("variation_names", [])
                            
                            scene_data = {
                                "image": disentanglement_images[idx],
                                "index": idx,
                                "structured": base_scene.get("structured"),
                                "original_text": base_scene.get("original_text", ""),
                                "source": "disentanglement",
                                "variation_name": var_names[idx] if idx < len(var_names) else f"Variation {idx+1}",
                                "parameter_type": disentanglement_metadata.get("parameter_type", ""),
                                "seed": disentanglement_metadata.get("seed")
                            }
                            return disentanglement_images[idx], scene_data
                    
                    return None, {}
                
                def refine_image(current_img, refine_text, strength, use_seed, scene_data, history, prev_structured):
                    """Refine image using FIBO VLM with chained refinements"""
                    if not current_img or not refine_text.strip():
                        return "Please provide image and instruction", current_img, None, None, history, scene_data, prev_structured, "", "", "", ""
                    
                    try:
                        # STEP 1: Use prev_structured if chaining, else use original
                        if prev_structured:
                            structured = prev_structured
                            using = "PREVIOUS REFINEMENT (chained)"
                            print(f"\n🔗 CHAINED REFINEMENT")
                        else:
                            structured = scene_data.get("structured") if scene_data else None
                            using = "ORIGINAL from scene"
                            print(f"\n🆕 FIRST REFINEMENT")
                        
                        if not structured:
                            return "No structured prompt found", current_img, None, None, history, scene_data, prev_structured, "", "", "", ""
                        
                        # STEP 2: Get seed
                        seed = None
                        if use_seed and scene_data:
                            seed = scene_data.get("seed")
                            if seed:
                                print(f"✅ Using original seed: {seed}")
                        
                        # STEP 3: Call FIBO VLM to refine
                        try:
                            client = get_bria_client(verbose=True)
                            print(f"☁️ Calling FIBO VLM with instruction: {refine_text}")
                            
                            modified_structured = client.generate_structured_prompt(
                                prompt=refine_text,
                                original_structured=structured
                            )
                            
                            if not modified_structured:
                                return "VLM refinement failed", current_img, None, None, history, scene_data, prev_structured, "", "", "", ""
                            
                            print(f"✅ VLM returned refined prompt")
                            
                        except Exception as e:
                            return f"VLM Error: {str(e)}", current_img, None, None, history, scene_data, prev_structured, "", "", "", ""
                        
                        # STEP 4: Generate image
                        w, h = current_img.size
                        aspect = "16:9" if w > h else "9:16" if h > w else "1:1"
                        
                        update_metrics(api_calls=2)
                        
                        payload = build_payload(
                            structured_prompt=modified_structured,
                            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                            guidance_scale=5,
                            steps_num=50,
                            seed=seed,
                            aspect_ratio=aspect,
                            sync=True
                        )
                        payload["enhance_image"] = False
                        
                        refined_img, _ = call_bria_and_return_image_and_payload(payload, verbose=True)
                        
                        if refined_img:
                            source_label = "Parameter Exploration" if scene_data.get("source") == "disentanglement" else "Storyboard"
                            
                            # CRITICAL: Update scene_data with refined prompt for chaining
                            updated_scene_data = {
                                "image": refined_img,
                                "index": scene_data.get("index", 0),
                                "structured": modified_structured,  # KEY: use refined for next iteration
                                "original_text": scene_data.get("original_text", ""),
                                "source": scene_data.get("source", "storyboard"),
                                "seed": seed
                            }
                            
                            if scene_data.get("source") == "disentanglement":
                                updated_scene_data["variation_name"] = scene_data.get("variation_name", "")
                                updated_scene_data["parameter_type"] = scene_data.get("parameter_type", "")
                            
                            # Format JSONs
                            original_json_str = json.dumps(structured, indent=2, ensure_ascii=False)
                            refined_json_str = json.dumps(modified_structured, indent=2, ensure_ascii=False)
                            
                            # Generate diff
                            diff_html = show_structured_diff(structured, modified_structured)
                            
                            # Detect changes
                            changes_detected = []
                            if modified_structured.get("objects") != structured.get("objects"):
                                changes_detected.append("objects")
                            if modified_structured.get("lighting") != structured.get("lighting"):
                                changes_detected.append("lighting")
                            if modified_structured.get("background_setting") != structured.get("background_setting"):
                                changes_detected.append("background")
                            
                            changes_text = ", ".join(changes_detected) if changes_detected else "various"
                            
                            # Changes HTML
                            changes_html = "<div style='background: white; padding: 20px; border-radius: 12px;'><h4>Modified Sections:</h4>"
                            for section in ["objects", "lighting", "aesthetics", "photographic_characteristics", "background_setting"]:
                                if modified_structured.get(section) != structured.get(section):
                                    changes_html += f"<div style='background: #c8e6c9; padding: 8px; margin: 4px; border-radius: 4px;'>✓ {section}</div>"
                                else:
                                    changes_html += f"<div style='background: #f5f5f5; padding: 8px; margin: 4px; border-radius: 4px;'>{section}</div>"
                            changes_html += "</div>"
                            
                            # Add to history
                            refinement_record = {
                                "before": current_img.copy(),
                                "after": refined_img.copy(),
                                "instruction": refine_text,
                                "strength": strength,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "seed": seed,
                                "source": source_label,
                                "changes": changes_text
                            }
                            new_history = history + [refinement_record]
                            
                            success = f"""
                            <div style='background: #4CAF50; color: white; padding: 20px; border-radius: 12px;'>
                                <strong>✅ Refinement Complete</strong><br>
                                Source: {source_label}<br>
                                Instruction: "{refine_text}"<br>
                                Modified: {changes_text}<br>
                                Seed: {seed if seed else "random"}<br>
                                ✅ <strong>State updated - next refinement builds on these changes</strong>
                            </div>
                            """
                            
                            update_metrics(scenes=1)
                            
                            # Return modified_structured as prev_structured for chaining
                            return success, current_img, refined_img, refined_img, new_history, updated_scene_data, modified_structured, original_json_str, refined_json_str, diff_html, changes_html
                        
                        else:
                            return "Image generation failed", current_img, None, None, history, scene_data, prev_structured, "", "", "", ""
                    
                    except Exception as e:
                        import traceback
                        error = f"Error: {str(e)}\n{traceback.format_exc()[:500]}"
                        return error, current_img, None, None, history, scene_data, prev_structured, "", "", "", ""
                
                def generate_marketing_document(history):
                    if not history:
                        return "<div style='background: #ffebee; padding: 12px; border-radius: 8px;'>No refinements</div>", None
                    
                    try:
                        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Creative Campaign Maker - Refinements</title>
    <style>
        body {{ font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); padding: 40px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #4CAF50, #2196F3); color: white; padding: 60px 40px; text-align: center; }}
        .refinement {{ margin: 40px; }}
        .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .comparison img {{ width: 100%; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FIBO AutoDirector - Refinement Showcase</h1>
            <p>Generated {len(history)} refinements</p>
        </div>
"""
                        
                        for idx, record in enumerate(history, 1):
                            before_buffer = io.BytesIO()
                            record['before'].save(before_buffer, format='PNG')
                            before_b64 = base64.b64encode(before_buffer.getvalue()).decode('utf-8')
                            
                            after_buffer = io.BytesIO()
                            record['after'].save(after_buffer, format='PNG')
                            after_b64 = base64.b64encode(after_buffer.getvalue()).decode('utf-8')
                            
                            html_content += f"""
        <div class="refinement">
            <h2>Refinement #{idx}: "{record['instruction']}"</h2>
            <p>Seed: {record.get('seed', 'Random')} | Changes: {record.get('changes', 'N/A')}</p>
            <div class="comparison">
                <div><h3>Before</h3><img src="data:image/png;base64,{before_b64}"></div>
                <div><h3>After</h3><img src="data:image/png;base64,{after_b64}"></div>
            </div>
        </div>
"""
                        
                        html_content += """
    </div>
</body>
</html>
"""
                        
                        temp_file = Path(tempfile.gettempdir()) / f"fibo_refinements_{int(time.time())}.html"
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        return f"<div style='background: #e8f5e9; padding: 12px; border-radius: 8px;'>✅ Document generated</div>", str(temp_file)
                    
                    except Exception as e:
                        return f"<div style='background: #ffebee; padding: 12px; border-radius: 8px;'>Error: {str(e)}</div>", None
                
                def clear_refinement_history():
                    return [], "<div style='background: #fff3e0; padding: 12px; border-radius: 8px;'>History cleared</div>", None
                
                def save_current_to_library(scene_data):
                    if not scene_data or not scene_data.get("structured"):
                        return "", "<div style='background: #ffebee; padding: 12px; border-radius: 8px;'>No prompt available</div>"
                    
                    structured_json = json.dumps(scene_data["structured"], indent=2, ensure_ascii=False)
                    return structured_json, "<div style='background: #e8f5e9; padding: 12px; border-radius: 8px;'>Ready to save - go to Prompt Library tab</div>"
                
                # Wire up
                load_scenes_btn.click(
                    load_refine_scenes,
                    inputs=[image_source],
                    outputs=[scene_selector, selected_image, refine_progress]
                )
                
                scene_selector.change(
                    lambda scene, src: (*select_refine_scene(scene, src), None),
                    inputs=[scene_selector, image_source],
                    outputs=[selected_image, current_scene_data, prev_structured_prompt]
                )
                
                image_source.change(
                    load_refine_scenes,
                    inputs=[image_source],
                    outputs=[scene_selector, selected_image, refine_progress]
                )
                
                refine_btn.click(
                    refine_image,
                    inputs=[
                        selected_image, 
                        refine_prompt, 
                        refine_strength, 
                        keep_seed, 
                        current_scene_data, 
                        refinement_history,
                        prev_structured_prompt
                    ],
                    outputs=[
                        refine_progress, 
                        before_image, 
                        after_image, 
                        selected_image, 
                        refinement_history,
                        current_scene_data,
                        prev_structured_prompt,
                        original_json_display,
                        refined_json_display,
                        diff_display,
                        changes_display
                    ]
                )
                
                generate_doc_btn.click(
                    generate_marketing_document,
                    inputs=[refinement_history],
                    outputs=[marketing_doc_output, marketing_doc_file]
                )
                
                export_workflow_btn.click(
                    export_refinement_workflow,
                    inputs=[refinement_history],
                    outputs=[workflow_export_file, workflow_export_msg]
                )
                
                clear_history_btn.click(
                    clear_refinement_history,
                    outputs=[refinement_history, marketing_doc_output, marketing_doc_file]
                )

# TAB 4: Prompt Library
            with gr.Tab("📚 Prompt Library"):
                gr.HTML("""
                 <div style='background: white; color: black; padding: 24px; border: 2px solid black; border-radius: 4px; margin-bottom: 24px;'>
                    <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>Prompt Library</h2>
                    <p style='margin: 0; font-size: 16px;'>
                        Save and reuse your best structured prompts. Build a library of styles and compositions.
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Save Prompt")
                        library_save_name = gr.Textbox(label="Prompt Name", placeholder="My Robot Style")
                        library_save_structured = gr.Code(language="json", label="Structured Prompt", lines=15)
                        library_save_btn = gr.Button("💾 Save to Library", variant="primary")
                        library_save_msg = gr.HTML("")
                    
                    with gr.Column():
                        gr.Markdown("### Load Prompt")
                        library_load_dropdown = gr.Dropdown(choices=[], label="Saved Prompts", interactive=True)
                        library_load_btn = gr.Button("📥 Load from Library", variant="secondary")
                        library_load_structured = gr.Code(language="json", label="Loaded Prompt", lines=15)
                        library_load_msg = gr.HTML("")
                
                library_save_btn.click(
                    save_prompt_to_library,
                    inputs=[library_save_name, library_save_structured],
                    outputs=[library_load_dropdown, library_save_msg]
                )
                
                library_load_btn.click(
                    load_prompt_from_library,
                    inputs=[library_load_dropdown],
                    outputs=[library_load_structured, library_load_msg]
                )

# # TAB 5: Video Export 
            with gr.Tab("🎬 Video Export"):
                    gr.HTML("""
                         <div style='background: white; color: black; padding: 24px; border: 2px solid black; border-radius: 4px; margin-bottom: 24px;'>
                            <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>Video Studio</h2>
                            <p style='margin: 0; font-size: 16px;'>
                                Transform storyboard frames into videos using AI video generation.
                            </p>
                        </div>
                        """)
                        
                    with gr.Tabs():
                            # TAB: AI Video Generation
                            with gr.Tab("🤖 AI Video Generation"):
                                # gr.HTML("""
                                # <div style='background: white; color: black; padding: 24px; border: 2px solid black; border-radius: 4px; margin-bottom: 24px;'>
                                #     <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>✨ Stable Video Diffusion</h3>
                                #     <p style='margin: 0; font-size: 14px;'>
                                #         Generate 5-second AI videos from your storyboard frames.<br>
                                       
                                #     </p>
                                # </div>
                                # """)
                                
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("### Step 1: Select Frame")
                                        
                                        video_scene_selector = gr.Dropdown(
                                            choices=[],
                                            label="Choose Scene from Storyboard",
                                            interactive=True
                                        )
                                        video_load_btn = gr.Button("📥 Load Storyboard", variant="secondary", size="lg")
                                        
                                        video_input_image = gr.Image(
                                            label="Selected Frame", 
                                            type="pil", 
                                            interactive=False,
                                            height=300
                                        )
                                        
                                        gr.Markdown("### Step 2: Motion Settings")
                                        
                                        motion_amount = gr.Slider(
                                            1, 255, 
                                            value=127, 
                                            step=1,
                                            label="Motion Amount"
                                        )
                                        
                                        generate_video_btn = gr.Button("🎬 Generate AI Video", variant="primary", size="lg")
                                    
                                    with gr.Column():
                                        video_progress = gr.HTML("<div style='padding: 60px; text-align: center;'>Load a frame to begin</div>")
                                        video_output = gr.Video(label="Generated Video")
                                        video_file_output = gr.File(label="Download Video")
                                
                                # State
                                video_current_scene = gr.State({})
                                
                                # Functions (MUST BE INSIDE THE TAB)
                                def load_video_scenes():
                                    global storyboard_images, storyboard_scenes
                                    if not storyboard_images:
                                        return gr.update(choices=[]), None, "No storyboard found"
                                    
                                    choices = []
                                    for i in range(len(storyboard_images)):
                                        text = storyboard_scenes[i].get('text', '')
                                        text = re.sub(r'^Scene\s+\d+:\s*', '', text)
                                        choices.append(f"Scene {i+1}: {text[:50]}...")
                                    
                                    return gr.update(choices=choices, value=choices[0] if choices else None), storyboard_images[0] if storyboard_images else None, f"Loaded {len(choices)} scenes"
                                
                                def select_video_scene(scene_name):
                                    global storyboard_images, storyboard_scenes
                                    if not scene_name or not storyboard_images:
                                        return None, {}
                                    
                                    idx = int(scene_name.split(":")[0].split()[1]) - 1
                                    
                                    if 0 <= idx < len(storyboard_images):
                                        scene_info = storyboard_scenes[idx] if idx < len(storyboard_scenes) else {}
                                        scene_data = {
                                            "image": storyboard_images[idx],
                                            "index": idx,
                                            "text": scene_info.get("text", "")
                                        }
                                        return storyboard_images[idx], scene_data
                                    
                                    return None, {}
                                
                                def format_progress_html(message, elapsed=None, step=None):
                                    """Format progress message with timing and styling"""
                                    color = "#1976d2" if "✅" in message else "#f57c00" if "🔄" in message else "#d32f2f"
                                    
                                    html = f"""
                                    <div style='padding: 20px; background: #f5f5f5; border-radius: 12px; border-left: 4px solid {color};'>
                                        <div style='font-size: 16px; font-weight: 500; margin-bottom: 8px;'>{message}</div>
                                    """
                                    
                                    if step:
                                        html += f"<div style='color: #666; font-size: 14px; margin-bottom: 4px;'>Step: {step}</div>"
                                    
                                    if elapsed is not None:
                                        html += f"<div style='color: #666; font-size: 14px;'>Elapsed: {elapsed:.1f}s</div>"
                                    
                                    html += "</div>"
                                    return html
                                
                                def generate_ai_video(input_img, motion, scene_data):
                                    import time
                                    
                                    if not input_img:
                                        yield format_progress_html("❌ Please select a frame"), None, None
                                        return
                                    
                                    try:
                                        start_time = time.time()
                                        scene_text = scene_data.get("text", "Scene") if scene_data else "Scene"
                                        
                                        # Step 1: Initialize
                                        yield format_progress_html(
                                            f"🎬 Starting video generation for: {scene_text[:50]}...",
                                            elapsed=time.time() - start_time,
                                            step="1/4 - Initializing"
                                        ), None, None
                                        
                                        # Step 2: Prepare image
                                        yield format_progress_html(
                                            "🖼️ Preparing image (resizing to 1024x576)...",
                                            elapsed=time.time() - start_time,
                                            step="2/4 - Image Processing"
                                        ), None, None
                                        
                                        # Step 3: Try Hugging Face
                                        yield format_progress_html(
                                            "🔄 Trying Hugging Face Inference API (free)...",
                                            elapsed=time.time() - start_time,
                                            step="3/4 - Hugging Face API"
                                        ), None, None
                                        
                                        hf_start = time.time()
                                        video_path = generate_video_from_frame_hf(input_img, motion_bucket_id=int(motion))
                                        hf_elapsed = time.time() - hf_start
                                        
                                        if not video_path:
                                            # Step 4: Try Replicate as fallback
                                            yield format_progress_html(
                                                f"🔄 Hugging Face failed ({hf_elapsed:.1f}s), trying Replicate API...",
                                                elapsed=time.time() - start_time,
                                                step="4/4 - Replicate API (Fallback)"
                                            ), None, None
                                            
                                            replicate_start = time.time()
                                            video_path = generate_video_from_frame_replicate(input_img, motion_amount=int(motion))
                                            replicate_elapsed = time.time() - replicate_start
                                            
                                            if video_path:
                                                total_elapsed = time.time() - start_time
                                                update_metrics(videos=1)
                                                yield format_progress_html(
                                                    f"✅ Video generated via Replicate in {replicate_elapsed:.1f}s (total: {total_elapsed:.1f}s)",
                                                    elapsed=total_elapsed,
                                                    step="Complete"
                                                ), video_path, video_path
                                            else:
                                                total_elapsed = time.time() - start_time
                                                yield format_progress_html(
                                                    f"❌ Both Hugging Face and Replicate failed after {total_elapsed:.1f}s",
                                                    elapsed=total_elapsed,
                                                    step="Failed"
                                                ), None, None
                                        else:
                                            total_elapsed = time.time() - start_time
                                            update_metrics(videos=1)
                                            yield format_progress_html(
                                                f"✅ Video generated via Hugging Face in {hf_elapsed:.1f}s (total: {total_elapsed:.1f}s)",
                                                elapsed=total_elapsed,
                                                step="Complete"
                                            ), video_path, video_path
                                    
                                    except Exception as e:
                                        total_elapsed = time.time() - start_time
                                        yield format_progress_html(
                                            f"❌ Error: {str(e)}",
                                            elapsed=total_elapsed,
                                            step="Error"
                                        ), None, None
                                
                                # Wire up (MUST BE INSIDE THE TAB - SAME INDENTATION LEVEL AS FUNCTIONS)
                                video_load_btn.click(
                                    load_video_scenes,
                                    outputs=[video_scene_selector, video_input_image, video_progress]
                                )
                                
                                video_scene_selector.change(
                                    select_video_scene,
                                    inputs=[video_scene_selector],
                                    outputs=[video_input_image, video_current_scene]
                                )
                                
                                generate_video_btn.click(
                                    generate_ai_video,
                                    inputs=[video_input_image, motion_amount, video_current_scene],
                                    outputs=[video_progress, video_output, video_file_output]
                                )
                            
                            # TAB: Traditional Video
                            with gr.Tab("📹 Multi-Scene Video"):
                                gr.Markdown("### Coming Soon")

# TAB 6: Analytics
            with gr.Tab("📊 Analytics"):
                metrics_display = gr.HTML(get_metrics_html())
                refresh_btn = gr.Button("Refresh Metrics", variant="secondary")
                refresh_btn.click(lambda: get_metrics_html(), outputs=[metrics_display])
        
        # # Footer
        # gr.HTML(f"""
        # <div style='background: white; color: black; padding: 24px; border: 2px solid black; border-radius: 4px; margin-bottom: 24px;'>
        #             <h2 style='margin: 0 0 12px; color: black; font-weight: 600;'>{APP_NAME} v{APP_VERSION}</h3>
        #     <p style='color: #666;'>
        #         Complete FIBO VLM Integration • Prompt Upsampling • Chained Refinements • Batch Generation • Prompt Library
        #     </p>
        # </div>
        # """)
    
    return demo

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(f"{APP_NAME} v{APP_VERSION}")
    print("=" * 80)
    print("FIBO VLM Features:")
    print("  * Prompt Upsampling (Simple → Detailed)")
    print("  * Chained Refinements (Preserves Previous Changes)")
    print("  * Batch Variation Generation")
    print("  * Visual Diff Viewer")
    print("  * Prompt Library System")
    print("  * Workflow Export")
    print("  * Negative Prompt Presets")
    print("=" * 80)
    
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=True,
        show_error=True
    )