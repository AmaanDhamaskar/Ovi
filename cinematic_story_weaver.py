"""
Cinematic Story Weaver
======================
A creative Gradio application powered by OVI (Twin Backbone Cross-Modal Fusion).

Given a story beat / scene description, this app:
  1. Enriches and structures the user's idea into a full cinematic scene
  2. Auto-formats the prompt with proper OVI tags (<S>…<E>, Audio: …)
  3. Generates a 5–10 second synchronized audio-video clip via OVI
  4. Displays the result with scene metadata and prompt breakdown

Usage (with OVI model weights):
    python cinematic_story_weaver.py --ckpt_dir ./ckpts --model 960x960_10s

Usage (demo / no model):
    python cinematic_story_weaver.py --demo
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Optional imports — only needed when running with real model weights
# ---------------------------------------------------------------------------
try:
    from ovi.ovi_fusion_engine import OviFusionEngine  # type: ignore
    OVI_AVAILABLE = True
except ImportError:
    OVI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Cinematic scene presets — help users get started quickly
# ---------------------------------------------------------------------------
SCENE_PRESETS = {
    "🎬 Drama: Confession": {
        "scene_idea": "A man stands in the rain outside a brightly lit apartment window, working up the courage to knock.",
        "mood": "Melancholic",
        "genre": "Drama",
        "speech": "I should have told you sooner. I'm sorry.",
        "sfx": "Heavy rain, distant thunder, wet footsteps on pavement.",
    },
    "🚀 Sci-Fi: Launch": {
        "scene_idea": "A lone astronaut steps onto the boarding gantry of a massive rocket as dawn breaks over the launch pad.",
        "mood": "Epic / Awe",
        "genre": "Sci-Fi",
        "speech": "T-minus ten. This is what we trained for.",
        "sfx": "Wind, distant countdown beeps, massive rocket engine ignition roar.",
    },
    "🌿 Nature: Dawn": {
        "scene_idea": "Golden morning light filters through a dense rainforest canopy. Mist rises from the forest floor.",
        "mood": "Peaceful",
        "genre": "Nature Documentary",
        "speech": None,
        "sfx": "Birds calling, insects humming, gentle water flow, rustling leaves.",
    },
    "🎸 Music: Performance": {
        "scene_idea": "A guitarist closes their eyes and shreds an electric solo on a neon-lit stage in front of a roaring crowd.",
        "mood": "Energetic",
        "genre": "Music",
        "speech": None,
        "sfx": "Electric guitar solo, crowd cheering, stage feedback, driving drums.",
    },
    "🏙️ Thriller: Chase": {
        "scene_idea": "A courier sprints across rooftops at night, leaping between buildings as city lights blur below.",
        "mood": "Tense",
        "genre": "Thriller",
        "speech": "Almost there — don't look down!",
        "sfx": "Running footsteps, laboured breathing, distant police sirens, wind rush.",
    },
    "🍃 Romance: Meeting": {
        "scene_idea": "Two strangers reach for the same book in a quiet afternoon bookshop and their eyes meet.",
        "mood": "Warm / Tender",
        "genre": "Romance",
        "speech": "Sorry — you can have it. I was just browsing.",
        "sfx": "Soft acoustic guitar, pages turning, ambient café noise, light chuckle.",
    },
}

# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------
def build_visual_description(scene_idea: str, mood: str, genre: str,
                              shot_type: str, time_of_day: str) -> str:
    """Craft a detailed visual description from user inputs."""
    parts = [scene_idea.strip()]
    if shot_type != "Auto":
        parts.append(f"Shot as a {shot_type.lower()}.")
    if time_of_day != "Auto":
        parts.append(f"Time of day: {time_of_day.lower()}.")
    if mood:
        parts.append(f"Mood: {mood.lower()}.")
    return " ".join(parts)


def format_ovi_prompt(visual: str,
                      speech: Optional[str],
                      audio_desc: Optional[str],
                      model_variant: str) -> str:
    """Assemble the combined OVI prompt with the correct tag format."""
    prompt = visual
    if speech and speech.strip():
        prompt += f" <S>{speech.strip()}<E>"
    if audio_desc and audio_desc.strip():
        if "720" in model_variant:
            prompt += f" <AUDCAP>{audio_desc.strip()}<ENDAUDCAP>"
        else:
            prompt += f" Audio: {audio_desc.strip()}"
    return prompt


def build_audio_description(sfx: str, mood: str, speaker_traits: str,
                             speech: Optional[str]) -> str:
    """Combine SFX and speaker traits into a rich audio description."""
    parts = []
    if sfx and sfx.strip():
        parts.append(sfx.strip())
    if speech and speech.strip() and speaker_traits and speaker_traits.strip():
        parts.append(f"Speaker: {speaker_traits.strip()}.")
    if mood:
        parts.append(f"Overall audio mood: {mood.lower()}.")
    return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Mock generation (demo mode)
# ---------------------------------------------------------------------------
def _mock_generate(prompt: str, n_steps: int) -> Tuple[str, str]:
    """Simulate generation delay and return placeholder outputs."""
    for i in range(n_steps):
        time.sleep(0.02)
        yield i + 1, None, None, None  # step, video, audio, info  (generator protocol)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_scene(
    scene_idea: str,
    mood: str,
    genre: str,
    shot_type: str,
    time_of_day: str,
    speech_text: str,
    sfx_description: str,
    speaker_traits: str,
    negative_video: str,
    negative_audio: str,
    model_variant: str,
    sample_steps: int,
    video_cfg: float,
    audio_cfg: float,
    seed: int,
    cpu_offload: bool,
    use_demo_mode: bool,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str, str]:
    """
    Core generation function.

    Returns:
        video_path   : path to generated .mp4 (or None in demo mode)
        audio_path   : path to generated .wav (or None in demo mode)
        prompt_text  : formatted OVI prompt shown to user
        info_text    : scene metadata / generation log
    """
    if not scene_idea.strip():
        return None, None, "", "⚠️  Please describe a scene to get started."

    # --- Build prompt components ---
    visual    = build_visual_description(scene_idea, mood, genre, shot_type, time_of_day)
    audio_desc = build_audio_description(sfx_description, mood, speaker_traits, speech_text)
    prompt    = format_ovi_prompt(visual, speech_text or None, audio_desc or None, model_variant)
    neg_prompt_v = negative_video or "jitter, bad hands, blur, distortion"
    neg_prompt_a = negative_audio or "robotic, muffled, echo, distorted"

    info_lines = [
        "**Scene Analysis**",
        f"- Visual:  {visual[:80]}{'...' if len(visual) > 80 else ''}",
        f"- Speech:  {speech_text or '(none)'}",
        f"- Audio:   {audio_desc[:80]}{'...' if len(audio_desc) > 80 else ''}" if audio_desc else "- Audio:   (none)",
        "",
        f"**Generation Settings**",
        f"- Model:   {model_variant}",
        f"- Steps:   {sample_steps}",
        f"- CFG:     video={video_cfg}, audio={audio_cfg}",
        f"- Seed:    {seed}",
    ]

    if use_demo_mode or not OVI_AVAILABLE:
        info_lines += [
            "",
            "ℹ️  **Demo mode** — OVI model weights not loaded.",
            "Install the OVI package and provide `--ckpt_dir` to generate real video.",
            "",
            "The formatted prompt below is ready to use with the real model.",
        ]
        return None, None, prompt, "\n".join(info_lines)

    # --- Real generation via OviFusionEngine ---
    progress(0, desc="Loading OVI engine…")
    engine = OviFusionEngine(
        model_name=model_variant,
        ckpt_dir=os.environ.get("OVI_CKPT_DIR", "./ckpts"),
        cpu_offload=cpu_offload,
    )

    progress(0.1, desc="Encoding prompt…")
    out_dir = Path(tempfile.mkdtemp())

    progress(0.2, desc=f"Generating ({sample_steps} steps)…")
    video_np, audio_np, _ = engine.generate(
        prompt=prompt,
        negative_prompt_video=neg_prompt_v,
        negative_prompt_audio=neg_prompt_a,
        sample_steps=sample_steps,
        video_guidance_scale=video_cfg,
        audio_guidance_scale=audio_cfg,
        seed=seed,
    )

    progress(0.9, desc="Saving outputs…")
    import imageio
    import soundfile as sf

    video_path = str(out_dir / "output.mp4")
    audio_path = str(out_dir / "output.wav")

    imageio.mimwrite(video_path, video_np, fps=24, quality=8)
    sf.write(audio_path, audio_np, 16000)

    info_lines.append("")
    info_lines.append("✅ Generation complete.")
    progress(1.0, desc="Done.")
    return video_path, audio_path, prompt, "\n".join(info_lines)


# ---------------------------------------------------------------------------
# Apply a preset to all relevant UI fields
# ---------------------------------------------------------------------------
def apply_preset(preset_name: str):
    if preset_name not in SCENE_PRESETS:
        return [gr.update()] * 6
    p = SCENE_PRESETS[preset_name]
    return (
        p["scene_idea"],
        p["mood"],
        p["genre"],
        p.get("speech", "") or "",
        p.get("sfx", "") or "",
        "",  # speaker_traits
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CSS = """
.title-block { text-align: center; padding: 20px 0 8px; }
.title-block h1 { font-size: 2.2em; font-weight: 700; margin: 0; }
.title-block p  { color: #666; margin-top: 4px; }
.prompt-box textarea { font-family: monospace; font-size: 0.85em; }
.info-box    textarea { font-size: 0.83em; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 12px;
       font-size: 0.78em; font-weight: 600; margin: 2px; }
"""

DESCRIPTION = """
**Cinematic Story Weaver** uses [OVI](https://github.com/character-ai/Ovi) (Character AI) to
generate **synchronized audio-video clips** from a scene description — no frame-by-frame editing,
no separate audio post-production. Describe a scene, tune the mood, and let the twin backbone
diffusion model compose visuals and sound in a single pass.
"""

with gr.Blocks(css=CSS, title="Cinematic Story Weaver") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="title-block">
      <h1>🎬 Cinematic Story Weaver</h1>
      <p>Powered by <b>OVI</b> — Twin Backbone Cross-Modal Audio-Video Generation</p>
    </div>
    """)
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── Left column: Inputs ─────────────────────────────────────────────
        with gr.Column(scale=3):

            # Scene presets
            with gr.Group():
                gr.Markdown("### 🎭 Quick Start — Scene Presets")
                preset_dd = gr.Dropdown(
                    choices=list(SCENE_PRESETS.keys()),
                    label="Load a preset scene",
                    value=None,
                    interactive=True,
                )

            gr.Markdown("---")

            # Scene description
            with gr.Group():
                gr.Markdown("### 🖊️ Scene Description")
                scene_idea = gr.Textbox(
                    label="What happens in this scene?",
                    placeholder="A woman stands at a cliff edge watching a storm roll in over the sea…",
                    lines=3,
                )
                with gr.Row():
                    mood = gr.Dropdown(
                        label="Mood / Tone",
                        choices=["Auto", "Epic / Awe", "Melancholic", "Tense",
                                 "Peaceful", "Energetic", "Warm / Tender",
                                 "Eerie", "Comedic"],
                        value="Auto",
                    )
                    genre = gr.Dropdown(
                        label="Genre",
                        choices=["Auto", "Drama", "Sci-Fi", "Thriller",
                                 "Romance", "Nature Documentary", "Action",
                                 "Horror", "Comedy", "Music"],
                        value="Auto",
                    )
                with gr.Row():
                    shot_type = gr.Dropdown(
                        label="Shot Type",
                        choices=["Auto", "Wide Shot", "Medium Shot",
                                 "Close-Up", "Extreme Close-Up",
                                 "Aerial / Bird's Eye", "Low Angle"],
                        value="Auto",
                    )
                    time_of_day = gr.Dropdown(
                        label="Time of Day",
                        choices=["Auto", "Dawn", "Morning",
                                 "Midday", "Golden Hour",
                                 "Dusk", "Night"],
                        value="Auto",
                    )

            gr.Markdown("---")

            # Audio details
            with gr.Group():
                gr.Markdown("### 🔊 Audio Details")
                speech_text = gr.Textbox(
                    label="Dialogue / Speech (optional)",
                    placeholder="We should leave before the storm hits.",
                    lines=2,
                )
                sfx_description = gr.Textbox(
                    label="Sound Effects & Ambience",
                    placeholder="Crashing waves, howling wind, seagulls, distant thunder.",
                    lines=2,
                )
                speaker_traits = gr.Textbox(
                    label="Speaker Voice Traits (optional)",
                    placeholder="Female voice, mid-30s, calm but urgent, slight British accent.",
                    lines=1,
                )

            gr.Markdown("---")

            # Advanced settings (collapsible)
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    model_variant = gr.Dropdown(
                        label="OVI Model Variant",
                        choices=["960x960_10s", "960x960_5s", "720x720_5s"],
                        value="960x960_10s",
                    )
                    sample_steps = gr.Slider(
                        label="Sampling Steps",
                        minimum=10, maximum=100, step=5, value=50,
                    )
                with gr.Row():
                    video_cfg = gr.Slider(
                        label="Video CFG Scale",
                        minimum=1.0, maximum=10.0, step=0.5, value=4.0,
                    )
                    audio_cfg = gr.Slider(
                        label="Audio CFG Scale",
                        minimum=1.0, maximum=10.0, step=0.5, value=3.0,
                    )
                seed = gr.Number(label="Seed (-1 = random)", value=42, precision=0)
                with gr.Row():
                    negative_video = gr.Textbox(
                        label="Video Negative Prompt",
                        value="jitter, bad hands, blur, distortion, watermark",
                    )
                    negative_audio = gr.Textbox(
                        label="Audio Negative Prompt",
                        value="robotic, muffled, echo, distorted, clipping",
                    )
                cpu_offload = gr.Checkbox(label="CPU Offload (low VRAM)", value=False)
                demo_mode   = gr.Checkbox(label="Demo Mode (no model weights)", value=True)

            # Generate button
            generate_btn = gr.Button("🎬 Generate Scene", variant="primary", size="lg")

        # ── Right column: Outputs ────────────────────────────────────────────
        with gr.Column(scale=4):
            gr.Markdown("### 🎥 Generated Scene")

            video_out = gr.Video(
                label="Video Output",
                height=400,
                show_label=False,
            )

            with gr.Row():
                audio_out = gr.Audio(
                    label="Audio Output (standalone)",
                    type="filepath",
                )

            gr.Markdown("### 📋 OVI Formatted Prompt")
            prompt_out = gr.Textbox(
                label="Combined prompt sent to T5 encoder",
                lines=5,
                interactive=False,
                elem_classes=["prompt-box"],
                show_copy_button=True,
            )

            gr.Markdown("### ℹ️ Scene Details")
            info_out = gr.Markdown(
                value="_Generation details will appear here after you click Generate._"
            )

    # ── Examples ────────────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 💡 Example Prompts")
    gr.Examples(
        examples=[
            [
                "A pianist performs a melancholic nocturne in an empty concert hall lit by a single spotlight.",
                "Melancholic", "Drama", "Medium Shot", "Night",
                "Every note carries a memory.", "Grand piano, hall reverb, distant rain on windows.",
                "Male, 40s, whispery, contemplative.", "", "", "960x960_10s", 50, 4.0, 3.0, 42, False, True,
            ],
            [
                "A wildlife biologist quietly observes a family of wolves in a snowy clearing at sunrise.",
                "Peaceful", "Nature Documentary", "Wide Shot", "Dawn",
                "", "Wolf howls, wind, crunching snow, distant bird calls.",
                "", "", "", "960x960_10s", 50, 4.0, 3.0, 7, False, True,
            ],
            [
                "Two rival chefs face off in a high-speed cooking battle, sparks flying from their pans.",
                "Energetic", "Action", "Medium Shot", "Auto",
                "This is my kitchen. You don't belong here.", "Sizzling pans, knife chopping, fire burst, crowd gasps.",
                "Female, assertive, sharp.", "", "", "720x720_5s", 50, 4.0, 3.0, 99, False, True,
            ],
        ],
        inputs=[scene_idea, mood, genre, shot_type, time_of_day,
                speech_text, sfx_description, speaker_traits,
                negative_video, negative_audio, model_variant,
                sample_steps, video_cfg, audio_cfg, seed, cpu_offload, demo_mode],
        outputs=[video_out, audio_out, prompt_out, info_out],
        fn=generate_scene,
        cache_examples=False,
    )

    # ── How it works section ─────────────────────────────────────────────────
    with gr.Accordion("📖 How OVI Works", open=False):
        gr.Markdown("""
**OVI** (Character AI, 2025) is a unified audio-video generation model that produces
both modalities in a **single diffusion pass** — no sequential pipeline, no post-hoc alignment.

#### Architecture
```
Combined T5 Prompt Embedding (one encoder for both branches)
         │
┌────────┴────────┐
│  Video DiT      │◄──── blockwise ────►│  Audio DiT      │
│  (Wan2.2 init)  │  cross-attention    │  (trained from  │
│  5B params      │  bidirectional      │   scratch, 5B)  │
└────────┬────────┘                     └────────┬────────┘
  Video Latents (31 frames)              Audio Latents (157 tokens)
  Wan2.2 3D VAE                          MMAudio 1D VAE
         ↓                                        ↓
  Decoded Video (720p/24fps)             Decoded Audio (16kHz → BigVGAN)
```

#### Key Innovations
- **Symmetric twin backbone** — identical architecture eliminates need for projection layers
- **Scaled RoPE** — audio RoPE scaled by 31/157 ≈ 0.197 to align temporal resolution with video
- **Flow Matching** — linear interpolant `z_t = (1-t)·z_noise + t·z_data` with velocity target
- **Combined loss** — `L_total = 0.85·L_video + 0.15·L_audio` with shared timestep
- **Unified T5 prompt** — single text embedding for both modalities strengthens cross-modal coherence

#### Training
| Stage | Data | Steps | Params Trained |
|-------|------|-------|---------------|
| 1: Audio Pretrain | Hundreds of thousands of hours of speech + SFX | 50k | Audio tower (5B) |
| 2: AV Fusion | Millions of paired audio-video clips | 40k | Both attention layers (5.7B / 11B) |

*Paper: [arXiv:2510.01284](https://arxiv.org/abs/2510.01284) · Repo: [character-ai/Ovi](https://github.com/character-ai/Ovi)*
        """)

    # ── Event wiring ──────────────────────────────────────────────────────────
    preset_dd.change(
        fn=apply_preset,
        inputs=[preset_dd],
        outputs=[scene_idea, mood, genre, speech_text, sfx_description, speaker_traits],
    )

    generate_btn.click(
        fn=generate_scene,
        inputs=[
            scene_idea, mood, genre, shot_type, time_of_day,
            speech_text, sfx_description, speaker_traits,
            negative_video, negative_audio,
            model_variant, sample_steps, video_cfg, audio_cfg,
            seed, cpu_offload, demo_mode,
        ],
        outputs=[video_out, audio_out, prompt_out, info_out],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cinematic Story Weaver — OVI Demo")
    parser.add_argument("--demo",      action="store_true", help="Run in demo mode (no model)")
    parser.add_argument("--ckpt_dir",  default="./ckpts",   help="Path to OVI checkpoints")
    parser.add_argument("--share",     action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",      type=int, default=7860)
    args = parser.parse_args()

    if args.ckpt_dir:
        os.environ["OVI_CKPT_DIR"] = args.ckpt_dir

    if not OVI_AVAILABLE and not args.demo:
        print("⚠️  OVI package not found. Launching in demo mode.")
        print("    Install with: pip install -e git+https://github.com/character-ai/Ovi")

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
