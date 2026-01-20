import gradio as gr
import torch
from heartlib import HeartMuLaGenPipeline
import os
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODELS_DIR = "./ckpt"
DEFAULT_LYRICS_PATH = "./assets/lyrics.txt"
DEFAULT_TAGS_PATH = "./assets/tags.txt"

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_default(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

default_lyrics = load_default(DEFAULT_LYRICS_PATH)
default_tags = load_default(DEFAULT_TAGS_PATH)

# Global pipeline cache
pipe = None

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def generate_music(lyrics, tags, version, max_length, topk, temperature, cfg_scale, model_path, progress=gr.Progress()):
    global pipe
    
    if not lyrics.strip():
        yield None, "⚠️ Please enter some lyrics first!"
        return
    
    if not os.path.exists(model_path):
        yield None, f"❌ Model path '{model_path}' does not exist."
        return
    
    try:
        # ─── STAGE 1: Load Model ───
        if pipe is None or getattr(pipe, 'version', None) != version:
            progress(0, desc="Loading model into GPU...")
            yield None, "⏳ Loading model into GPU memory... (First time takes 1-2 minutes)"
            
            start_load = time.time()
            pipe = HeartMuLaGenPipeline.from_pretrained(
                model_path,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                version=version,
            )
            pipe.version = version
            load_time = time.time() - start_load
            print(f"[OK] Model loaded in {load_time:.1f}s")
        
        # ─── STAGE 2: Prepare Files ───
        progress(0.1, desc="Preparing generation...")
        lyrics_file = "temp_lyrics.txt"
        tags_file = "temp_tags.txt"
        
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(lyrics)
        with open(tags_file, "w", encoding="utf-8") as f:
            f.write(tags if tags.strip() else "pop,vocal")
            
        timestamp = int(time.time())
        os.makedirs("./assets", exist_ok=True)
        output_path = os.path.abspath(f"./assets/output_{timestamp}.wav")  # WAV doesn't need torchcodec


        # ─── STAGE 3: Generate ───
        progress(0.2, desc="Generating music...")
        yield None, f"🎵 Generating {max_length}s of music... (Watch terminal for detailed progress)"
        
        start_gen = time.time()
        with torch.no_grad():
            pipe(
                {"lyrics": lyrics_file, "tags": tags_file},
                max_audio_length_ms=max_length * 1000,
                save_path=output_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )
        
        gen_time = time.time() - start_gen
        print(f"[OK] Generation complete in {gen_time:.1f}s")
        
        progress(1.0, desc="Done!")
        yield output_path, f"✅ Generated {max_length}s in {gen_time:.0f}s | Saved to: {output_path}"
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        yield None, f"❌ Error: {error_msg}"

# ═══════════════════════════════════════════════════════════════════════════════
# MODERN DARK THEME CSS
# ═══════════════════════════════════════════════════════════════════════════════
custom_css = """
/* ═══ ROOT VARIABLES ═══ */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #16161f;
    --bg-hover: #1e1e2a;
    --accent-primary: #00d4aa;
    --accent-secondary: #00b894;
    --accent-glow: rgba(0, 212, 170, 0.3);
    --text-primary: #ffffff;
    --text-secondary: #a0a0b0;
    --text-muted: #6b6b7b;
    --border-color: #2a2a3a;
    --border-glow: #00d4aa;
}

/* Hide Gradio footer */
footer {
    display: none !important;
}

/* ═══ GLOBAL STYLES ═══ */
.gradio-container {
    background: var(--bg-primary) !important;
    max-width: 1400px !important;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    overflow-x: hidden !important;
}

body {
    overflow-x: hidden !important;
}


.dark {
    --background-fill-primary: var(--bg-primary) !important;
    --background-fill-secondary: var(--bg-secondary) !important;
}

/* ═══ HEADER ═══ */
.header-container {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    background: linear-gradient(180deg, rgba(0, 212, 170, 0.08) 0%, transparent 100%);
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.logo-text {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #00f5d4 50%, #00d4aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 60px var(--accent-glow);
}

.tagline {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* ═══ CARD STYLES ═══ */
.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
}

.card:hover {
    border-color: rgba(0, 212, 170, 0.3) !important;
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4), 0 0 20px var(--accent-glow) !important;
}

/* ═══ SECTION LABELS ═══ */
.section-label {
    color: var(--accent-primary) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-bottom: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

.section-label::before {
    content: '';
    width: 3px;
    height: 16px;
    background: var(--accent-primary);
    border-radius: 2px;
}

/* ═══ TEXTAREAS ═══ */
textarea, input[type="text"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ═══ SLIDERS ═══ */
input[type="range"] {
    accent-color: var(--accent-primary) !important;
}

.slider-container {
    background: var(--bg-secondary) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ═══ GENERATE BUTTON ═══ */
.generate-btn {
    background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%) !important;
    border: none !important;
    color: #000 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 16px 32px !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}

.generate-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(0, 212, 170, 0.5) !important;
}

.generate-btn:active {
    transform: translateY(0) !important;
}

/* ═══ AUDIO PLAYER ═══ */
audio {
    width: 100% !important;
    border-radius: 12px !important;
    background: var(--bg-secondary) !important;
}

.audio-container {
    background: var(--bg-secondary) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    border: 1px solid var(--border-color) !important;
}

/* ═══ STATUS BOX ═══ */
.status-box {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
}

.status-box textarea {
    background: transparent !important;
    border: none !important;
    color: var(--accent-primary) !important;
    font-size: 13px !important;
}

/* ═══ ACCORDION ═══ */
.accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    margin-top: 1rem !important;
}

.info-section {
    max-height: 300px;
    overflow-y: auto;
    padding-right: 0.5rem;
}

.info-section::-webkit-scrollbar {
    width: 6px;
}

.info-section::-webkit-scrollbar-track {
    background: var(--bg-secondary);
    border-radius: 3px;
}

.info-section::-webkit-scrollbar-thumb {
    background: var(--accent-primary);
    border-radius: 3px;
}

.info-section h4 {
    color: var(--accent-primary) !important;
    font-size: 0.9rem !important;
    margin: 1rem 0 0.5rem 0 !important;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-color);
}

.info-section h4:first-child {
    margin-top: 0 !important;
}

.info-section p, .info-section li {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
    margin-bottom: 0.4rem;
}

.info-section code {
    background: var(--bg-primary) !important;
    color: var(--accent-primary) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}

.info-section ul {
    margin: 0.5rem 0;
    padding-left: 1.2rem;
}

/* ═══ TIPS CARD ═══ */
.tips-card {
    background: linear-gradient(135deg, rgba(0, 212, 170, 0.05) 0%, rgba(0, 184, 148, 0.02) 100%) !important;
    border: 1px solid rgba(0, 212, 170, 0.2) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    margin-top: 1rem !important;
}

.tips-card h4 {
    color: var(--accent-primary) !important;
    margin-bottom: 0.75rem !important;
    font-size: 0.9rem !important;
}

.tips-card p, .tips-card li {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
}

/* ═══ MARKDOWN ═══ */
.prose h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.prose p, .prose li {
    color: var(--text-secondary) !important;
}

.prose code {
    background: var(--bg-secondary) !important;
    color: var(--accent-primary) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}

/* ═══ ANIMATIONS ═══ */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px var(--accent-glow); }
    50% { box-shadow: 0 0 40px var(--accent-glow); }
}

.generating {
    animation: pulse-glow 2s ease-in-out infinite;
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(css=custom_css, title="Regalune", theme=gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#0a0a0f",
    body_background_fill_dark="#0a0a0f",
    block_background_fill="#16161f",
    block_background_fill_dark="#16161f",
    input_background_fill="#12121a",
    input_background_fill_dark="#12121a",
)) as demo:
    
    # ─── HEADER ───
    gr.HTML("""
        <style>
            @keyframes text-shimmer {
                0% { background-position: -200% center; }
                100% { background-position: 200% center; }
            }
            @keyframes glow-pulse {
                0%, 100% { text-shadow: 0 0 20px rgba(0, 212, 170, 0.5), 0 0 40px rgba(0, 212, 170, 0.3); }
                50% { text-shadow: 0 0 30px rgba(0, 212, 170, 0.7), 0 0 60px rgba(0, 212, 170, 0.4); }
            }
            .header-inner {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0;
            }
            .logo-text {
                font-size: 3.4rem !important;
                font-weight: 800 !important;
                background: linear-gradient(135deg, #ffd700 0%, #fff8dc 20%, #ffd700 40%, #b8860b 60%, #ffd700 80%, #fff8dc 100%);
                background-size: 300% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: 6px;
                font-family: 'Cinzel', 'Times New Roman', serif;
                text-shadow: 0 0 30px rgba(255, 215, 0, 0.4);
                animation: text-shimmer 5s linear infinite;
                margin-bottom: 0.4rem !important;
                filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
            }
            .tagline {
                color: #a0a0b0 !important;
                font-size: 1rem !important;
                font-weight: 400;
                letter-spacing: 0.5px;
                margin-bottom: 0.5rem !important;
            }
            .credit-text {
                color: #4a4a5a;
                font-size: 0.7rem;
                letter-spacing: 1px;
                margin-top: 0.3rem;
            }
            .credit-text a {
                color: #5a5a6a;
                text-decoration: none;
            }
            .credit-text a:hover {
                color: #00d4aa;
            }
        </style>
        <div class="header-container">
            <div class="header-inner">
                <div class="logo-text">REGALUNE</div>
                <div class="tagline">AI-Powered Music Generation from Lyrics</div>
                <div class="credit-text">Built on <a href="https://github.com/HeartMuLa/heartlib" target="_blank">HeartMuLa</a></div>
            </div>
        </div>
    """)


    
    with gr.Row(equal_height=False):
        # ═══ LEFT COLUMN: INPUTS ═══
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">LYRICS</div>')
            lyrics_input = gr.Textbox(
                label="",
                lines=14,
                value=default_lyrics,
                placeholder="[Intro]\n\n[Verse 1]\nYour lyrics here...\n\n[Chorus]\nCatchy chorus...\n\n[Outro]",
                show_label=False,
                container=False,
            )
            
            gr.HTML('<div class="section-label" style="margin-top: 1.25rem;">STYLE TAGS</div>')
            tags_input = gr.Textbox(
                label="",
                lines=2,
                value=default_tags,
                placeholder="pop, female vocal, piano, emotional, 120bpm",
                show_label=False,
                container=False,
            )
            
            gr.HTML('<div class="section-label" style="margin-top: 1.25rem;">DURATION</div>')
            max_length_slider = gr.Slider(
                label="",
                minimum=10,
                maximum=240,
                value=30,
                step=10,
                show_label=False,
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    topk_slider = gr.Slider(label="Top-k Sampling", minimum=1, maximum=100, value=50, step=1)
                    temp_slider = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                
                with gr.Row():
                    cfg_slider = gr.Slider(label="CFG Scale", minimum=1.0, maximum=5.0, value=1.5, step=0.1)
                    version_dropdown = gr.Dropdown(label="Model", choices=["3B"], value="3B")
                
                model_path_input = gr.Textbox(label="Model Path", value=DEFAULT_MODELS_DIR, visible=False)
            
            generate_btn = gr.Button(
                "✦ GENERATE MUSIC",
                variant="primary",
                size="lg",
                elem_classes=["generate-btn"]
            )
        
        # ═══ RIGHT COLUMN: OUTPUT ═══
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">OUTPUT</div>')
            audio_output = gr.Audio(
                label="",
                type="filepath",
                show_label=False,
            )
            
            gr.HTML('<div class="section-label" style="margin-top: 1.25rem;">STATUS</div>')
            status_output = gr.Textbox(
                label="",
                lines=5,
                interactive=False,
                show_label=False,
                value="Ready. Click Generate to create music.",
                elem_classes=["status-box"]
            )
            
            # ─── INFO/TIPS ACCORDION ───
            with gr.Accordion("📖 Usage Guide & Tips", open=False):
                gr.HTML("""
                    <div class="info-section">
                        <h4>🎤 Writing Lyrics</h4>
                        <p>Structure your lyrics with section markers for best results:</p>
                        <ul>
                            <li><code>[Intro]</code> - Instrumental opening</li>
                            <li><code>[Verse]</code> or <code>[Verse 1]</code>, <code>[Verse 2]</code> - Main story sections</li>
                            <li><code>[Pre-Chorus]</code> - Build-up before chorus</li>
                            <li><code>[Chorus]</code> - The catchy, repeating hook</li>
                            <li><code>[Bridge]</code> - A contrasting section</li>
                            <li><code>[Outro]</code> - Closing section</li>
                        </ul>
                        <p>Keep lyrics poetic and rhythmic. Avoid overly long lines.</p>
                        
                        <h4>🏷️ Style Tags</h4>
                        <p>Comma-separated descriptors that guide the music style:</p>
                        <ul>
                            <li><strong>Genre:</strong> <code>pop</code>, <code>rock</code>, <code>jazz</code>, <code>electronic</code>, <code>folk</code>, <code>r&b</code></li>
                            <li><strong>Mood:</strong> <code>happy</code>, <code>sad</code>, <code>energetic</code>, <code>melancholic</code>, <code>romantic</code></li>
                            <li><strong>Instruments:</strong> <code>piano</code>, <code>guitar</code>, <code>synthesizer</code>, <code>drums</code>, <code>strings</code></li>
                            <li><strong>Vocals:</strong> <code>male vocal</code>, <code>female vocal</code>, <code>choir</code></li>
                            <li><strong>Tempo:</strong> <code>slow</code>, <code>fast</code>, <code>upbeat</code></li>
                        </ul>
                        <p>Example: <code>indie pop, female vocal, acoustic guitar, dreamy, romantic</code></p>
                        
                        <h4>⏱️ Duration Guide</h4>
                        <ul>
                            <li><strong>10-30 seconds:</strong> Quick tests, previews</li>
                            <li><strong>60 seconds:</strong> Short song demo</li>
                            <li><strong>120-180 seconds:</strong> Full song length</li>
                            <li><strong>240 seconds:</strong> Extended track (4 minutes max)</li>
                        </ul>
                        
                        <h4>⚡ Performance Notes</h4>
                        <ul>
                            <li><strong>First run:</strong> Model loads into GPU memory (~1-2 minutes)</li>
                            <li><strong>Generation speed:</strong> ~30-40 seconds of compute per 1 second of audio</li>
                            <li><strong>Subsequent runs:</strong> Faster (model stays cached)</li>
                            <li><strong>Tip:</strong> Start with short clips (10-30s) to test your lyrics and tags</li>
                        </ul>
                        
                        <h4>⚙️ Advanced Settings Explained</h4>
                        <ul>
                            <li><strong>Top-k:</strong> Controls diversity. Lower = more focused, Higher = more varied</li>
                            <li><strong>Temperature:</strong> Creativity level. 0.7-1.0 is balanced, >1.2 is experimental</li>
                            <li><strong>CFG Scale:</strong> How closely to follow your tags. Higher = stricter adherence</li>
                        </ul>
                        
                        <h4>💻 Command Line Usage</h4>
                        <p>You can also generate music via CLI:</p>
                        <p><code>py -3.10 ./examples/run_music_generation.py --model_path=./ckpt --version="3B"</code></p>
                        <p>Add <code>--lyrics "path/to/lyrics.txt"</code> and <code>--tags "pop,piano"</code> for custom input.</p>
                    </div>
                """)

    # ─── EVENT HANDLERS ───
    generate_btn.click(
        fn=generate_music,
        inputs=[
            lyrics_input, tags_input, version_dropdown,
            max_length_slider, topk_slider, temp_slider,
            cfg_slider, model_path_input
        ],
        outputs=[audio_output, status_output]
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  * Regalune - AI Music Generator")
    print("    Powered by HeartMuLa")
    print("="*60)
    print("  Starting server...")
    print("="*60 + "\n")
    demo.launch(share=False, show_error=True)
