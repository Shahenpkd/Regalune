# Regalune

**Regalune** is a non-commercial fork of the original [HeartMuLa](https://github.com/HeartMuLa/heartlib) project.
This edition introduces a full graphical user interface and includes internal fixes
and improvements for modern GPU environments.

---

## About This Project

The original HeartMuLa project did not include a graphical user interface.
Regalune was created to provide a user-friendly way to interact with the HeartMuLa
music generation model, along with compatibility fixes for modern hardware.

This project is intended for personal, educational, and experimental use only.

---

## Features

- 🎨 **Full Graphical User Interface** – Generate music visually without command line
- 🖥️ **Compatibility fixes for RTX 50-series GPUs** (Blackwell architecture)
- 🔧 **Windows Audio Fix** – Reliable audio saving on Windows
- 📖 **Built-in Usage Guide** – Comprehensive tips within the UI
- 🎵 **High-Quality Music Generation** – Powered by HeartMuLa 3B model

---

## Screenshots

![Regalune UI](./assets/screenshot.png)

---

## Requirements

- Python 3.10
- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- Windows 10/11 (Linux untested but should work)

---

## Installation

### 1. Clone and download models

```bash
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib

# Download models from Hugging Face
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B'
hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss'
```

### 2. Install dependencies

```bash
pip install -e .
pip install gradio soundfile
```

### 3. For RTX 50-series GPUs (required for Blackwell architecture)

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. Run the UI

```bash
python gradio_app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Usage

1. **Lyrics**: Write song lyrics with section markers like `[Verse]`, `[Chorus]`, `[Bridge]`
2. **Style Tags**: Add comma-separated descriptors like `pop, female vocal, piano, emotional`
3. **Duration**: Set length in seconds (start with 10-30s for testing)
4. **Advanced Settings**: Adjust Top-k, Temperature, and CFG Scale for different results
5. **Generate**: Click the button and wait (first run loads the model ~1-2 min)

---

## Attribution

Regalune is built on top of **HeartMuLa**, originally created by the HeartMuLa authors.

- **Original Repository**: https://github.com/HeartMuLa/heartlib
- **Research Paper**: https://arxiv.org/abs/2601.10547

All original model code and concepts remain credited to their respective authors.

---

## License

This project is released under the **Creative Commons Attribution–NonCommercial 4.0
International (CC BY-NC 4.0)** license, in accordance with the original HeartMuLa project.

- ✅ You are free to share and adapt this work
- ✅ Attribution must be provided
- ❌ Commercial use is **not permitted**

License details: https://creativecommons.org/licenses/by-nc/4.0/

---

## Disclaimer

Regalune is an independent project and is **not officially affiliated with or endorsed by**
the original HeartMuLa project or its maintainers.

---

## Credits

- **Original Model & Research**: [HeartMuLa Team](https://github.com/HeartMuLa)
- **Regalune UI & Fixes**: [Shahenpkd](https://github.com/Shahenpkd)
