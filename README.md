# CUS-GS Visual Features

CUS-GS Visual Features extracts dense visual embeddings for CUS-GS using CLIP, SigLIP, DINOv2, and LLaMA-V. It also constructs compact feature banks through similarity-based redundancy reduction.

## Installation

Prepare a Linux system with x86_64 architecture and an NVIDIA GPU. Also ensure that `git` and `python3` are installed.

We recommend using Python 3.14 for better performance, but Python >=3.11 is also supported.

1. Clone the repository
```bash
git clone https://github.com/Unukalhai727/CUS-GS-Visual-Features.git
cd CUS-GS-Visual-Features
```

2. Set up the virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install accelerate transformers tqdm
```

Pretrained models are downloaded from Hugging Face on first use. Run `hf auth login` beforehand if authentication is required, for example when using LLaMA-V.

## Usage

The scene directory must contain an `images/` subdirectory. Select an encoder and run:

```bash
python main.py -s <scene_path> -e {clip,siglip,dinov2,llamav}
```

Available encoders are `clip`, `siglip`, `dinov2`, and `llamav`. Per-image embeddings and the reduced feature bank are saved as follows:

```text
<scene>/features/<encoder>/
├── embeds/
│   ├── <image-name>.pth
│   └── ...
└── mb<threshold>.pth
```
