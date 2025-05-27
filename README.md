# ancient-lm

**Ancient Language Model** is a stylized text generator trained on sacred and philosophical texts such as the *Tao Te Ching*, *Analects*, and *Bhagavad Gita*.  
It fine-tunes GPT-2 to produce poetic, meditative passages based on user-provided themes.

---

## Features

- Fine-tuned GPT-2 on ancient and spiritual wisdom
- Prompt-based text generation (e.g., `Theme: silence`)
- Runs on GPU or CPU
- Clean Python architecture for training and generation
- Corpus can be enriched using your own texts or a web crawler

---

## Example Usage

```bash
python generate.py
```

Prompt:

">" Theme: stillness

Output:

— Scroll —
Stillness is the root of clarity. The sage acts by not acting.



## File Structure

corpus/                 # Contains all_combined.txt (training data)
generate.py             # Generate poetic output with prompts
train_ancient_model.py  # Fine-tune GPT-2 on corpus
requirements.txt        # Dependencies (torch, transformers, datasets)



## Installation

pip install -r requirements.txt

Requires Python 3.10+ and PyTorch (with GPU recommended)

To train:

python train_ancient_model.py

To generate:

python generate.py


