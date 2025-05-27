import warnings
warnings.filterwarnings("ignore")

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("model_output")
tokenizer = GPT2Tokenizer.from_pretrained("model_output")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_length=40):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=1.1,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    lines = decoded.split("\n")
    seen = set()
    cleaned = "\n".join(
        line.strip() for line in lines
        if line.strip() and line not in seen and not seen.add(line)
    )
    return cleaned

# CLI prompt
if __name__ == "__main__":
    print("╔══════════════════════════════╗")
    print("║  Ancient Wisdom Generator    ║")
    print("╚══════════════════════════════╝")
    print("Enter a theme (e.g. 'Theme: stillness') or 'q' to quit.")

    while True:
        prompt = input("\n> ")
        if prompt.strip().lower() == "q":
            break
        result = generate_text(prompt)
        print("\n— Scroll —\n" + result + "\n")