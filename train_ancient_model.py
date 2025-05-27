from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from datasets import load_dataset
import torch

# Debug info
print("GPU Available:", torch.cuda.is_available())
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load dataset using Hugging Face Datasets
dataset = load_dataset("text", data_files={"train": "corpus/all_combined.txt"})

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    save_steps=200,
    save_total_limit=1,
    logging_steps=50,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),           # Mixed precision if on GPU
    report_to="none"                          # Disable WandB/Hub
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("model_output")
tokenizer.save_pretrained("model_output")
