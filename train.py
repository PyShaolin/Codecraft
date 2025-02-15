import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

dataset = load_dataset("IAM")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

def preprocess_batch(batch):
    images = [processor(Image.open(img).convert("RGB"), return_tensors="pt").pixel_values for img in batch["image"]]
    labels = processor.tokenizer(batch["text"], padding="max_length", max_length=128, return_tensors="pt").input_ids
    return {"pixel_values": torch.cat(images), "labels": labels}

dataset = dataset.map(preprocess_batch, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=3,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./models/fine_tuned_trocr")
