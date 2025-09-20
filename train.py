import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import Dataset as HFDataset

DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_DIR = "outputs/model"

class HandwritingDataset(Dataset):
    def __init__(self, images_dir, labels_dir, processor):
        self.images = []
        self.texts = []
        self.processor = processor

        for fname in os.listdir(images_dir):
            if fname.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(images_dir, fname)
                label_path = os.path.join(labels_dir, fname.replace(".jpg", ".txt").replace(".png", ".txt"))

                if not os.path.exists(label_path):
                    continue

                with open(label_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                self.images.append(img_path)
                self.texts.append(text)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        text = self.texts[idx]
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        return {"pixel_values": pixel_values, "labels": text}

def collate_fn(batch, processor):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = processor.tokenizer([item["labels"] for item in batch], padding=True, return_tensors="pt").input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100  # игнор паддингов при лоссе
    return {"pixel_values": pixel_values, "labels": labels}

def main():

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    dataset = HandwritingDataset(IMAGES_DIR, LABELS_DIR, processor)

    hf_dataset = HFDataset.from_dict({
        "pixel_values": [dataset[i]["pixel_values"] for i in range(len(dataset))],
        "labels": dataset.texts
    })

    def preprocess(batch):
        images = [Image.open(img).convert("RGB") for img in batch["pixel_values"]]
        labels = batch["labels"]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        labels_enc = processor.tokenizer(labels, padding="max_length", truncation=True, return_tensors="pt").input_ids
        labels_enc[labels_enc == processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values.squeeze(), "labels": labels_enc.squeeze()}

    hf_dataset = hf_dataset.map(preprocess, batched=True, batch_size=4, remove_columns=["pixel_values", "labels"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # ускорение на GPU
        push_to_hub=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        eval_dataset=hf_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=lambda batch: collate_fn(batch, processor),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f" Модель обучена и сохранена в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
