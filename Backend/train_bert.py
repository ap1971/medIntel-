import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("training_data.csv")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode labels
label_mapping = {error: idx for idx, error in enumerate(df["errors"].unique())}
df["label"] = df["errors"].map(label_mapping)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(df["form_text"], df["label"], test_size=0.2)

# Tokenize text
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

# Convert to PyTorch dataset
# class FormDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, torch.tensor(self.labels[idx])
class FormDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])  # Ensure labels are added correctly
        return item

train_dataset = FormDataset(train_encodings, train_labels.tolist())
val_dataset = FormDataset(val_encodings, val_labels.tolist())

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model & tokenizer
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

# Save label mapping
joblib.dump(label_mapping, "label_mapping.pkl")

print("Model training complete and saved!")
