import json
import torch
from training_data import TrainingData as td
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc = Accelerator() if torch.cuda.is_available() else None
# ------------ Data ------------
data = td.data

label2id = {label: i for i, label in enumerate(set(label for _, label in data))}
id2label = {v: k for k, v in label2id.items()}
texts = [x[0] for x in data]
labels = [label2id[x[1]] for x in data]

# ------------ Tokenizer ------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


# ------------ Dataset Class ------------
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels 

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)







# ------------ Training Model and Running App ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

train_dataset = IntentDataset(train_encodings, labels)
    # test_dataset = IntentDataset(test_encodings, test_labels)

model_instance = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=16,
    per_device_train_batch_size=16 if torch.cuda.is_available() else 4,
    learning_rate=3e-5,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=torch.cuda.is_available(),
    logging_dir="./logs",
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model_instance,
    args=training_args,
    train_dataset=train_dataset
    )

if acc:
    print("🚀 Training with GPU acceleration")
    model_instance, trainer = acc.prepare(model_instance, trainer)
    trainer.train()
    model = acc.unwrap_model(model_instance)
else:
    print("⏳ Training on CPU")
    trainer.train()
    model = model_instance


    
    # -------- Save Model & Tokenizer --------
save_dir = "./intent_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Also save id2label for inference
   
with open(f"{save_dir}/id2label.json", "w") as f:
        
    json.dump(id2label, f)

print("✅ Model and tokenizer saved to", save_dir)

  
