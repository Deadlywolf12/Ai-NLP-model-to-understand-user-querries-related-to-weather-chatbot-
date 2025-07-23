import random
from flask import Flask, request, jsonify
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
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

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

# ------------ Flask App Setup ------------
app = Flask(__name__)
model = None

# ------------ Prediction Function ------------
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    confidence = probs[0][predicted_class].item()

     # Debug: Print probabilities
    print(f"Predicted class: {predicted_class} ({id2label[predicted_class]})")
    print(f"Confidence: {confidence:.4f}")
    
    return id2label[predicted_class], confidence

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("message", "")
    intent, confidence = predict_intent(user_input)

    if confidence < 0.3:
        return jsonify({"reply": "Sorry, I didn't understand that."})

    responses = {
        "greeting": "Hey there!, how are you?",
        "farewell": "Goodbye! see you",
        "small_talk": "I'm doing great, thanks for asking!",
        "weather_query": "Let me check the weather for you...",
        "agree_query": "Good to know you got it! want to ask anything else?",
        "rain_query": "Look like no raining and no chance of rain...",
        "temp_query": "letme get temperature...",
        "location_specific_weather_query": "letme get weather of another city...",
        "clothing_advice": "letme get clothing advice...",
        "humid_query": "letme get humid advice...",
        "wind_query": "letme get wind advice...",
        "sun_query": "letme get sun advice...",
        "air_query": "letme get air advice...",
        "walk_query": "letme get walk advice...",
        "cycle_query": "letme get cycle advice...",
        "outing_query": "letme get outing advice...",
        "help": "how can i help you?...",
        "neg_res": "oky i got it , nothing u need",
        "mood_good": "happy to know you are in good mood",
        "mood_bad": "sadend to know you are in bad mood",
        "confused": "im sorry it made you confused dumbhead"
    }

    reply = responses.get(intent, "Hmm, I don't know how to respond.")
    return jsonify({"reply": reply})

# ------------ Training Model and Running App ------------
if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dataset = IntentDataset(encodings, labels)

    model_instance = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=15,
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
        train_dataset=dataset
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


    model.eval()

    app.run(host="0.0.0.0", port=5000)