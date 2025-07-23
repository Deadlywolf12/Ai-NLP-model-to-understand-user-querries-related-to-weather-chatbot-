from flask import Flask, request, jsonify
import torch
from training_data import TrainingData as td
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

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
    print(f"All probabilities: {probs.tolist()}")
    return id2label[predicted_class], confidence

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("message", "")
    intent, confidence = predict_intent(user_input)

    if confidence < 0.3:
        return jsonify({"reply": "Sorry, I didn't understand that."})

    responses = {
        "greeting": "Hey there!",
        "farewell": "Goodbye!",
        "small_talk": "I'm doing great, thanks for asking!",
        "weather_query": "Let me check the weather for you..."
    }

    reply = responses.get(intent, "Hmm, I don't know how to respond.")
    return jsonify({"reply": reply})

# ------------ Training Model and Running App ------------
if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        logging_dir="./logs",
        save_strategy="no",
        report_to="none"  
    )

    trainer = Trainer(
        model=model_instance,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    model = model_instance

    app.run(host="0.0.0.0", port=5000)