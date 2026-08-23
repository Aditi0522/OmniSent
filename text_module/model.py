import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#replace with fine tuned model later
PRETRAINED_MODEL = "j-hartmann/emotion-english-distilroberta-base"

class TextEmotionClassifier:
    def __init__(self, model_name=PRETRAINED_MODEL, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # This model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
        self.label_map = {
            'anger': 'Angry', 'disgust': 'Disgusted', 'fear': 'Fearful',
            'joy': 'Happy', 'neutral': 'Neutral', 'sadness': 'Sad', 'surprise': 'Surprised'
        }

    def predict(self, text: str) -> tuple[dict, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
        id2label = self.model.config.id2label
        emotions = {}
        for i, p in enumerate(probs):
            orig = id2label[i]
            mapped = self.label_map.get(orig, orig)
            emotions[mapped] = round(float(p), 4)
        confidence = float(probs.max())
        return emotions, confidence

