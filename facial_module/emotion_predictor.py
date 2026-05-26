## Predicts emotions for for every cropped frame

import cv2
import torch
import logging
import numpy as np
from PIL import Image
from torch.cuda import is_available
from torchvision import transforms
from .model import EmotiEffLibFineTuned, load_model
from utils.utils import EMOTIONS

class EmotionPredictor:
    def __init__(self, model_path: str, device:torch.device = None):
        self.logger = logging.getLogger(__name__)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path,self.device)
        self.preprocess =  transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
    ])

    def preprocess_face(self,face_bgr) -> torch.Tensor:
        """
        Convert an OpenCV BGR face crop to a model-ready tensor.
        Args:
           face_bgr: numpy array (H, W, 3) in BGR color order (from OpenCV)
        Returns:
           tensor: (1, 3, 224, 224) normalized tensor ready for model
        """
        face_rgb = cv2.cvtColor(face_bgr,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        tensor = self.preprocess(pil_image)
        return tensor.unsqueeze(0)  #(1,3,224,224)

    def predict_emotion(self,face_bgr: np.ndarray) -> tuple[dict,float]:
        try:
            tensor = self.preprocess_face(face_bgr).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits,dim=1)[0]
            emotions = {EMOTIONS[i]: round(float(probs[i]),4) for i in range(len(EMOTIONS))}
            confidence = float(probs.max())
            return emotions, confidence
        except Exception as e:
            self.logger.error(f"Error predicting emotions: {e}")
            fallback = {e:round(1.0/7,4) for e in EMOTIONS}
            return fallback, 0.0
        

