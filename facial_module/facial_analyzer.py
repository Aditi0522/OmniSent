## Heart of the module, class which links all the functions related
## to facial emotion detection.load_model

## 1. Open video
## 2. Create TemporalSmoother + EmotionTimeline + UsabilityMetricsCalculator
## 3. Frame loop:
##   a. Read frame, skip based on process_fps
##   b. Detect face (SCRFD) → get face_crop
##   c. Predict emotions (your EfficientNet-B0) → 7-class probabilities
##   d. Feed into timeline (smoothing happens inside)
##   e. Compute UX metrics for this frame
##   f. Detect UX events (frustration spike, etc.)
## 4. Build segments from timeline
## 5. Aggregate summary
## 6. Return the full result dict

import os
import cv2
import logging
from pathlib import Path
from utils.utils import load_model
from insightface.app import FaceAnalysis

class FacialEmotionAnalyzer:
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        try:
            self.model = FaceAnalysis(
                    name = 'buffalo_s',
                    providers = ['CPUExecutionProvider']
                    )
            self.model.prepare(ctx_id=0, det_size=(640,640))
            self.logger.info("Scrfd face detection model loaded successfully")

        except Exception as e:
            self.logger.info(f"Failed to load SCRFD face detector: {e}")

        try:
            self.emotion_det_model = load_model(model_path)

        except Exception as e:
            self.logger.info(f"Failed to load Facial Emotion detection model: {e}")


    def analyze_video(self, video_path: str, output_dir):
        os.makedirs(output_dir, exist_ok = True)
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
           self.logger.error(f"video cannot be opened: {video_path}")
           return
        
        video_name = Path(video_path).stem
    
        last_processed_second = -1
        frame_count = 0
        saved_faces = 0

        while True:
            
            ret, frame = video.read()
            if not ret: break
            timestamp_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            current_second = int(timestamp_ms/500)  # 2 frame-sp
            if current_second == last_processed_second:
                continue
            last_processed_second = current_second
 
            frame_count+=1
 
            scale_factor = 1.5
            frame = cv2.resize(
                    frame, None,
                    fx = scale_factor,
                    fy = scale_factor,
                    interpolation = cv2.INTER_CUBIC
                    )
            faces = self.model.get(frame)
            if len(faces) == 0:
                self.logger.info("No faces detected")
                continue

            for i,face in enumerate(faces):
                det_score = getattr(face, 'det_score', 0)
                if det_score < 0.5:
                    self.logger.info(f"det_score < 0.5: {det_score}")
                    continue
                x1,y1,x2,y2 = face.bbox.astype(int)
                margin_w = int((x2-x1) * 0.4)
                margin_h = int((y2 - y1) * 0.4)

                x1_new = max(0,x1-margin_w)
                y1_new = max(0,y1-margin_h)
                x2_new = min(frame.shape[1], x2 + margin_w)
                y2_new = min(frame.shape[0], y2 + margin_h)

                face_crop = frame[y1_new:y2_new, x1_new:x2_new]
                
                if face_crop.size == 0:
                    self.logger.info("no face cropped")
                    continue

                face_crop = cv2.resize(face_crop, (224,224))
                save_path = os.path.join(
                        output_dir,
                        f"frame{frame_count}_face{i}.jpg"
                        )
                cv2.imwrite(save_path,face_crop)
                saved_faces+=1

        video.release()

        self.logger.info(f"Processed video: {video_name}")
        self.logger.info(f"Frames processed: {frame_count}")
        self.logger.info(f"Faces saved: {saved_faces}")   

    def analyze_frame(self):
        pass

    def get_usability_metrics(self):
        pass





