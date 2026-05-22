## This file uses SCRFD model to detect faces in a frame and returns
## a cropped frame containing (boxing) the face.

import logging
from typing import List, Dict, Any
from insightface.app import FaceAnalysis

class SCRFDFaceDetector:
    def __init__(self, det_size = (640,640)):
        self.logger = logging.getLogger(__name__)
        self.det_size = det_size
        try:
            self.model: Any = FaceAnalysis(
                    name = 'buffalo_s',
                    providers = ['CPUExecutionProvider']
                    )
            self.model.prepare(ctx_id = 0, det_size=self.det_size)
            self.logger.info("SCRFD Face detection model initialized")
        except Exception as e:
            self.model = None
            self.logger.info(f"SCRFD face detection model not initialized: {e}")

    def detect_faces(self, frame) -> List[Dict]:
        """
        Detects the largest face in the frame and retus face metadata
        including padded crop.
        """

        results = []
        try:
            if frame is None:
                self.logger.warning("Input frame is None")
                return results

            faces = self.model.get(frame)

            if not faces:
                self.logger.info("No face detected in the frame")
                return results
            
            face = max(faces, key = lambda f: (f.bbox[2] - f.bbox[0])*(f.bbox[3] - f.bbox[1]))
            bbox = face.bbox.astype(int)
            det_score = float(face.det_score)

            if det_score<0.5:
                return results

            x1,y1,x2,y2 = bbox
            h,w = frame.shape[:2]
            pad_x = int((x2-x1)*0.1)
            pad_y = int((y2-y1)*0.1)
            x1 = max(0,x1-pad_x)
            y1 = max(0,y1-pad_y)
            x2 = min(w,x2+pad_x)
            y2 = min(h,y2+pad_y)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size ==0:
                self.logger.info("Cropped face size = 0")
                return results
            results.append({
                "bbox": [x1,y1,x2,y2],
                "det_score": det_score,
                "face_crop": face_crop
                })
            return results
        except Exception as e:
            self.logger.info(f"Error during face detection: {e}")
            return results





        
