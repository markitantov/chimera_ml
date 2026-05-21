from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from feature_extractors.video_models import ResEmoteNet, ResNet50
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


class EmoAffectNetPreprocessInput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = torch.flip(x, dims=(0,))
        x[0, :, :] -= 91.4953
        x[1, :, :] -= 103.8827
        x[2, :, :] -= 131.0912
        return x


class VideoFeatureExtractor:
    def __init__(
        self,
        win_max_length: int = 3,
        target_fps: int = 10,
        model_name: str = "emoaffectnet",
        face_model_name: str = "models/yolov8n-face.pt",
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.win_max_length = int(win_max_length)
        self.target_fps = int(target_fps)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name

        if "emoaffectnet" in self.model_name.lower():
            self.preprocessor = transforms.Compose(
                [
                    transforms.Resize((224, 224), interpolation=Image.Resampling.NEAREST),
                    transforms.PILToTensor(),
                    EmoAffectNetPreprocessInput(),
                ]
            )

            self.model = ResNet50(num_classes=7, channels=3)
            checkpoint_source = Path(checkpoint_path) if checkpoint_path is not None else Path("models/emoaffectnet.pt")
            checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint)
            self._input_shape = (3, 224, 224)

            self.output_shape = (
                int(self.target_fps * self.win_max_length),
                512,
            )
        elif "resemotenet" in self.model_name.lower():
            self.preprocessor = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            self.model = ResEmoteNet()
            checkpoint_source = Path(checkpoint_path) if checkpoint_path is not None else Path("models/resemotenet.pt")
            checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._input_shape = (3, 64, 64)

            self.output_shape = (
                int(self.target_fps * self.win_max_length),
                256,
            )
        else:
            raise NotImplementedError

        self.model = self.model.to(self.device).eval()

        self.face_model_name = str(face_model_name)
        self.face_model: YOLO | None = None

        self._current_video_path: str | None = None
        self._video_fps: float = 0.0
        self._frame_interval: int = 1
        self._frame_features: dict[int, np.ndarray] = {}
        self._zero_features = self._extract_zeros_features()

    def _extract_zeros_features(self) -> np.ndarray:
        zeros = torch.zeros((1, *self._input_shape), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            features = self.model.extract_features(zeros)

        return features.detach().cpu().numpy()[0]

    def _prepare_video(self, file_path: str) -> None:
        if self._current_video_path == file_path:
            return

        if self.face_model is None:
            self.face_model = YOLO(self.face_model_name)

        self._current_video_path = file_path
        self._frame_features = {}

        cap = cv2.VideoCapture(file_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self._video_fps = float(fps if fps > 0 else self.target_fps)
        self._frame_interval = max(1, int(self._video_fps / max(self.target_fps, 1)))

        if hasattr(self.face_model, "predictor") and hasattr(self.face_model.predictor, "trackers"):
            trackers = self.face_model.predictor.trackers
            if trackers:
                trackers[0].reset()

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % self._frame_interval == 0:
                frame_features = self._extract_frame_features(frame)
                if frame_features is None:
                    # Preserve the original notebook behavior for consecutive frames without faces.
                    frame_features = self._frame_features.get(frame_index - 1, self._zero_features).copy()

                self._frame_features[frame_index] = frame_features

            frame_index += 1

        cap.release()

    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_model.track(
            rgb_frame,
            persist=True,
            imgsz=640,
            conf=0.01,
            iou=0.5,
            augment=False,
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes if results else None
        if boxes is None or boxes.xyxy.cpu().tolist() == []:
            return None

        height, width = rgb_frame.shape[:2]
        frame_features = np.zeros(self.output_shape[1], dtype=np.float32)
        face_count = 0

        for box in boxes:
            coords = box.xyxy.int().cpu().tolist()[0]
            start_x, start_y = max(0, coords[0]), max(0, coords[1])
            end_x, end_y = min(width - 1, coords[2]), min(height - 1, coords[3])
            if start_x >= end_x or start_y >= end_y:
                continue

            face_region = rgb_frame[start_y:end_y, start_x:end_x]
            if face_region.size == 0:
                continue

            model_input = self.preprocessor(Image.fromarray(face_region)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.extract_features(model_input)

            frame_features += features.detach().cpu().numpy()[0]
            face_count += 1

        if face_count == 0:
            return None

        return frame_features / float(face_count)

    def __call__(self, window: dict[str, Any]) -> torch.Tensor:
        file_path = window["video_path"]
        if not file_path:
            return torch.zeros(self.output_shape, dtype=torch.float32)

        file_path = str(file_path)
        if not Path(file_path).exists():
            raise FileNotFoundError(file_path)

        self._prepare_video(file_path)

        start_frame = int(float(window["start_sec"]) * self._video_fps)
        end_frame = int(float(window["end_sec"]) * self._video_fps)
        selected = [
            features for frame_idx, features in self._frame_features.items() if start_frame <= frame_idx < end_frame
        ]

        if not selected:
            features = np.zeros(self.output_shape, dtype=np.float32)
        else:
            features = np.asarray(selected, dtype=np.float32)
            if features.shape[0] < self.output_shape[0]:
                pad = np.repeat(features[-1][None, :], self.output_shape[0] - features.shape[0], axis=0)
                features = np.concatenate((features, pad), axis=0)
            elif features.shape[0] > self.output_shape[0]:
                features = features[: self.output_shape[0]]

        return torch.as_tensor(features, dtype=torch.float32)
