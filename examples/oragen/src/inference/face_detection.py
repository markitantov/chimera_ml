from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
from PIL import Image

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference import InferenceContext


@dataclass
class FaceDetectionStep:
    model: str | dict[str, Any] = "yolo26n.pt"
    fps: float = 1.0
    conf: float = 0.35
    expand_ratio: float = 0.1
    _detector: Any = field(default=None, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        ctx.set_artifact("fps", float(self.fps))
        cache_dir = ctx.get_artifact("cache_dir")
        detector = self._load_detector(Path(cache_dir))

        capture = cv2.VideoCapture(str(ctx.input_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {ctx.input_path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = max(float(self.fps), 1.0)

        frame_interval = max(round(source_fps / max(float(self.fps), 1e-6)), 1)
        frames: list[dict[str, Any]] = []
        faces: list[dict[str, Any]] = []
        frame_index = 0
        sampled_index = 0

        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                if frame_index % frame_interval != 0:
                    frame_index += 1
                    continue

                timestamp = float(frame_index) / source_fps
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)
                frames.append(
                    {
                        "index": sampled_index,
                        "timestamp": timestamp,
                        "image": frame_image,
                        "source_frame_index": frame_index,
                    }
                )

                best_face = self._detect_best_face(
                    detector=detector,
                    image_rgb=frame_rgb,
                    image=frame_image,
                    timestamp=timestamp,
                    frame_index=sampled_index,
                    device=ctx.device,
                )

                if best_face is not None:
                    faces.append(best_face)

                sampled_index += 1
                frame_index += 1
        finally:
            capture.release()

        ctx.set_artifact("frames", frames)
        ctx.set_artifact("faces", faces)
        ctx.set_artifact(
            "selected_face_track",
            {"mode": "best_face_per_frame", "num_faces": len(faces)},
        )
        return ctx

    def _detect_best_face(
        self,
        *,
        detector: Any,
        image_rgb: Any,
        image: Image.Image,
        timestamp: float,
        frame_index: int,
        device: str,
    ) -> dict[str, Any] | None:
        results = detector.predict(
            source=image_rgb,
            conf=float(self.conf),
            verbose=False,
            device=device,
        )

        result = results[0]
        if result.boxes is None:
            return None

        image_height, image_width = image_rgb.shape[0], image_rgb.shape[1]
        best_face: dict[str, Any] | None = None
        best_rank: tuple[int, float] | None = None

        for box, _, score in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            strict=False,
        ):
            x1, y1, x2, y2 = [round(value) for value in box]
            box_width = max(x2 - x1, 1)
            box_height = max(y2 - y1, 1)
            expand_x = round(box_width * self.expand_ratio)
            expand_y = round(box_height * self.expand_ratio)
            clipped_box = {
                "x1": max(0, x1 - expand_x),
                "y1": max(0, y1 - expand_y),
                "x2": min(image_width, x2 + expand_x),
                "y2": min(image_height, y2 + expand_y),
            }

            area = max(clipped_box["x2"] - clipped_box["x1"], 0) * max(clipped_box["y2"] - clipped_box["y1"], 0)
            rank = (area, float(score))
            if best_rank is not None and rank <= best_rank:
                continue

            best_rank = rank
            best_face = {
                "frame_index": int(frame_index),
                "timestamp": float(timestamp),
                "crop_image": image.crop((clipped_box["x1"], clipped_box["y1"], clipped_box["x2"], clipped_box["y2"])),
                "score": float(score),
                "box": clipped_box,
            }

        return best_face

    def _load_detector(self, cache_dir: Path) -> Any:
        if self._detector is not None:
            return self._detector

        from ultralytics import YOLO, settings
        from ultralytics.utils.downloads import attempt_download_asset, safe_download

        cache_dir.mkdir(parents=True, exist_ok=True)
        settings.update({"weights_dir": str(cache_dir)})
        model_ref = self.model
        if isinstance(model_ref, dict):
            model_ref = model_ref.get("url") or model_ref.get("path")
            if model_ref is None:
                raise ValueError("face_detection_step model dict must contain 'url' or 'path'.")

        print(f"[inference] face_detection_step: resolving detector weights {model_ref}")
        local_path = Path(str(model_ref)).expanduser()
        parsed = urlparse(str(model_ref))
        if local_path.exists():
            model_path = local_path
        elif parsed.scheme in {"http", "https"}:
            model_path = cache_dir / Path(parsed.path).name
            safe_download(url=str(model_ref), file=model_path, min_bytes=1e5)
        else:
            model_path = Path(str(attempt_download_asset(cache_dir / str(model_ref)))).expanduser()

        self._detector = YOLO(str(model_path))
        return self._detector


@INFERENCE_STEPS.register("face_detection_step")
def face_detection_step(**params: Any) -> FaceDetectionStep:
    return FaceDetectionStep(**params)
