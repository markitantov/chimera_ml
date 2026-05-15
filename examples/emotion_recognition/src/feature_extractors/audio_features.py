import torch
from feature_extractors.audio_models import W2V2, ExHuBERT
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor


class AudioFeatureExtractor:
    def __init__(
        self,
        sr: int = 16000,
        win_max_length: int = 3,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ) -> None:
        self.sr = sr
        self.win_max_length = win_max_length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = str(model_name)

        if "wav2vec2" in self.model_name.lower():  # audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
            self.processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
            self.model = W2V2.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
        elif "exhubert" in self.model_name.lower():  # amiriparian/ExHuBERT
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.model = ExHuBERT.from_pretrained(
                "amiriparian/ExHuBERT", trust_remote_code=True, revision="b158d45ed8578432468f3ab8d46cbe5974380812"
            )
        else:
            raise NotImplementedError

        self.model = self.model.to(self.device)

        self.output_shape = (
            int(self._infer_feature_length()),
            int(self.model.config.hidden_size),
        )

    def _infer_feature_length(self) -> int:
        input_length = int(self.sr * self.win_max_length)
        if hasattr(self.model, "wav2vec2") and hasattr(self.model.wav2vec2, "_get_feat_extract_output_lengths"):
            return int(self.model.wav2vec2._get_feat_extract_output_lengths(input_length))

        if hasattr(self.model, "hubert") and hasattr(self.model.hubert, "_get_feat_extract_output_lengths"):
            return int(self.model.hubert._get_feat_extract_output_lengths(input_length))

        raise AttributeError("AudioFeatureExtractor could not infer feature length from the loaded model.")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self.processor(
            waveform,
            sampling_rate=self.sr,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.sr * self.win_max_length,
        )["input_values"]

        waveform = waveform.to(self.device)
        with torch.no_grad():
            features = self.model.extract_features(waveform)

        return features.detach().cpu().squeeze()
