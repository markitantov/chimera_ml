import random

import torch


from common.utils import DatasetType

class ModalityDropAugmentation:
    def __init__(self, audio_drop_prob: float = 0.3, video_drop_prob: float = 0.3) -> None:
        self.audio_drop_prob = float(audio_drop_prob)
        self.video_drop_prob = float(video_drop_prob)

    def __call__(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        audio, video = inputs
        if random.random() < self.audio_drop_prob:
            audio = torch.zeros_like(audio)

        if random.random() < self.video_drop_prob:
            video = torch.zeros_like(video)
        
        return audio, video
    

class MultiAugment(torch.nn.Module):
    def __init__(self, dataset_type) -> None:
        super(MultiAugment, self).__init__()
        self.dataset_type = dataset_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_data, v_data = x
        if self.dataset_type == DatasetType.BOTH:
            return a_data, v_data
        
        if self.dataset_type == DatasetType.AUDIO:
            v_data = torch.zeros(v_data.shape)
        else:
            a_data = torch.zeros(a_data.shape)
        
        return a_data, v_data
