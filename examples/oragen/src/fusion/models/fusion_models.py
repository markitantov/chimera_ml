import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


from common.utils import FeaturesType, multitask_dict_to_tensor
from common.models import (
    MultiHeadAttention, 
    StatPoolLayer,
    AGenderClassificationHead,
    AGenderClassificationHeadV2,
    MaskAGenderClassificationHead,
    Permute,
)


def _features_type(value) -> FeaturesType:
    if isinstance(value, FeaturesType):
        return value
    if isinstance(value, str):
        key = value.strip().upper()
        if key in FeaturesType.__members__:
            return FeaturesType[key]
    return FeaturesType(int(value))


class MTCMAModelV1(nn.Module):
    def __init__(self, dim_q: int, dim_v: int, num_heads: int = 4) -> None:
        super().__init__()
        self.dim_q = dim_q
        self.dim_v = dim_v
        
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_k = nn.LayerNorm(dim_v)
        self.norm_v = nn.LayerNorm(dim_v)
        
        self.fc_q = nn.Linear(dim_q, 256)
        self.fc_k = nn.Linear(dim_v, 256)
        self.fc_v = nn.Linear(dim_v, 256)

        self.self_attention = MultiHeadAttention(input_dim=256, num_heads=4, dropout=.2)
        
        self.mlp = nn.Sequential(
          nn.Linear(256, 128),
          nn.GELU(),
          nn.Dropout(.2),
          nn.LayerNorm(128),
          nn.Linear(128, 256),
          nn.GELU(),
          nn.Dropout(.2),
        )

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = self.fc_q(self.norm_q(queries))
        keys = self.fc_k(self.norm_k(keys))
        values = self.fc_v(self.norm_v(values))
        
        x = self.self_attention(queries=queries, keys=keys, values=values, mask=None)
        x = x + self.mlp(x)
        return x
    

class MTCMAModelV2(nn.Module):
    def __init__(self, dim_q: int, dim_v: int, num_heads: int = 4) -> None:
        super().__init__()
        self.dim_q = dim_q
        self.dim_v = dim_v
        
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_k = nn.LayerNorm(dim_v)
        self.norm_v = nn.LayerNorm(dim_v)
        
        self.self_attention = MultiHeadAttention(input_dim=256, num_heads=4, dropout=.2)
        
        self.mlp = nn.Sequential(
          nn.Linear(256, 128),
          nn.GELU(),
          nn.Dropout(.2),
          nn.LayerNorm(128),
          nn.Linear(128, 256),
          nn.GELU(),
          nn.Dropout(.2),
        )

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = self.norm_q(queries)
        keys = self.norm_k(keys)
        values = self.norm_v(values)
        
        x = self.self_attention(queries=queries, keys=keys, values=values, mask=None)
        x = x + self.mlp(x)
        return x


class AVModelV1(BaseModel):
    def __init__(self, features_type: FeaturesType | int | str) -> None:
        super().__init__()
        
        self.features_type = _features_type(features_type)
        
        # Image feature branch
        if self.features_type == FeaturesType.EARLY:
            self.downsampling_a = nn.Linear(199, 4)
            self.downsampling_v = nn.Identity()
            
            self.mtcma_av = MTCMAModelV1(dim_q=512, dim_v=151296)
            self.mtcma_va = MTCMAModelV1(dim_q=151296, dim_v=512)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            self.downsampling_a = nn.Linear(199, 4)
            self.downsampling_v = nn.Identity()
            
            self.mtcma_av = MTCMAModelV1(dim_q=1024, dim_v=9984)
            self.mtcma_va = MTCMAModelV1(dim_q=9984, dim_v=1024)
        else:
            self.downsampling_a = nn.Identity()
            self.downsampling_v = nn.Linear(4, 1)
            
            self.mtcma_av = MTCMAModelV1(dim_q=256, dim_v=128, num_heads=1)
            self.mtcma_va = MTCMAModelV1(dim_q=128, dim_v=256, num_heads=1)
        
        self.stp = StatPoolLayer(dim=1)
        
        self.fc = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=.3)
        
        self.cl_head = AGenderClassificationHead(input_size=256, output_size=3)

    def forward(self, batch: Batch) -> ModelOutput:
        a, v = batch.inputs["audio"], batch.inputs["image"]
        bs = a.shape[0]
        
        if self.features_type == FeaturesType.EARLY:    
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
            a = a.permute(0, 2, 1)
        else:
            v = v.permute(0, 2, 1)
            a = a.unsqueeze(dim=1).permute(0, 2, 1)
        
        a = self.downsampling_a(a)
        v = self.downsampling_v(v)
        
        a = a.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        
        av = self.mtcma_av(queries=a, keys=v, values=v)
        va = self.mtcma_va(queries=v, keys=a, values=a)
        
        x = torch.cat((av, va), dim=1)
        x = self.stp(x)
        
        x = self.dp(self.relu(self.fc(x)))        
        outputs = self.cl_head(x)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)


class AVModelV2(BaseModel):
    def __init__(self, features_type: FeaturesType | int | str) -> None:
        super().__init__()
        self.features_type = _features_type(features_type)
        # Image feature branch
        if self.features_type == FeaturesType.EARLY:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(512, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(197 * 768, 256),
            )
        elif self.features_type == FeaturesType.INTERMEDIATE:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(1024, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(13 * 768, 256),
            )
        else:
            self.downsampling_a = nn.Sequential(
                nn.Identity(),
            )

            self.downsampling_v = nn.Sequential(
                nn.Linear(4, 1),
                Permute((0, 2, 1)),
                nn.Linear(128, 256),
            )
            
        self.mtcma_av = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        self.mtcma_va = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        
        self.stp = StatPoolLayer(dim=1)
        
        out_features = 256 if self.features_type == FeaturesType.LATE else 512
        
        self.gender_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )

        self.age_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )
        
        self.cl_head = AGenderClassificationHeadV2(256)

    def forward(self, batch: Batch) -> ModelOutput:
        a, v = batch.inputs["audio"], batch.inputs["image"]
        bs = a.shape[0]        
        if self.features_type == FeaturesType.EARLY:    
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
            a = a.permute(0, 2, 1)
        else:
            v = v.permute(0, 2, 1)
            a = a.unsqueeze(dim=1)

        a = self.downsampling_a(a)
        v = self.downsampling_v(v)

        av = self.mtcma_av(queries=a, keys=v, values=v)
        va = self.mtcma_va(queries=v, keys=a, values=a)

        av = a + av
        va = v + va
        if self.features_type == FeaturesType.LATE:
            av = av.squeeze()
            va = va.squeeze()
        else:
            av = self.stp(av)
            va = self.stp(va)

        x_gender = self.gender_branch(av)
        x_age = self.age_branch(va)

        outputs = self.cl_head(x_gender, x_age)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)


class AVModelV3(BaseModel):
    def __init__(self, features_type: FeaturesType | int | str) -> None:
        super().__init__()
        self.features_type = _features_type(features_type)
        # Image feature branch
        if self.features_type == FeaturesType.EARLY:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(512, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(197 * 768, 256),
            )
        elif self.features_type == FeaturesType.INTERMEDIATE:
            self.downsampling_a = nn.Sequential(
                nn.Linear(199, 4),
                Permute((0, 2, 1)),
                nn.Linear(1024, 256),
            )

            self.downsampling_v = nn.Sequential(
                Permute((0, 2, 1)),
                nn.Linear(13 * 768, 256),
            )
        else:
            self.downsampling_a = nn.Sequential(
                nn.Identity(),
            )

            self.downsampling_v = nn.Sequential(
                nn.Linear(4, 1),
                Permute((0, 2, 1)),
                nn.Linear(128, 256),
            )
            
        self.mtcma_av = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        self.mtcma_va = MTCMAModelV2(dim_q=256, dim_v=256, num_heads=1 if self.features_type == FeaturesType.LATE else 4)
        
        self.stp = StatPoolLayer(dim=1)
        
        out_features = 256 if self.features_type == FeaturesType.LATE else 512
        
        self.gender_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )

        self.age_branch = nn.Sequential(
            nn.Linear(out_features, 256),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )
        
        self.cl_head = AGenderClassificationHead(256, output_size=3)
        
    def get_agender_features(self, batch: Batch) -> ModelOutput:
        a, v = batch.inputs["audio"], batch.inputs["image"]
        bs = a.shape[0]        
        if self.features_type == FeaturesType.EARLY:    
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
        elif self.features_type == FeaturesType.INTERMEDIATE:
            v = v.reshape(bs, 4, -1).permute(0, 2, 1)
            a = a.permute(0, 2, 1)
        else:
            v = v.permute(0, 2, 1)
            a = a.unsqueeze(dim=1)

        a = self.downsampling_a(a)
        v = self.downsampling_v(v)

        av = self.mtcma_av(queries=a, keys=v, values=v) # -> va
        va = self.mtcma_va(queries=v, keys=a, values=a) # -> av

        av = a + av # -> a + va
        va = v + va # -> v + av
        if self.features_type == FeaturesType.LATE:
            av = av.squeeze()
            va = va.squeeze()
        else:
            av = self.stp(av)
            va = self.stp(va)

        return self.gender_branch(av), self.age_branch(va)

    def forward(self, batch: Batch) -> ModelOutput:
        x_gender, x_age = self.get_agender_features(batch)
        outputs = self.cl_head(x_gender + x_age)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)


class MaskAgenderAVModelV1(BaseModel):
    def __init__(self, features_type: FeaturesType | int | str, checkpoint_path: str = None) -> None:
        super().__init__()
        self.av_model = AVModelV3(features_type=features_type)
            
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.av_model.load_state_dict(checkpoint['model_state_dict'])
            
        for param in self.av_model.parameters():
            param.requires_grad = False
        
        self.cl_maskhead = nn.Linear(256, 6)

    def forward(self, batch: Batch) -> ModelOutput:
        x_gender, x_age = self.av_model.get_agender_features(batch)
        agender_res = self.av_model.cl_head(x_gender + x_age)
        mask_res = {'mask': self.cl_maskhead(x_gender + x_age)}
        outputs = {**agender_res, **mask_res}
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)
    
    
class MaskAgenderAVModelV2(BaseModel):
    def __init__(self, features_type: FeaturesType | int | str, checkpoint_path: str = None) -> None:
        super().__init__()
        
        self.av_model = AVModelV3(features_type=features_type)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.av_model.load_state_dict(checkpoint['model_state_dict'])
            
        for param in self.av_model.parameters():
            param.requires_grad = False
        
        self.av_model.cl_head = MaskAGenderClassificationHead(256, output_size=3)

    def forward(self, batch: Batch) -> ModelOutput:
        x_gender, x_age = self.av_model.get_agender_features(batch)
        outputs = self.av_model.cl_head(x_gender + x_age)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)
              

class MaskAgenderAVModelV3(BaseModel):
    def __init__(self, features_type: FeaturesType | int | str, checkpoint_path: str = None) -> None:
        super().__init__()
        
        self.av_model = AVModelV3(features_type=features_type)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.av_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.av_model.cl_head = MaskAGenderClassificationHead(256, output_size=3)

    def forward(self, batch: Batch) -> ModelOutput:
        x_gender, x_age = self.av_model.get_agender_features(batch)
        outputs = self.av_model.cl_head(x_gender + x_age)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)   


'''
EARLY
A: torch.Size([512, 199])
V: torch.Size([4, 197, 768])

INTERMEDIATE
A: torch.Size([199, 1024])
V: torch.Size([4, 13, 768])

LATE
A: torch.Size([256])
V: torch.Size([4, 128])
'''

MaskAgenderAVModelV1
@MODELS.register("av_model_v1")
def av_model_v1(**params):
    return AVModelV1(**params) 


@MODELS.register("av_model_v2")
def av_model_v2(**params):
    return AVModelV2(**params) 


@MODELS.register("av_model_v3")
def av_model_v3(**params):
    return AVModelV3(**params) 

@MODELS.register("mask_agender_av_model_v1")
def mask_agender_av_model_v1(**params):
    return MaskAgenderAVModelV1(**params) 


@MODELS.register("mask_agender_av_model_v2")
def mask_agender_av_model_v2(**params):
    return MaskAgenderAVModelV2(**params) 


@MODELS.register("mask_agender_av_model_v3")
def mask_agender_av_model_v3(**params):
    return MaskAgenderAVModelV3(**params) 


if __name__ == "__main__":
    device = torch.device('cpu')
    
    features = [
        {
            'a': torch.zeros((10, 512, 199)).to(device),
            'v': torch.zeros((10, 4, 197, 768)).to(device),
            'features_type': FeaturesType.EARLY,
        },
        {
            'a': torch.zeros((10, 199, 1024)).to(device),
            'v': torch.zeros((10, 4, 13, 768)).to(device),
            'features_type': FeaturesType.INTERMEDIATE,
        },
        {
            'a': torch.rand((10, 256)).to(device),
            'v': torch.rand((10, 4, 128)).to(device),
            'features_type': FeaturesType.LATE,
        },
    ]
    
    for f in features:
        model = AVModelV3(features_type=f['features_type'])
        print(model([f['a'], f['v']]))