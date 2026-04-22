import torch
import torch.nn as nn

from transformers import ViTForImageClassification

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.core.types import ModelOutput
from chimera_ml.models.base import BaseModel


class DPAL(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        queries = self.query(x1)
        keys = self.key(x2)
        values = self.value(x3)
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class GAL(nn.Module):
    def __init__(self, input_dim_F1: int, input_dim_F2: int, gated_dim: int) -> None:
        super().__init__()
        
        self.WF1 = nn.Parameter(torch.Tensor(input_dim_F1, gated_dim))
        self.WF2 = nn.Parameter(torch.Tensor(input_dim_F2, gated_dim))

        nn.init.xavier_uniform_(self.WF1)
        nn.init.xavier_uniform_(self.WF2)

        dim_size_f = input_dim_F1 + input_dim_F2

        self.WF = nn.Parameter(torch.Tensor(dim_size_f, gated_dim))
        nn.init.xavier_uniform_(self.WF)
        
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        h_f1 = torch.tanh(torch.matmul(f1, self.WF1))
        h_f2 = torch.tanh(torch.matmul(f2, self.WF2))
        z_f = torch.softmax(torch.matmul(torch.cat([f1, f2], dim=1), self.WF), dim=1)
        h_f = z_f * h_f1 + (1 - z_f) * h_f2
        return h_f


class StatisticalPoolingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_x = torch.mean(x, dim=1)
        std_x = torch.std(x, dim=1)
        stat_x = torch.cat((mean_x, std_x), dim=1)
        return stat_x


class ModelV1(nn.Module):
    def __init__(self, input_dim: int, drop: float, gated_dim: int, n_cl: int) -> None:
        super().__init__()

        self.fcl_x1 = nn.Linear(input_dim, input_dim)
        self.drop_fcl_x1 = nn.Dropout(p=drop)
        self.fcl_x2 = nn.Linear(input_dim, input_dim)
        self.drop_fcl_x2 = nn.Dropout(p=drop)
        self.stat_pool = StatisticalPoolingLayer()
        self.gal = GAL(input_dim*2, input_dim*2, gated_dim)
        self.fcl = nn.Linear(gated_dim, gated_dim)
        self.drop_fcl = nn.Dropout(p=drop)
        self.classifier = nn.Linear(gated_dim, n_cl)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.drop_fcl_x1(self.fcl_x1(x))
        x2 = self.drop_fcl_x2(self.fcl_x2(x))
        x1 = self.stat_pool(x1)
        x2 = self.stat_pool(x2)
        gx = self.gal(x1, x2)
        gx = self.drop_fcl(self.fcl(gx))
        return gx
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.features(x)
        out = self.classifier(gx)
        return {'gen': out[:, 0:-1], 'age': out[:, -1]}


class ModelV2(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, drop: float, n_cl: int) -> None:
        super().__init__()

        self.fcl_x = nn.Linear(input_dim, input_dim)
        self.drop_fcl_x = nn.Dropout(p=drop)
        self.dpal = DPAL(input_dim)
        self.stat_pool = StatisticalPoolingLayer()
        self.fcl = nn.Linear(input_dim*2, hid_dim)
        self.drop_fcl = nn.Dropout(p=drop)
        self.classifier = nn.Linear(hid_dim, n_cl)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.drop_fcl_x(self.fcl_x(x))
        att_x1 = self.dpal(x1, x1, x1)
        att_x1 = self.stat_pool(att_x1)
        return self.drop_fcl(self.fcl(att_x1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.drop_fcl_x(self.fcl_x(x))
        att_x1 = self.dpal(x1, x1, x1)
        att_x1 = self.stat_pool(att_x1)
        stat_att_x1 = self.drop_fcl(self.fcl(att_x1))
        out = self.classifier(stat_att_x1)
        return {'gen': out[:, 0:-1], 'age': out[:, -1]}
    
    
class AGenderImageVITGSAModel(BaseModel):
    def __init__(self, input_dim: int = 768, gated_dim: int = 128, 
                 drop: float = 0.0, n_cl: int = 3) -> None:
        super().__init__()

        self.vit_model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        self.model_final = ModelV1(input_dim=input_dim, gated_dim=gated_dim, drop=drop, n_cl=n_cl)

    def forward(self, batch: Batch) -> ModelOutput:
        vit_output = self.vit_model(batch.inputs["image"], output_hidden_states=True)
        features_vit = torch.stack(vit_output.hidden_states).squeeze()[:, 0, :].unsqueeze(0)
        outputs = self.model_final(features_vit)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)


class AGenderImageVITDPALModel(BaseModel):
    def __init__(self, input_dim: int = 768, gated_dim: int = 128, 
                 drop: float = 0.0, n_cl: int = 3) -> None:
        super().__init__()
        self.vit_model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        self.model_final = ModelV2(input_dim=input_dim, hid_dim=gated_dim, drop=drop, n_cl=n_cl)
        
    def early_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit_model.vit.embeddings(x)

    def intermediate_features(self, x: torch.Tensor) -> torch.Tensor:
        vit_output = self.vit_model(x, output_hidden_states=True)
        features_vit = torch.stack(vit_output.hidden_states)[:, :, 0, :]
        return features_vit.permute(1, 0, 2)

    def late_features(self, x: torch.Tensor) -> torch.Tensor:
        vit_output = self.vit_model(x, output_hidden_states=True)
        features_vit = torch.stack(vit_output.hidden_states)[:, :, 0, :]
        features_vit = features_vit.permute(1, 0, 2)
        return self.model_final.get_features(features_vit)

    def forward(self, batch: Batch) -> ModelOutput:
        vit_output = self.vit_model(batch.inputs["image"], output_hidden_states=True)
        features_vit = torch.stack(vit_output.hidden_states)[:, :, 0, :]
        features_vit = features_vit.permute(1, 0, 2)
        
        outputs = self.model_final(features_vit)
        return ModelOutput(preds=multitask_dict_to_tensor(outputs), aux=outputs)    


@MODELS.register("agender_image_vit_gsa_model")
def agender_image_vit_gsa_model(**params):
    return AGenderImageVITGSAModel(**params)


@MODELS.register("agender_image_vit_dpal_model")
def agender_image_vit_dpal_model(**params):
    return AGenderImageVITDPALModel(**params)