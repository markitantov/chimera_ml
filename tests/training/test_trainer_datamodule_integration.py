import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.data.datamodule import DataModule
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.prf_metric import PRFMetric
from chimera_ml.training.config import TrainConfig
from chimera_ml.training.trainer import Trainer


class _VariableLengthBinaryDataset(Dataset):
    def __init__(self, n_samples: int) -> None:
        self._n_samples = n_samples

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> dict[str, object]:
        label = idx % 2
        seq_len = 2 + (idx % 4)  # variable length: 2..5
        sign = 1.0 if label == 1 else -1.0

        inputs: dict[str, torch.Tensor] = {
            "text": torch.full((seq_len, 1), fill_value=sign, dtype=torch.float32),
        }

        # "audio" is intentionally missing for part of samples to exercise modality masks.
        if idx % 3 != 0:
            inputs["audio"] = torch.full((seq_len, 1), fill_value=0.5 * sign, dtype=torch.float32)

        return {
            "inputs": inputs,
            "target": torch.tensor(label, dtype=torch.long),
            "meta": {"sample_id": idx},
        }


class _MaskedSequenceClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._head = torch.nn.Linear(1, 2)

    def forward(self, batch: Batch) -> ModelOutput:
        text = batch.inputs["text"]  # [B, T, 1]
        masks = batch.get_masks()
        assert isinstance(masks, dict)

        sequence_mask = masks["sequence_mask"].to(text.device)
        sequence_mask = sequence_mask[:, : text.shape[1]]
        weights = sequence_mask.unsqueeze(-1).to(text.dtype)

        pooled = (text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self._head(pooled)
        return ModelOutput(preds=logits)


class _CrossEntropyLoss(BaseLoss):
    def __call__(self, output: ModelOutput, batch: Batch) -> torch.Tensor:
        assert batch.targets is not None
        return F.cross_entropy(output.preds, batch.targets.long())


def test_integration_datamodule_train_and_validate_with_micro_f1():
    torch.manual_seed(0)

    train_dataset = _VariableLengthBinaryDataset(n_samples=96)
    val_dataset = _VariableLengthBinaryDataset(n_samples=48)
    datamodule = DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        shuffle_train=False,
        num_workers=0,
        pin_memory=False,
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    assert train_loader is not None
    assert val_loader is not None

    first_batch = next(iter(train_loader))
    assert first_batch.inputs["text"].shape[1] > 2  # confirms variable-length padding
    masks = first_batch.get_masks()
    assert isinstance(masks, dict)
    assert "sequence_mask" in masks
    assert "audio_mask" in masks
    assert masks["audio_mask"].sum().item() < float(first_batch.inputs["text"].shape[0])

    model = _MaskedSequenceClassifier()
    trainer = Trainer(
        model=model,
        loss_fn=_CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.25),
        metrics=[PRFMetric(average="micro")],
        config=TrainConfig(
            epochs=6,
            mixed_precision=False,
            use_scheduler=False,
            device="cpu",
            log_every_steps=1000,
        ),
        mlflow_logger=None,
        logger=None,
        callbacks=[],
        scheduler=None,
    )

    trainer.fit(train_loader, val_loaders={"val": val_loader})
    eval_logs = trainer.evaluate({"val": val_loader})

    assert "val/micro_f1" in eval_logs
    assert eval_logs["val/micro_f1"] >= 0.95
