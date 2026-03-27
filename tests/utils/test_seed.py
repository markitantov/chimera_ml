import os
import random

import numpy as np
import torch

from chimera_ml.utils.seed import define_seed


def test_define_seed_sets_reproducible_state(monkeypatch):
    calls = {"manual_seed": [], "manual_seed_all": []}

    monkeypatch.setattr(torch.cuda, "manual_seed", lambda s: calls["manual_seed"].append(s))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: calls["manual_seed_all"].append(s))

    define_seed(123)

    assert os.environ["PYTHONHASHSEED"] == "123"
    assert calls["manual_seed"] == [123]
    assert calls["manual_seed_all"]
    assert all(v == 123 for v in calls["manual_seed_all"])
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    a = random.random()
    b = float(np.random.rand())
    c = float(torch.rand(1).item())

    define_seed(123)
    assert random.random() == a
    assert float(np.random.rand()) == b
    assert float(torch.rand(1).item()) == c
