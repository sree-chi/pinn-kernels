import os

import pytest
import torch

# Without a GPU, run Triton kernels through Triton's CPU interpreter. This must
# be set before triton is first imported.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
