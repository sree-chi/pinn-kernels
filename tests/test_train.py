import pytest

from pinn_kernels.train import Config, train


@pytest.mark.parametrize("physics,backend", [("autograd", "torch"), ("fd", "torch"), ("fd", "llvm")])
def test_forward_training_reduces_loss(physics, backend):
    cfg = Config(problem="forward", physics=physics, backend=backend, epochs=60, n_colloc=500,
                 grid=(16, 20), log_every=59)
    res, _ = train(cfg, verbose=False)
    first, last = res.history[0]["total"], res.history[-1]["total"]
    assert last < first


def test_llvm_and_torch_backends_train_identically():
    common = dict(problem="inverse", physics="fd", epochs=40, grid=(16, 20), log_every=39)
    a, _ = train(Config(backend="torch", **common), verbose=False)
    b, _ = train(Config(backend="llvm", **common), verbose=False)
    assert abs(a.history[-1]["total"] - b.history[-1]["total"]) < 1e-4 * a.history[-1]["total"]
    assert abs(a.alpha - b.alpha) < 1e-5
