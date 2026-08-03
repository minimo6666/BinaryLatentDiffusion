"""Optional dependency shim; FID is not used by this experiment."""


def calculate_metrics(*args, **kwargs):
    raise RuntimeError("torch_fidelity is not installed and FID is not part of this experiment")
