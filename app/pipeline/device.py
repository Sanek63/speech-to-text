def detect_device() -> str:
    """
    cuda если доступна, иначе mps (Apple Silicon — не подходит для NeMo, только для ASR),
    иначе cpu.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"
