from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import random
import os
import numpy as np
import torch
import typing as tp


def get_logger(filename="training", logger_name="TrainingLogger"):
    logger = getLogger(logger_name)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(asctime)s %(levelname)s %(message)s"))
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.propagate = False
    return logger


def get_device():
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Get the name of the current GPU (device 0)
        current_gpu_name = torch.cuda.get_device_name(0)

        print(f"CUDA is available with {num_gpus} GPU(s).")
        print(f"Current GPU: {current_gpu_name}")

        # Set the default device to GPU 0
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    return device


def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device,
    *args,
    **kwargs,
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
