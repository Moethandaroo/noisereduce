import torch
from torch.types import Number


@torch.no_grad()
def amp_to_db(x: torch.Tensor, eps: float = torch.finfo(torch.float64).eps, top_db: float = 40) -> torch.Tensor:
    """
    Convert the input tensor from amplitude to decibel scale.

    This function transforms an amplitude input `x` (e.g., audio signal or magnitude spectrum)
    into a decibel (dB) scale.
    It applies the formula:

        dB = 20 * log10(abs(x) + eps)

    where `eps` is a small constant added to avoid log of zero. The function also limits the maximum
    value of the decibel output to `top_db` decibels below the maximum value across the last axis of `x`.

    Arguments:
        x (torch.Tensor): Input tensor.
        eps (float): Small value to avoid numerical instability. (default: torch.finfo(torch.float64).eps)
        top_db (float): threshold the output at ``top_db`` below the peak (default: 40)

    Returns:
        torch.Tensor: Output tensor in decibel scale.
    """
    x_db = 20 * torch.log10(x.abs() + eps)
    return torch.max(x_db, (x_db.max(-1).values - top_db).unsqueeze(-1))


@torch.no_grad()
def temperature_sigmoid(x: torch.Tensor, x0: float, temp_coeff: float) -> torch.Tensor:
    """
    Apply a sigmoid function with temperature scaling.

    Arguments:
        x (torch.Tensor): Input tensor.
        x0 (float): Parameter that controls the threshold of the sigmoid.
        temp_coeff (float): Parameter that controls the slope of the sigmoid.

    Returns:
        torch.Tensor: Output tensor after applying the sigmoid with temperature scaling.
    """
    return torch.sigmoid((x - x0) / temp_coeff)


@torch.no_grad()
def linspace(start: Number, stop: Number, num: int = 50, endpoint: bool = True, **kwargs) -> torch.Tensor:
    """
    Generate a linearly spaced 1-D tensor.

    Arguments:
        start (Number): The starting value of the sequence.
        stop (Number): The end value of the sequence, unless `endpoint` is set to False.
                        In that case, the sequence consists of all but the last of `num + 1`
                        evenly spaced samples, so that `stop` is excluded. Note that the step
                        size changes when `endpoint` is False.
        num (int): Number of samples to generate. Default is 50. Must be non-negative.
        endpoint (bool): If True, `stop` is the last sample. Otherwise, it is not included.
                          Default is True.
        **kwargs: Additional arguments to be passed to the `linspace` function.

    Returns:
        torch.Tensor: 1-D tensor of `num` equally spaced samples from `start` to `stop`.
    """
    if endpoint:
        return torch.linspace(start, stop, num, **kwargs)
    else:
        return torch.linspace(start, stop, num + 1, **kwargs)[:-1]
