"""Self-contained, device-agnostic vendoring of LIIF (Chen et al., CVPR 2021).

Adapted from the official repo https://github.com/yinboc/liif (models/{liif,rdn,mlp}.py, utils.make_coord)
with the registry made local and all `.cuda()` calls replaced by device-agnostic ops so the model can
run on CPU/MPS under our no-leak evaluation protocol.
"""
import copy

import torch

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]
    model = models[model_spec["name"]](**model_args)
    if load_sd:
        model.load_state_dict(model_spec["sd"])
    return model


def make_coord(shape, ranges=None, flatten=True, device=None):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    if device is not None:
        ret = ret.to(device)
    return ret


from . import rdn  # noqa: E402,F401  (populate registry)
from . import mlp  # noqa: E402,F401
from . import liif  # noqa: E402,F401
