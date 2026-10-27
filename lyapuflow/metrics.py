import torch


def reconstruction_metrics(estimate, reference, domain):
    error = (estimate-reference).flatten(1).square().mean(1)
    result = {"mse": error.mean().item()}
    if domain == "image":
        result["psnr_db"] = (10 * torch.log10(4 / error.clamp_min(1e-12))).mean().item()
    else:
        # SI-SDR with per-channel DC removal, then pooled channels/time.
        x = estimate - estimate.mean(-1, keepdim=True)
        y = reference - reference.mean(-1, keepdim=True)
        x, y = x.flatten(1), y.flatten(1)
        projection = ((x*y).sum(1) / y.square().sum(1).clamp_min(1e-12))[:, None] * y
        ratio = projection.square().sum(1) / (x-projection).square().sum(1).clamp_min(1e-12)
        result["si_sdr_db"] = (10*ratio.clamp_min(1e-12).log10()).mean().item()
    return result
