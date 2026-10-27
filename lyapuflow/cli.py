"""Train flow priors, solve inverse problems, or run self-contained demos."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .controller import ControllerConfig, LyapuFlow
from .data import MediaDataset, load_image, load_audio, save_sample
from .models import ConvFlow, flow_matching_loss
from .metrics import reconstruction_metrics
from .operators import (Measurement, Identity, Mask, ImageSuperResolution, ImageBlur,
                        AudioLowPass, AudioReverb)


def make_operator(args, reference):
    task = args.task
    if task == "denoise":
        return Identity().to(reference.device)
    if task == "inpaint":
        mask = torch.ones_like(reference)
        if args.domain == "image":
            h, w = reference.shape[-2:]
            mask[..., h//4:3*h//4, w//4:3*w//4] = 0
        else:
            n = reference.shape[-1]
            mask[..., n//3:2*n//3] = 0
        return Mask(mask)
    if args.domain == "image":
        if task == "superres":
            return ImageSuperResolution(args.factor).to(reference.device)
        if task in ("gaussian", "motion"):
            return ImageBlur(task, args.kernel_size, args.blur_sigma, args.angle).to(reference.device)
    else:
        if task in ("bandwidth", "superres"):
            factor = args.factor if task == "superres" else 1
            cutoff = args.cutoff or args.sample_rate / (2 * max(2, factor)) * 0.9
            return AudioLowPass(args.sample_rate, cutoff, factor).to(reference.device)
        if task == "dereverb":
            if args.ir:
                ir = load_audio(args.ir, args.sample_rate)[0]
            else:
                # Explicit synthetic known-room demo impulse response.
                ir = torch.zeros(257)
                ir[0], ir[80], ir[160], ir[256] = 1, 0.45, 0.25, 0.1
            return AudioReverb(ir).to(reference.device)
    raise ValueError(f"task {task} is not supported for domain {args.domain}")


def train(args):
    torch.manual_seed(args.seed)
    ds = MediaDataset(args.data, args.domain, args.size, args.length, args.sample_rate)
    model = ConvFlow(args.domain, ds.channels, args.width, args.depth).to(args.device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    checkpoint = Path(args.output)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    while step < args.iterations:
        for batch in loader:
            model.train()
            loss = flow_matching_loss(model, batch.to(args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 100 == 0 or step == args.iterations:
                print(json.dumps({"step": step, "loss": loss.item()}), flush=True)
            if step >= args.iterations:
                break
    torch.save({"model": model.state_dict(), "spec": model.spec,
                "data": {"size": args.size, "length": args.length, "sample_rate": args.sample_rate},
                "steps": step}, checkpoint)


def synthetic(domain, size, length, sample_rate, device):
    if domain == "image":
        a = torch.linspace(-1, 1, size, device=device)
        yy, xx = torch.meshgrid(a, a, indexing="ij")
        return torch.stack((torch.sin(3*xx), torch.cos(4*yy), torch.sin(2*(xx+yy))))[None] * 0.7
    t = torch.arange(length, device=device) / sample_rate
    return (0.4*torch.sin(2*torch.pi*220*t) + 0.2*torch.sin(2*torch.pi*660*t))[None, None]


def reconstruct(args, demo=False, sd3=False):
    torch.manual_seed(args.seed)
    if sd3:
        from .sd3 import SD3Prior
        prior = SD3Prior(args.model_id, args.prompt, args.device, args.guidance)
        velocity, decoder = prior.velocity, prior.decode
        domain, channels = "image", 3
        metadata = {"size": args.size}
    elif not demo:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["spec"]["domain"] != args.domain:
            raise ValueError("checkpoint domain differs from requested domain")
        model = ConvFlow(**saved["spec"]).to(args.device).eval()
        model.load_state_dict(saved["model"])
        model.requires_grad_(False)
        velocity, decoder = model, None
        domain, channels = args.domain, saved["spec"]["channels"]
        metadata = saved["data"]
        if domain == "audio" and args.sample_rate != metadata["sample_rate"]:
            raise ValueError("sample rate must match the training checkpoint")
    else:
        # An actual analytic Gaussian flow, independent of the reference:
        # x_t = sigma(t) x_0, sigma(t)=1-0.5t. No learned/demo oracle.
        velocity = lambda x, t: -0.5*x / (1-0.5*t).reshape(-1, *([1]*(x.ndim-1)))
        decoder = None
        domain, channels, metadata = args.domain, (3 if args.domain == "image" else 1), {}
    if demo:
        reference = synthetic(domain, args.size, args.length, args.sample_rate, args.device)
    elif domain == "image":
        reference = load_image(args.input, metadata.get("size", args.size), channels)[None].to(args.device)
    else:
        reference = load_audio(args.input, args.sample_rate, metadata["length"], channels)[None].to(args.device)
    operator = make_operator(args, reference)
    if args.observation:
        obs = torch.load(args.observation, map_location=args.device, weights_only=True)
        if not isinstance(obs, torch.Tensor):
            raise ValueError("observation file must contain a tensor")
    else:
        obs = operator(reference)
        obs = obs + args.noise * torch.randn_like(obs)
    residual = Measurement(operator, obs).to(args.device)
    if sd3:
        initial = prior.noise(reference.shape[-2], reference.shape[-1])
    else:
        initial = torch.randn_like(reference)
    switch = args.switch_step if args.switch_step is not None else (args.steps if sd3 else 0)
    config = ControllerConfig(args.steps, args.alpha, args.trust_region, switch)
    guided = LyapuFlow(velocity, residual, config, decoder).sample(initial)
    baseline = LyapuFlow(velocity, residual, ControllerConfig(args.steps, 0, 0, args.steps), decoder).sample(initial, record=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ext = ".png" if domain == "image" else ".wav"
    for name, x in (("reference", reference), ("reconstruction", guided.sample), ("unguided", baseline.sample)):
        save_sample(out / (name+ext), x[0], domain, args.sample_rate)
    torch.save(obs.detach().cpu(), out / "observation.pt")
    torch.save(guided.sample.cpu(), out / "reconstruction.pt")
    torch.save(guided.history, out / "history.pt")
    report = {"config": asdict(config), "domain": domain, "task": args.task, "seed": args.seed,
              "demo": demo, "measurement_noise": args.noise if not args.observation else None}
    for name, x in (("guided", guided.sample), ("unguided", baseline.sample)):
        report[name] = reconstruction_metrics(x, reference, domain)
        with torch.no_grad():
            r = residual(x).flatten(1)
            report[name]["measurement_rmse"] = r.square().mean().sqrt().item()
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--domain", choices=["image", "audio"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="checkpoint .pt path")
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    for command in ("demo", "reconstruct", "sd3"):
        q = sub.add_parser(command)
        if command == "sd3":
            q.set_defaults(domain="image")
            q.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-medium")
            q.add_argument("--prompt", default="a high quality photo")
            q.add_argument("--guidance", type=float, default=2.0)
        else:
            q.add_argument("--domain", choices=["image", "audio"], required=True)
        if command != "demo":
            q.add_argument("--input", required=True, help="clean reference used to simulate observations and evaluate")
        if command == "reconstruct":
            q.add_argument("--checkpoint", required=True)
        q.add_argument("--observation", help="optional actual measurement .pt tensor [B,C,...]")
        q.add_argument("--output", required=True, help="output directory")
        q.add_argument("--task", choices=["denoise", "inpaint", "superres", "gaussian", "motion", "bandwidth", "dereverb"], default="inpaint")
        q.add_argument("--steps", type=int, default=28 if command == "sd3" else 100)
        q.add_argument("--alpha", type=float, default=5.0)
        q.add_argument("--trust-region", type=float, default=1.0)
        q.add_argument("--switch-step", type=int)
        q.add_argument("--noise", type=float, default=0.03)
        q.add_argument("--factor", type=int, default=4)
        q.add_argument("--kernel-size", type=int, default=61)
        q.add_argument("--blur-sigma", type=float, default=3.0)
        q.add_argument("--angle", type=float, default=0.0)
        q.add_argument("--cutoff", type=float)
        q.add_argument("--ir", help="known room impulse response WAV")
    for p in sub.choices.values():
        p.add_argument("--size", type=int, default=768 if p.prog.endswith(" sd3") else 64)
        p.add_argument("--length", type=int, default=16384)
        p.add_argument("--sample-rate", type=int, default=16000)
        p.add_argument("--device", default="cpu")
        p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "train":
        if args.iterations < 1 or args.batch_size < 1:
            parser.error("iterations and batch-size must be positive")
        train(args)
    else:
        reconstruct(args, demo=args.command == "demo", sd3=args.command == "sd3")


if __name__ == "__main__":
    main()
