"""Differentiable SD3 adapter in the paper's noise->data time convention.

Diffusers predicts d z / d sigma with sigma decreasing from 1 to 0.
Here t=1-sigma and v=-prediction. We use an explicit uniform t grid;
this is not an assertion about the authors' undisclosed scheduler details.
"""
import torch


class SD3Prior:
    def __init__(self, model_id, prompt, device="cuda", guidance=2.0, *, pipeline=None):
        if pipeline is None:
            try:
                from diffusers import StableDiffusion3Pipeline
            except ImportError as e:
                raise ImportError('Install the SD3 dependencies: pip install -e ".[sd3]"') from e
            # FP32 input/energy calculations, reduced precision network on CUDA.
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
            pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe = pipeline.to(device)
        self.device, self.guidance = device, guidance
        for name in ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(self.pipe, name, None)
            if module is not None:
                module.eval().requires_grad_(False)
        with torch.no_grad():
            positive, negative, pooled, negative_pooled = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt="",
                do_classifier_free_guidance=True, device=device)
        self.embeds = torch.cat((negative, positive)).detach()
        self.pooled = torch.cat((negative_pooled, pooled)).detach()

    def velocity(self, z, t):
        if z.shape[0] != 1:
            raise ValueError("SD3Prior currently supports one prompt/sample at a time")
        dtype = self.pipe.transformer.dtype
        latent = torch.cat((z, z)).to(dtype)
        timestep = torch.cat((1-t, 1-t)).to(dtype) * self.pipe.scheduler.config.num_train_timesteps
        prediction = self.pipe.transformer(
            hidden_states=latent, timestep=timestep,
            encoder_hidden_states=self.embeds.to(dtype),
            pooled_projections=self.pooled.to(dtype), return_dict=False)[0].float()
        uncond, cond = prediction.chunk(2)
        return -(uncond + self.guidance * (cond-uncond))

    def decode(self, z):
        vae = self.pipe.vae
        scaled = z.to(vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
        # No clamp/no no_grad: gradients must reach the latent endpoint.
        return vae.decode(scaled, return_dict=False)[0].float()

    def noise(self, height, width):
        scale = self.pipe.vae_scale_factor
        patch = self.pipe.transformer.config.patch_size
        if height % (scale*patch) or width % (scale*patch):
            raise ValueError(f"height and width must be divisible by {scale*patch}")
        return torch.randn(1, self.pipe.transformer.config.in_channels,
                           height//scale, width//scale, device=self.device, dtype=torch.float32)
