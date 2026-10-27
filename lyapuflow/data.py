from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset


def load_image(path, size=None, channels=3):
    with Image.open(path) as im:
        im = im.convert("L" if channels == 1 else "RGB")
        if size is not None:
            im = ImageOps.fit(im, (size, size), method=Image.Resampling.BICUBIC)
        a = np.array(im, dtype=np.float32, copy=True) / 127.5 - 1
    if a.ndim == 2:
        a = a[..., None]
    return torch.from_numpy(a).permute(2, 0, 1)


def load_audio(path, sample_rate=16000, length=None, channels=1):
    sr, a = wavfile.read(path)
    if np.issubdtype(a.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(a.dtype).max + 1) / 2
        a = (a.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(a.dtype, np.signedinteger):
        a = a.astype(np.float32) / (-float(np.iinfo(a.dtype).min))
    else:
        a = a.astype(np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if channels == 1:
        a = a.mean(axis=1, keepdims=True)
    elif a.shape[1] != channels:
        raise ValueError(f"expected {channels} audio channels, got {a.shape[1]}")
    if sr != sample_rate:
        gcd = math.gcd(sr, sample_rate)
        a = resample_poly(a, sample_rate // gcd, sr // gcd, axis=0)
    x = torch.from_numpy(np.array(a.T, dtype=np.float32, copy=True))
    if length is not None:
        x = F.pad(x[..., :length], (0, max(0, length-x.shape[-1])))
    return x


def save_sample(path, x, domain, sample_rate=16000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu()
    if domain == "image":
        a = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(a[..., 0] if a.shape[-1] == 1 else a).save(path)
    else:
        a = (x.clamp(-1, 1).T.numpy() * 32767).round().astype(np.int16)
        wavfile.write(path, sample_rate, a[:, 0] if a.shape[1] == 1 else a)


class MediaDataset(Dataset):
    def __init__(self, root, domain, size=64, length=16384, sample_rate=16000, channels=None):
        self.domain, self.size, self.length, self.sample_rate = domain, size, length, sample_rate
        self.channels = channels if channels is not None else (3 if domain == "image" else 1)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"} if domain == "image" else {".wav"}
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in extensions)
        if not self.paths:
            raise ValueError(f"no {domain} files found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.domain == "image":
            return load_image(self.paths[i], self.size, self.channels)
        # Random training crops; inference uses the deterministic loader.
        x = load_audio(self.paths[i], self.sample_rate, channels=self.channels)
        start = torch.randint(max(1, x.shape[-1] - self.length + 1), ()).item()
        x = x[..., start:start+self.length]
        return F.pad(x, (0, self.length - x.shape[-1]))
