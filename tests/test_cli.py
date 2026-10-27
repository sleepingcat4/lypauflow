import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from lyapuflow.data import save_sample, load_audio


def run(*args):
    return subprocess.run([sys.executable, "-m", "lyapuflow", *map(str, args)],
                          check=True, capture_output=True, text=True)


@pytest.mark.parametrize("domain", ["audio", "image"])
def test_train_checkpoint_reconstruct(tmp_path, domain):
    data = tmp_path / "data"
    data.mkdir()
    x = torch.rand((3, 16, 16) if domain == "image" else (1, 256))*.8-.4
    path = data / ("sample.png" if domain == "image" else "sample.wav")
    save_sample(path, x, domain)
    ckpt = tmp_path / "prior.pt"
    run("train", "--domain", domain, "--data", data, "--output", ckpt,
        "--iterations", 1, "--batch-size", 1, "--width", 8, "--depth", 1,
        "--size", 16, "--length", 256)
    output = tmp_path / "output"
    run("reconstruct", "--domain", domain, "--checkpoint", ckpt,
        "--input", path, "--output", output, "--steps", 2)
    report = json.loads((output / "metrics.json").read_text())
    assert report["domain"] == domain
    assert report["guided"]["measurement_rmse"] >= 0
    assert len(torch.load(output / "history.pt", weights_only=True)) == 2


def test_unsigned_audio_loading(tmp_path):
    import numpy as np
    from scipy.io import wavfile
    path = tmp_path / "unsigned.wav"
    wavfile.write(path, 16000, np.array([0, 128, 255], dtype=np.uint8))
    x = load_audio(path)
    torch.testing.assert_close(x, torch.tensor([[-1., 0., 127/128]]))
