# GANs Tutorial (using mps)

For this tutorial, you can train using Macbook GPU(mps).

<br/>

# Reference

1. <a href='https://pytorch.org/tutorials/'>PyTorch Tutorial</a>
2. Ian Goodfellow et al, Generative Adversarial Nets, 2014

<br/>

# Setting up environment

### Environment
- M3 Macbook Pro 
- macOS Sonoma 14.5
- VSCode 1.90.2 (Universal)
- <a href='https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html'>Miniconda</a>

<br/>

### Install Python and PyTorch

- Python 3.9
- PyTorch 2.3.1

<br/>

### Example

```
conda create pytorch-mps python=3.9
conda activate pytorch-mps
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda env update --file requirements.yml
python -m ipykernel install --user --name pytorch-mps --display-name "Python 3.9(pytorch-mps)"
```

<br/>

### Verification

```
import sys
import torch
import pandas as pd
import sklearn as sk

print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")

mps_device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"mps is", "available" if torch.backends.mps.is_available() else "not available")
```

<br/>

### Varification result

```
Python 3.9.19 (main, May  6 2024, 14:39:30) 
[Clang 14.0.6 ]
Pandas 2.2.2
Scikit-Learn 1.4.2
mps is available
```

<br/>
<br/>
<br/>
<br/>

<div align='center'>
    Copyright. 92berra 2024
</div>