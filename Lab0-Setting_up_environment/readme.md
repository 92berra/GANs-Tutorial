# Environment
- M3 Macbook Pro 
- macOS Sonoma 14.5
- VSCode 1.90.2 (Universal)
- <a href='https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html'>Miniconda</a>

<br/>

# Install Python and PyTorch

- Python 3.9
- PyTorch 2.3.1

<br/>

# Example

```
conda create pytorch-mps python=3.9
conda activate pytorch-mps
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda env update --file requirements.yml
python -m ipykernel install --user --name pytorch-mps --display-name "Python 3.9(pytorch-mps)"
```

<br/>
<br/>
<br/>

<div align='center'>
    Copyright. 92berra 2024
</div>