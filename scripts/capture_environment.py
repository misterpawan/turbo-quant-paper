"""Print relevant package and GPU versions without recording credentials."""
import importlib.metadata as metadata
import json
import platform
import subprocess
import torch

versions = {}
for name in ('torch', 'transformers', 'tokenizers', 'datasets', 'accelerate',
             'bitsandbytes', 'numpy', 'pandas', 'scipy', 'scikit-learn'):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
try:
    driver = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
        text=True, stderr=subprocess.DEVNULL).strip()
except (OSError, subprocess.CalledProcessError):
    driver = None
print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
                  'packages': versions, 'cuda_build': torch.version.cuda,
                  'cuda_available': torch.cuda.is_available(), 'gpu_driver': driver}, indent=2))
