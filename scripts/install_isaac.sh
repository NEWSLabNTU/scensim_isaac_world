#!/bin/bash
# Isaac Sim 6.2 pip install (roadmap 036): Python 3.12 from uv, everything under ISAAC_ROOT.
# Kit's caches default to ~/.cache/ov, ~/.local/share/ov and ~/.nvidia-omniverse; on this host
# they are symlinks into ISAAC_ROOT so the shader cache stays off /home. Then: pip install pyzmq.
set -eu
I=${ISAAC_ROOT:?ISAAC_ROOT is not set: where to install Isaac Sim 6.2 (a fast disk with ~60 GB free)}
export UV_CACHE_DIR=$I/uv-cache
# A cache symlink whose target is gone (an ISAAC_ROOT deleted and reinstalled) hangs Kit at
# start: omni.client loops on mkdir (EEXIST) and stat (ENOENT) at 100 % CPU, silently.
for c in ~/.cache/ov ~/.local/share/ov ~/.nvidia-omniverse; do
    if [ -L "$c" ] && [ ! -e "$c" ]; then mkdir -p "$(readlink -m "$c")"; echo "created $(readlink -m "$c") for $c"; fi
done
[ -d $I/venv ] || uv venv --python 3.12 $I/venv
source $I/venv/bin/activate
uv pip install --upgrade pip
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
# isaacsim[all] brings torchvision 0.26.0 from PyPI, built for CUDA 13, against torch 2.11.0+cu128:
# `import torchvision` then fails ("operator torchvision::nms does not exist"). The cu128 build.
uv pip install torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torchvision
uv pip install pyzmq  # the scensim world protocol (037) and the lockstep spike
uv pip install pytest  # aib's test-isaac lane imports it (two of its suites; no Isaac venv had it)
python -c "import importlib.metadata as m; print('isaacsim', m.version('isaacsim'))"
PYTHONNOUSERSITE=1 python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
echo INSTALL-DONE
