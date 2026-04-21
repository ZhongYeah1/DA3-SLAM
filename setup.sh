#!/bin/bash
set -e

echo "Installing base requirements..."
pip3 install -r requirements.txt

# Install gtsam-develop with SL(4) support from test.pypi.org
echo "Installing gtsam-develop with SL(4) support..."
pip install gtsam-develop --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/

# Install Depth-Anything-3 from sibling directory
echo "Installing Depth-Anything-3..."
pip install -e ../Depth-Anything-3

# Install SALAD for loop closure
mkdir -p third_party
cd third_party
if [ ! -d "salad" ]; then
    echo "Cloning SALAD..."
    git clone https://github.com/Dominic101/salad.git
fi
pip install -e ./salad

# Download SALAD pretrained checkpoint
CKPT_DIR="$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints"
CKPT_PATH="$CKPT_DIR/dino_salad.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading SALAD checkpoint..."
    mkdir -p "$CKPT_DIR"
    wget -q -O "$CKPT_PATH" \
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
    echo "SALAD checkpoint saved to $CKPT_PATH"
else
    echo "SALAD checkpoint already exists at $CKPT_PATH"
fi
cd ..

# Install this repo
echo "Installing DA3-SLAM..."
pip install -e .

echo "Installation Complete"
