# Jittor Version of DragGAN

A Jittor implementation of [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://arxiv.org/abs/2305.10973).

## Usage

### DragGAN

Run DragGAN GUI:
```bash
python visualizer_drag.py ./weights/jt_stylegan3_ffhq_weights_t.pkl
```

### StyleGAN3 

Randomly sample face results:
```
python gen_images.py --outdir=output --trunc=1 --seeds=2 --network=./weights/jt_stylegan3_ffhq_weights_t.pkl
```

## Acknowledgements

The original implementation of DragGAN is available at [DragGAN](https://github.com/XingangPan/DragGAN).

This repo is based on the Jittor implementation of StyleGAN3 at [Jittor_StyleGAN3](https://github.com/ty625911724/Jittor_StyleGAN3).