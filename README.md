# OminiControl Replicate Deployment

This repository contains the deployment configuration for running [OminiControl](https://github.com/Yuanshi9815/OminiControl) on [Replicate](https://replicate.com). OminiControl is a powerful and versatile subject-driven image generation model that enables precise control over image generation while maintaining high fidelity.

## Credits

This implementation is based on the OminiControl project:

- **Original Repository**: [OminiControl](https://github.com/Yuanshi9815/OminiControl)
- **Paper**: ["OminiControl: Control Any Elements in Any Images"](https://arxiv.org/abs/2411.15098)
- **Authors**: Yuan Shi*, Jing Shi*, Michael J. Black, Yebin Liu, Yiyi Liao

If you use this model, please cite:
```bibtex
@article{shi2023ominicontrol,
  title={OminiControl: Control Any Elements in Any Images},
  author={Shi, Yuan and Shi, Jing and Black, Michael J and Liu, Yebin and Liao, Yiyi},
  journal={arXiv preprint arXiv:2411.15098},
  year={2023}
}
```

## Overview

This deployment uses [Cog](https://github.com/replicate/cog) to package the OminiControl model, allowing you to:
- Generate images with specific subject control
- Choose between 512x512 and 1024x1024 resolutions
- Fine-tune the generation process with custom prompts
- Deploy easily to Replicate or run locally

## Requirements

- NVIDIA GPU with CUDA support
- [Cog](https://github.com/replicate/cog) installed
- [HuggingFace Account](https://huggingface.co/) with access token
- Python 3.11+

## Installation

1. Install Cog:
```bash
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

2. Clone this repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

3. Set up your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

## Project Structure

```
├── cog.yaml            # Cog configuration file
├── predict.py          # Main prediction script
├── download_weights.py # Script to download model weights
└── README.md          # This file
```

## Local Development

1. Download the model weights:
```bash
python download_weights.py
```

2. Build the Docker image:
```bash
cog build
```

3. Run predictions locally:
```bash
cog predict -i image=@path/to/your/image.jpg -i prompt="Your prompt here" -i resolution=512
```

## Example Usage

Here's a complete example of how to use the model locally:

```bash
# Build the image
cog build

# Run a prediction
cog predict \
  -i image=@examples/cat.jpg \
  -i prompt="A cat sitting on a moon surface, with Earth visible in the background" \
  -i resolution=512 \
  -i num_inference_steps=8
```

## API Parameters

- `image` (Path): Input image for conditioning
- `prompt` (string): Text prompt describing the desired output
- `resolution` (int): Output resolution, either 512 or 1024 (default: 512)
- `num_inference_steps` (int): Number of denoising steps (default: 8, range: 1-50)

## Memory Optimization

The implementation includes several optimizations:
- Dynamic loading of LoRA weights based on selected resolution
- Automatic mixed precision inference
- GPU memory cleanup after each prediction
- CPU offloading when possible

## Deploying to Replicate

1. Push your model:
```bash
cog push r8.im/username/model-name
```

2. Your model will be available at `https://replicate.com/username/model-name`

## Troubleshooting

If you encounter issues:

1. Verify your HuggingFace token is set correctly
2. Ensure you have enough GPU memory (at least 12GB recommended)
3. Check CUDA compatibility with installed PyTorch version
4. Clear GPU memory if you encounter CUDA out of memory errors:
```python
import torch
torch.cuda.empty_cache()
```

## License

This deployment configuration is provided under the MIT License. However, please note that the original OminiControl model has its own license and usage restrictions. Make sure to check and comply with their license terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For any issues or questions, please open an issue in the repository.