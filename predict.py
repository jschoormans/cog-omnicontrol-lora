import os
import torch
from PIL import Image
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers.pipelines import FluxPipeline
from src.condition import Condition
from src.generate import seed_everything, generate
from download_weights import download_models

torch_float = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

class Predictor(BasePredictor):
    def setup(self):
        """Load the base model into memory but defer LoRA loading until prediction time"""

        download_models()
        
        # Load base model with minimal memory footprint
        self.pipe = FluxPipeline.from_pretrained(
            "models/flux-base",
            torch_dtype=torch_float,
            # device_map="auto"  # Automatically optimize device placement
        )
        
        # Initialize cache for LoRA weights
        self.current_resolution = None
        self.lora_cache = {}
    
    def load_lora_for_resolution(self, resolution):
        """Dynamically load LoRA weights based on requested resolution"""
        if self.current_resolution == resolution:
            return  # Already loaded
            
        # Clear previous LoRA if any
        if self.current_resolution is not None:
            self.pipe.unload_lora_weights()
            torch.cuda.empty_cache()
        
        # Load appropriate LoRA weights
        lora_path = "models/omini-control"
        if resolution == 512:
            weight_name = "omini/subject_512.safetensors"
        else:  # 1024
            weight_name = "omini/subject_1024_beta.safetensors"
            
        self.pipe.load_lora_weights(
            lora_path,
            weight_name=weight_name,
            adapter_name=f"subject_{resolution}"
        )
        self.current_resolution = resolution

    def predict(
        self,
        image: Path = Input(description="Input image for conditioning"),
        prompt: str = Input(description="Text prompt for generation"),
        resolution: int = Input(
            description="Output resolution (512 or 1024)",
            choices=[512, 1024],
            default=512
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", 
            default=8,
            ge=1,
            le=50
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Load appropriate LoRA weights
        self.load_lora_for_resolution(resolution)
        
        # Load and preprocess image
        input_image = Image.open(image)
        w, h = input_image.size
        min_size = min(w, h)
        
        # Center crop
        input_image = input_image.crop(
            (
                (w - min_size) // 2,
                (h - min_size) // 2,
                (w + min_size) // 2,
                (h + min_size) // 2,
            )
        )
        input_image = input_image.resize((512, 512))

        # Create condition
        condition = Condition("subject", image)

        # Move model to GPU just for inference if using CPU offloading
        if hasattr(self.pipe, "to"):
            self.pipe.to("cuda")

        # Generate image
        with torch.cuda.amp.autocast():  # Use automatic mixed precision
            output = generate(
                pipeline=self.pipe,
                prompt=prompt.strip(),
                conditions=[condition],
                num_inference_steps=num_inference_steps,
                height=resolution,
                width=resolution,
            ).images[0]
        
        # Move model back to CPU if using offloading
        if hasattr(self.pipe, "to"):
            self.pipe.to("cpu")
            torch.cuda.empty_cache()
        
        # Save and return the generated image
        output_path = Path("output.png")
        output.save(output_path)
        return output_path