import os
import torch
from PIL import Image
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers.pipelines import FluxPipeline
from src.condition import Condition
from src.generate import seed_everything, generate
from download_weights import download_models
import time
import subprocess
torch_float = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

MODEL_CACHE_TOP_DIR = "./model-cache"  # necessary for tars that also contain a directory.
SCHNELL_CACHE = "./model-cache/FLUX.1-schnell"
SCHNELL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def load_flux_with_loras(model_cache, model_url):
    if not os.path.exists(model_cache):
        download_weights(model_url, MODEL_CACHE_TOP_DIR)


class Predictor(BasePredictor):
    def setup(self):
        """Load the base model into memory but defer LoRA loading until prediction time"""


        load_flux_with_loras(SCHNELL_CACHE, SCHNELL_URL)
        # Load base model with minimal memory footprint

        # cog.server.exceptions.FatalWorkerException: Predictor errored during setup: DiffusionPipeline.from_pretrained() missing 1 required positional argument: 'pretrained_model_name_or_path'


        self.pipe = FluxPipeline.from_pretrained(
            SCHNELL_CACHE,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            # device_map="auto"  # Automatically optimize device placement
        )
        
        # Initialize cache for LoRA weights
        self.current_lora_name = None
        self.lora_cache = {}
    
    # def load_lora_for_resolution(self, resolution):
    #     """Dynamically load LoRA weights based on requested resolution"""
    #     if self.current_resolution == resolution:
    #         return  # Already loaded
            
    #     # Clear previous LoRA if any
    #     if self.current_resolution is not None:
    #         self.pipe.unload_lora_weights()
    #         torch.cuda.empty_cache()
        
    #     # Load appropriate LoRA weights
    #     lora_path = "models/omini-control"
    #     if resolution == 512:
    #         weight_name = "omini/subject_512.safetensors"
    #     else:  # 1024
    #         weight_name = "omini/subject_1024_beta.safetensors"
            
    #     self.pipe.load_lora_weights(
    #         lora_path,
    #         weight_name=weight_name,
    #         adapter_name=f"subject_{resolution}"
    #     )
    #     self.current_resolution = resolution


    def load_lora_for_unstaging(self):
        
        if self.current_lora_name is not None:
            self.pipe.unload_lora_weights()
            torch.cuda.empty_cache()
        
        lora_path = "models/unstaging"
        self.pipe.load_lora_weights(
            lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="subject"
        )
        self.current_lora_name = "subject"


    # write a function that reshapes the shortest side to N, keeping the aspect ratio
    def resize_shortest_side(self, image, N):
        w, h = image.size
        if w < h:
            return image.resize((N, h * N // w))
        else:
            return image.resize((w * N // h, N))



    def predict(
        self,
        image: Path = Input(description="Input image for conditioning"),
        prompt: str = Input(description="Text prompt for generation"),
        resolution: int = Input(
            description="Output resolution",
            default=768,
            ge=256,
            le=2048
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", 
            default=8,
            ge=1,
            le=50
        ),
        resolution_conditioning: int = Input(
            description="Resolution of the conditioning image",
            default=768,
            ge=256,
            le=2048
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # Load appropriate LoRA weights
        self.load_lora_for_unstaging()
        # Create condition


        image = Image.open(image).convert("RGB")

        image = self.resize_shortest_side(image, resolution_conditioning)
        condition = Condition("subject", image, position_delta=(0,0))

        # Move model to GPU just for inference if using CPU offloading
        if hasattr(self.pipe, "to"):
            self.pipe.to("cuda")

        # Generate image
        output = generate(
            pipeline=self.pipe,
            prompt=prompt.strip(),
            conditions=[condition],
            num_inference_steps=num_inference_steps,
            height=resolution,
            width=resolution,
        ).images[0]
    
        


        # Save and return the generated image
        output_path_fn = f"output_{self.current_lora_name}_{resolution}.png"
        output.save(output_path_fn)
        output_path = Path(output_path_fn)
        return output_path