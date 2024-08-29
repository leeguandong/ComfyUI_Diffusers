import os
import torch

import folder_paths
import comfy.model_management as mm

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    PNDMScheduler,

    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline
)

from huggingface_hub import snapshot_download


def download(model, name):
    model_name = model.rsplit('/', 1)[-1]
    model_dir = (os.path.join(folder_paths.models_dir, name, model_name))
    if not os.path.exists(model_dir):
        print(f"Downloading {model}")
        snapshot_download(repo_id=model, local_dir=model_dir, local_dir_use_symlinks=False)
        # huggingface-cli download --resume-download --local-dir-use-symlinks False LinkSoul/LLaSM-Cllama2 --local-dir LLaSM-Cllama2


class HFSchedulerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (
                    [
                        'DPMSolverMultistepScheduler',
                        'DPMSolverMultistepScheduler_SDE_karras',
                        'DDPMScheduler',
                        'DDIMScheduler',
                        'LCMScheduler',
                        'PNDMScheduler',
                        'DEISMultistepScheduler',
                        'EulerDiscreteScheduler',
                        'EulerAncestralDiscreteScheduler'
                    ], {
                        "default": 'DDIMScheduler'
                    }),
                "beta_start": (
                    "FLOAT", {"default": 0.00085, "min": 0.0, "max": 0.001, "step": 0.00005, "display": "slider"}),
                "beta_end": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 0.1, "step": 0.01, "display": "slider"}),
                "beta_schedule": (["scaled_linear"], {"default": "scaled_linear"}),
                "clip_sample": ("BOOLEAN", {"default": False}),
                "set_alpha_to_one": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = ("SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "load_scheduler"
    CATEGORY = "DiffusersUtils"

    def load_scheduler(self,
                       scheduler,
                       beta_start,
                       beta_end,
                       beta_schedule,
                       clip_sample,
                       set_alpha_to_one):
        scheduler_config = {
            'beta_start': beta_start,
            'beta_end': beta_end,
            'beta_schedule': beta_schedule,
            'clip_sample': clip_sample,
            'set_alpha_to_one': set_alpha_to_one
        }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDIMScheduler':
            noise_scheduler = DDIMScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        else:
            raise TypeError(f"not support {scheduler}!!!")

        return (noise_scheduler,)


class SDXL_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["stabilityai/stable-diffusion-xl-base-1.0"],
                    {"default": "stabilityai/stable-diffusion-xl-base-1.0"},
                ),
                "scheduler": ("SCHEDULER",)
            },
            "optional": {
                "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "DiffusersUtils"

    def load_model(self, model, scheduler, variant, use_safetensors):
        device = mm.get_torch_device()

        download(model, name="checkpoints")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant=variant,
            use_safetensors=use_safetensors,
            scheduler=scheduler).to(device)
        return (pipeline,)


class ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["runwayml/stable-diffusion-v1-5"],
                    {"default": "runwayml/stable-diffusion-v1-5"},
                ),
                "scheduler": ("SCHEDULER",)
            },
            "optional": {
                "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "DiffusersUtils"

    def load_model(self, model, scheduler, variant, use_safetensors):
        device = mm.get_torch_device()

        download(model, name="checkpoint")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant=variant,
            use_safetensors=use_safetensors,
            scheduler=scheduler).to(device)
        return (pipeline,)


class HFControlnet_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "lllyasviel/sd-controlnet-canny",
                        "diffusers/controlnet-depth-sdxl-1.0"
                    ],
                    {"default": "diffusers/controlnet-depth-sdxl-1.0"}),
                "vae": ("VAE",),
                "optional": {
                    "variant": ("STRING", {"default": "fp16"}),
                    "use_safetensors": ("BOOLEAN", {"default": False})
                }
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_controlnet"
    CATEGORY = "DiffusersUtils"

    def load_controlnet(self, model, vae, variant, use_safetensors):
        device = mm.get_torch_device()
        download(model, "controlnet")
        controlnet = ControlNetModel.from_pretrained(
            model,
            vae=vae,
            variant=variant,
            use_safetensors=use_safetensors).to(device)
        controlnet.enable_model_cpu_offload()
        return (controlnet,)


class HFVAE_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": (["madebyollin/sdxl-vae-fp16-fix"], {"default": "madebyollin/sdxl-vae-fp16-fix"}),
                "optional": {
                    "variant": ("STRING", {"default": "fp16"}),
                    "use_safetensors": ("BOOLEAN", {"default": False})
                }}
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "DiffusersUtils"

    def load_vae(self, vae, variant, use_safetensors):
        device = mm.get_torch_device()
        download(vae, "vae")
        vae = AutoencoderKL.from_pretrained(vae,
                                            variant=variant,
                                            use_safetensors=use_safetensors).to(device)
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "HFSchedulerLoader": HFSchedulerLoader,
    "SDXL_ModelLoader": SDXL_ModelLoader,
    "ModelLoader": ModelLoader,
    "HFControlnet_ModelLoader": HFControlnet_ModelLoader,
    "HFVAE_ModelLoader": HFVAE_ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFSchedulerLoader": "Scheduler Loader",
    "SDXL_ModelLoader": "SDXL Model Loader",
    "ModelLoader": "Model Loader",
    "HFControlnet_ModelLoader": "Controlnet ModelLoader",
    "HFVAE_ModelLoader": "VAE ModelLoader"
}
