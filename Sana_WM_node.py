 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from comfy_api.latest import  io
import uuid
import folder_paths
from .node_utils import  clear_comfyui_cache,tensor2image
from .Sana.inference_video_scripts.inference_sana_wm import (load_sana_wm_model,load_sana_wm_vae,
    load_sana_wm_text_encoder,gemma_encode_prompts,infer_sana_wm,prepare_camera_intrinsics,load_refiner,infer_refiner,decode_with_sana_vae_
    )
from .Sana.diffusion.refiner.diffusers_ltx2_refiner import ltx2_encode_prompt,load_ltx_te_connectors

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

MAX_SEED = np.iinfo(np.int32).max
node_sana_wm_path = os.path.dirname(os.path.abspath(__file__))

weigths_gguf_current_path = os.path.join(folder_paths.models_dir, "gguf")
if not os.path.exists(weigths_gguf_current_path):
    os.makedirs(weigths_gguf_current_path)
folder_paths.add_model_folder_path("gguf", weigths_gguf_current_path) #  gguf dir


class Sana_WM_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_Model",
            display_name="Sana_WM_SM_Model",
            category="Sana_WM_SM",
            inputs=[
                io.Combo.Input("diffusion_models",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, diffusion_models) -> io.NodeOutput:
        clear_comfyui_cache()
        checkpoint_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models != "none" else None
        model=load_sana_wm_model(checkpoint_path,node_sana_wm_path)
        return io.NodeOutput(model)
    
class Sana_WM_SM_VAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_VAE",
            display_name="Sana_WM_SM_VAE",
            category="Sana_WM_SM",
            inputs=[
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae")),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
                ],
            )
    @classmethod
    def execute(cls,vae) -> io.NodeOutput:
        clear_comfyui_cache()
        vae_path=folder_paths.get_full_path("vae",vae) if vae != "none" else None
        vae=load_sana_wm_vae(vae_path, device,node_sana_wm_path, )
        return io.NodeOutput(vae)    

    
class Sana_WM_SM_Gemma(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_Gemma",
            display_name="Sana_WM_SM_Gemma",
            category="Sana_WM_SM",
            inputs=[
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip")),

            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
                ],
        )
    @classmethod
    def execute(cls, clip) -> io.NodeOutput: 
        clip_path=folder_paths.get_full_path("clip",clip) if clip != "none" else None
        clear_comfyui_cache()
        _,clip=load_sana_wm_text_encoder(clip_path, node_sana_wm_path)
        return io.NodeOutput(clip)

class Sana_WM_SM_Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_Encode",
            display_name="Sana_WM_SM_Encode",
            category="Sana_WM_SM",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("streaming_prefetch_count", default=1, min=0, max=1024),
                io.String.Input("prompt",default="A first-person view from a strictly stationary observation point on a winding dirt path inside an enormous magical mushroom forest. Colossal mushroom stems rise like tree trunks on both sides, their broad ribbed caps forming layered canopies above a lantern-lit village nestled deep in the midground. A small round robot stands on the path ahead, surrounded by oversized leaves, beadlike dew drops, tiny mushrooms, vines, mossy soil, and scattered stones. The surfaces mix soft fungal textures, rough barklike stems, damp earth, glossy leaves, and warm metal lanterns. Purple and golden twilight fills the space, creating a whimsical yet physically grounded atmosphere with strong depth and scale. The observer s perspective remains fixed, with no dynamic camera movement and no actions taken by the person recording. Autonomous motion animates the world: glowing spores drift, giant butterflies flap overhead, lantern flames flicker, dew trembles on leaves, and distant village lights shimmer.",multiline=True),
                io.String.Input("negative_prompt",default= "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, 音频带有机械音、闷糊、回音、失真、电流声、爆音、杂音",multiline=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="te_cond"),
                ],
        )
    @classmethod
    def execute(cls, clip,streaming_prefetch_count, prompt,negative_prompt,) -> io.NodeOutput: 
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(node_sana_wm_path, "Sana/gemma-2-2b-it"))
        tokenizer.padding_side = "right"
        if streaming_prefetch_count<1:
            streaming_prefetch_count=None
            clip.to(device)
        cond, cond_mask, neg, neg_mask = gemma_encode_prompts(clip, tokenizer,streaming_prefetch_count, prompt, negative_prompt, device)
        print("cond shape:", cond.shape, "neg shape:", neg.shape)
        te_cond={"cond":cond,"cond_mask":cond_mask,"neg":neg,"neg_mask":neg_mask,"negative_prompt":negative_prompt,"prompt":prompt} 
        clear_comfyui_cache()
        if streaming_prefetch_count is None:
            clip.to("cpu")
        #torch.save(te_cond,os.path.join(folder_paths.get_output_directory(),"wm_gemma2_te_dict.pt"))
        return io.NodeOutput(te_cond)


class Sana_WM_SM_Camera(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_Camera",
            display_name="Sana_WM_SM_Camera",
            category="Sana_WM_SM",
            inputs=[
                io.Combo.Input("pi3x_pt",options= ["none"] + folder_paths.get_filename_list("checkpoints")),
                io.Image.Input("image",), 
                io.Int.Input("num_frames", default=81, min=24, max=4096, step=1),
                io.String.Input("action",default="w-80,jw-40,w-40,lw-60,w-100",multiline=False),
                io.Float.Input("translation_speed", default=0.055, min=0, max=1024,step=0.001),
                io.Float.Input("rotation_speed_deg", default=1.2, min=0, max=120,step=0.1),
                io.String.Input("camera_file",default="",multiline=False),
                io.Boolean.Input("save_intrinsics", default=False),
                io.String.Input("intrinsics_file",default="",multiline=False),
            ],
            outputs=[
                io.Conditioning.Output(display_name="cam_cond"),
                ],
        )
    @classmethod
    def execute(cls, pi3x_pt,image,num_frames,action,translation_speed,rotation_speed_deg, camera_file,save_intrinsics,intrinsics_file) -> io.NodeOutput: 
        image=tensor2image(image)
        pi3x_path=folder_paths.get_full_path("checkpoints",pi3x_pt) if pi3x_pt != "none" else None
        prefix = str(uuid.uuid4())[:8]
        save_path = os.path.join(folder_paths.get_output_directory(), f"{prefix}_intrinsics.npy")
        cropped, c2w, intrinsics_vec4,num_frames = prepare_camera_intrinsics(pi3x_path,image,num_frames,action,translation_speed,rotation_speed_deg,camera_file,device,save_intrinsics,save_path,intrinsics_file)
        cam_cond={"cropped":cropped,"c2w":c2w,"intrinsics_vec4":intrinsics_vec4,"num_frames":num_frames} 
        clear_comfyui_cache()
        #torch.save(cam_cond,os.path.join(folder_paths.get_output_directory(),"wm_cam_cond.pt"))
        return io.NodeOutput(cam_cond)


class Sana_WM_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_Sampler",
            display_name="Sana_WM_SM_Sampler",
            category="Sana_WM_SM",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("te_cond"),
                io.Conditioning.Input("cam_cond"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("steps", default=20, min=1, max=1024, step=1),
                io.Int.Input("fps", default=16, min=1, max=4096, step=1),
                io.Float.Input("cfg_scale", default=5.0, min=0.1, max=10.0, step=0.1, ),
                io.Int.Input("block_num", default=1, min=0, max=64,step=1),    
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Latent.Output(display_name="latent"),
            ],
        )
    
    @classmethod
    def execute(cls, model,vae,te_cond,cam_cond, seed, steps,fps,cfg_scale,block_num,) -> io.NodeOutput:
        clear_comfyui_cache()
        image,lat =infer_sana_wm(model,vae,te_cond,cam_cond,fps,steps,cfg_scale,None,seed,device,block_num)
        print("Inference completed, latent shape:", lat.shape)
        latent={"samples":lat.to(device),}
        torch.save(latent,os.path.join(folder_paths.get_output_directory(),"san_wm_lat_dict.pt"))
        return io.NodeOutput(image,latent)

class Sana_WM_SM_LTXModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_LTXModel",
            display_name="Sana_WM_SM_LTXModel",
            category="Sana_WM_SM",
            inputs=[
                io.Combo.Input("diffusion_models",options= ["none"] + folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),          
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
        )
    @classmethod
    def execute(cls, diffusion_models,gguf) -> io.NodeOutput: 
        diffusion_model_path=folder_paths.get_full_path("diffusion_models",diffusion_models) if diffusion_models != "none" else None
        gguf_path=folder_paths.get_full_path("gguf",gguf) if gguf != "none" else None
        model=load_refiner(diffusion_model_path or gguf_path,  os.path.join(node_sana_wm_path,"Sana/LTX2_Refiner"), )  # Load the refiner components to ensure they're cached and ready for use.
        return io.NodeOutput(model)
    
class Sana_WM_SM_LTXGemma(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_LTXGemma",
            display_name="Sana_WM_SM_LTXGemma",
            category="Sana_WM_SM",
            inputs=[
                io.Combo.Input("clip",options= ["none"] + folder_paths.get_filename_list("clip")),
                io.Combo.Input("gguf",options= ["none"] + folder_paths.get_filename_list("gguf")),
                io.Combo.Input("connectors",options= ["none"] + folder_paths.get_filename_list("clip")),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
                io.Clip.Output(display_name="connectors"),
                ],
        )
    @classmethod
    def execute(cls, clip,gguf,connectors) -> io.NodeOutput: 
        clip_path=folder_paths.get_full_path("clip",clip) if clip != "none" else None
        gguf_path=folder_paths.get_full_path("gguf",gguf) if gguf != "none" else None
        connectors_path=folder_paths.get_full_path("clip",connectors) if connectors != "none" else None
        clear_comfyui_cache()
        clip,connectors=load_ltx_te_connectors(clip_path or gguf_path,connectors_path,node_sana_wm_path)
        return io.NodeOutput(clip,connectors)

class Sana_WM_SM_LtxEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_LtxEncode",
            display_name="Sana_WM_SM_LtxEncode",
            category="Sana_WM_SM",
            inputs=[
                io.Clip.Input("clip"),
                io.Clip.Input("connectors"),
                io.Int.Input("streaming_prefetch_count", default=1, min=0, max=1024),
                io.String.Input("prompt",default="A first-person view from a strictly stationary observation point on a winding dirt path inside an enormous magical mushroom forest. Colossal mushroom stems rise like tree trunks on both sides, their broad ribbed caps forming layered canopies above a lantern-lit village nestled deep in the midground. A small round robot stands on the path ahead, surrounded by oversized leaves, beadlike dew drops, tiny mushrooms, vines, mossy soil, and scattered stones. The surfaces mix soft fungal textures, rough barklike stems, damp earth, glossy leaves, and warm metal lanterns. Purple and golden twilight fills the space, creating a whimsical yet physically grounded atmosphere with strong depth and scale. The observer s perspective remains fixed, with no dynamic camera movement and no actions taken by the person recording. Autonomous motion animates the world: glowing spores drift, giant butterflies flap overhead, lantern flames flicker, dew trembles on leaves, and distant village lights shimmer.",multiline=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="ltx_cond"),
                ],
        )
    @classmethod
    def execute(cls, clip, connectors, streaming_prefetch_count, prompt) -> io.NodeOutput: 

        cond, cond_mask = ltx2_encode_prompt(clip, connectors,prompt,os.path.join(node_sana_wm_path,"Sana/ltx_te"),device,streaming_prefetch_count)
        ltx_cond={"prompt_embeds":cond,"prompt_attention_mask":cond_mask,} 
        clear_comfyui_cache()
        torch.save(ltx_cond,os.path.join(folder_paths.get_output_directory(),"san_wm_te_dict.pt"))
        return io.NodeOutput(ltx_cond)
   
class Sana_WM_SM_LTXSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sana_WM_SM_LTXSampler",
            display_name="Sana_WM_SM_LTXSampler",
            category="Sana_WM_SM",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("fps", default=16, min=1, max=4096, step=1),
                io.Int.Input("block_num", default=1, min=0, max=64,step=1),   
                io.Boolean.Input("enable_tile", default=False),
                io.Conditioning.Input("ltx_cond",optional=True),
                io.Latent.Input("latent",optional=True), 
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
    @classmethod
    def execute(cls, model,vae,seed,fps,block_num,enable_tile,ltx_cond=None,latent=None,) -> io.NodeOutput: 
        if ltx_cond is None:
            ltx_cond=torch.load(os.path.join(folder_paths.get_output_directory(),"san_wm_te_dict.pt"))
        if latent is None:
            latent=torch.load(os.path.join(folder_paths.get_output_directory(),"san_wm_lat_dict.pt"))
        latent=infer_refiner(model,latent["samples"].to(device),ltx_cond,device,fps,seed,block_num)
        image=decode_with_sana_vae_( vae,latent, device,enable_tile )
        
        return io.NodeOutput(image)
    
