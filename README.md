# ComfyUI_Sana_WM
[SANA-WM](https://github.com/NVlabs/Sana): Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

# Update
  PR #1: Add codes
  you can run sana-wm only, or run it with refiner ,可以先跑sana-wm，效果满意后（比如没有肢体错落或者画面崩掉），再精炼放大

1.Installation  
----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Sana_WM
```
2.requirements  
----
```
pip install -r requirements.txt
```
3.checkpoints 
----
  
links: [sana-dit,refiner-connectors,refiner-dit,sana-vae](https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional)  
links: [gemma-3-12b-it-qat-Q4_0.gguf](https://huggingface.co/smthem/LTX-2.3-test-gguf/tree/main)  
links: [pi3x](https://huggingface.co/yyfz233/Pi3X/tree/main)
links: [gemma2b](https://huggingface.co/google/gemma-4-E2B-it/tree/main)
```
├── ComfyUI/models/diffusion_models/
|     ├── sana-wm-refiner.safetensors # 40G  # refiner
|     ├── sana-wm.safetensors #10 G
├── ComfyUI/models/checkpoints/
|     ├── pi3x.safetensors
├── ComfyUI/models/vae/
|     ├── ltx2-diff-vae.safetensors # rename from diffusion_pytorch_model.safetensors
├── ComfyUI/models/clip/
|     ├── gemma-2-2b-it-bf16.safetensors
|     ├── sana-wm-refiner-connectors.safetensors # refiner
├── ComfyUI/models/gguf/
|     ├── gemma-3-12b-it-qat-Q4_0.gguf # refiner

```


4.Example
----
![](https://github.com/smthemex/ComfyUI_Sana_WM/blob/main/example_workflows/example.png)


BibTeX
----
```
@misc{xie2024sana,
      title={Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer},
      author={Enze Xie and Junsong Chen and Junyu Chen and Han Cai and Haotian Tang and Yujun Lin and Zhekai Zhang and Muyang Li and Ligeng Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2410.10629},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10629},
    }
```
