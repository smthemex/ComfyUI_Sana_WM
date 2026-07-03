from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .Sana_WM_node import (Sana_WM_SM_Model,Sana_WM_SM_VAE,Sana_WM_SM_Gemma,Sana_WM_SM_LTXSampler,Sana_WM_SM_LtxEncode,Sana_WM_SM_LTXGemma,
                           Sana_WM_SM_Encode , Sana_WM_SM_Sampler,Sana_WM_SM_Camera,Sana_WM_SM_LTXModel)


class Sana_WM_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Sana_WM_SM_Model,
            Sana_WM_SM_VAE,
            Sana_WM_SM_Gemma,
            Sana_WM_SM_Encode,
            Sana_WM_SM_Sampler,
            Sana_WM_SM_Camera,
            Sana_WM_SM_LTXModel,
            Sana_WM_SM_LTXSampler,
            Sana_WM_SM_LtxEncode,
            Sana_WM_SM_LTXGemma,
        ]   

async def comfy_entrypoint() -> Sana_WM_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return Sana_WM_SM_Extension()


