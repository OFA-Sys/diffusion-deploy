
# Diffusion deployment

This repository contains scripts for the deployment of diffusion models (based on [diffusers](https://github.com/huggingface/diffusers)) on both GPU (Nvidia) and CPU (Intel). The aim is to significantly speed up the inference of diffusion models. It provides **a  ~12x speedup on CPUs and a ~4x speedup on GPUs.**
Integrated with [small-stable-diffusion-v0](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0/), it could generate an image in just **5s** on the CPU.  

## CPU speedup    
We develop the diffusion deployment on CPU based on [Intel OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). The pipeline `OpenVINOStableDiffusionPipeline` is modified from `OnnxStableDiffusionPipeline`. The code used here is in `pipeline_openvino_stable_diffusion.py`.
####  Results    
Here are some experimental results on stable diffusion v1.4 for comparison with the default Pytorch CPU and Onnx pipeline.  

| Pipeline  | Pytorch CPU | Onnx         | OpenVINO        |
| --------- | ----------- | ------------ | --------------- |
| Time Cost | 397s        | 77s ± 2.56 s | 33.9 s ± 247 ms |
| Speedup   | 1           | 5.2          | 11.7            |  

*Test setting: CPU Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz / PNDM scheduler 50 steps)*

#### Prerequisites  
There are several limitations of OpenVINO now, Therefore, we only support the following platforms for CPU speedup.  
+ Ubuntu 18.04, 20.04, RHEL(CPU only) or Windows 10 - 64 bit
+ Python 3.7, 3.8 or 3.9 for Linux and only Python3.9 for Windows  

**Requirements**  
+ diffuers
+ transformers 
+ openvino runtime

To install `openvino runtime`, you could simply use `pip install onnxruntime-openvino==1.13.0`.
#### Usage 
To use this deployment, you could follow the following code:  
```py
# Load a onnx pipeline firstly.  
from diffusers import OnnxStableDiffusionPipeline
onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(
    "OFA-Sys/small-stable-diffusion-v0",
    revision="onnx",
    provider="CPUExecutionProvider",
)
# Convert it to OpenVINO pipeline.  
import pipeline_openvino_stable_diffusion
openvino_pipe = pipeline_openvino_stable_diffusion.OpenVINOStableDiffusionPipeline.from_onnx_pipeline(onnx_pipe)

# Generate images.
images = openvino_pipe("an apple, 4k") . 
```

## GPU speedup   

We develop the deployment on GPU based on TensorRT and its plugins. 
#### Comparison 
Here are some experimental results of stable-diffusion-v1-4 and [small-stable-diffusion-v0](https://huggingface.co/OFA-Sys/small-stable-diffusion-v0/).  
| Model\Pipeline         | Pytorch GPU | TensorRT | TensorRT Plugin |
| ---------------------- | ----------- | -------- | --------------- |
| Stable diffusion       | 3.94        | 1.44s    | 1.07s           |
| Small stable diffusion | 2.7s        | 1.01s    | 0.65s           |


#### Prerequisites   
See the `gpu_requirements.txt` for requirments. To use plugins, we need `tensorrt>=8.5`. If you use `tensorrt==8.4`, you could run it by deleting `trt.init_libnvinfer_plugins(TRT_LOGGER, '')` in `gpu-trt-infer-demo.py` and not adding `PLUGIN_LIBS` to `LD_PRELOAD`.   

#### Usage  
```sh
export PLUGIN_LIBS="/path/to/libnvinfer_plugin.so.8.5.1"
export HF_TOKEN="Your_HF_TOKEN"

mkdir -p onnx engine output
LD_PRELOAD=${PLUGIN_LIBS} python3 demo-diffusion.py "a beautiful photograph of Mt. Fuji during cherry blossom" --enable-preview-features --hf-token=$HF_TOKEN -v

```



## Contributions
Contributions to this repository are welcome. If you would like to contribute, please open a pull request and make sure to follow the existing code style.
