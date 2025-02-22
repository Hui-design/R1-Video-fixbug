# R1-Video-fixbug

最近有很多开源项目致力于将Deepseek-R1/GPRO引入到多模态任务中，其中[Open-R1-Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video/)就是应用在Video Understanding的一个工作。我们在复现它的过程中，发现代码存在一个bug(**截止至2025-02-22**), 导致参考模型在get_per_token_probs时出现了问题，导致KL散度项起到不好的作用。另外，我们发现不止是Open-R1-Video存在这个问题，[open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal/blob/main/src/open_r1/trainer/grpo_trainer.py)也存在这个问题，让我怀疑到底是不是真的有这个bug，于此我们做了一些探索。

## What is the bug?
在Open-R1-Video/src/open_r1_video/trainer/grpo_trainer.py的444-453行中，当前模型model($$\pi_{\theta}$$)和ref_model($$\pi_{ref}$$)都需要过一下get_per_token_logps函数来跑一次model的forward获得logp。在GRPO中，对于$\pi_{\theta}$和$\pi_{\ref}$来说只是参数不一样，但是它们的输入是一样的（如Fig.1）。然而，在Open-R1-Video和open-r1-multimodal代码的实现中，这两个的输入是不一样的(如图2)，ref_model的输入少了**prompt_inputs。我们进一步check了一下[R1-V](https://github.com/Deep-Agent/R1-V/blob/main/src/r1-v/src/open_r1/trainer/grpo_trainer.py), 发现model和ref_model的输入是一样的。
![Fig1](assert/fig1.jpg)
<center>Figure 1</center>
![Fig2](assert/fig2.jpg)
<center>Figure 2</center>
![Fig2](assert/fig3.jpg)
<center>Figure 3</center>

## What does the bug affect?
Intuitively, Open-R1-Video and R1-multimodal are incorrect, while R1-V is the correct one. So, what are the impacts of this bug?
1. Issue with the input_embeds for Reference Model :
In the code, prompt_inputs mainly contain two keys: "pixel_values_videos" and "video_grid_thw." These two variables represent the video input (in R1-multimodal, this is an image). When these variables are passed into get_per_token_logps, they enter the model.forward method of Qwen2VL (specifically, lines 1667-1703 of transformers/src/transformers/models/qwen2_vl
/modeling_qwen2_vl.py/Qwen2VLForConditionalGeneration.forward). If both pixel_values_videos and pixel_values are None, the input to Inputs_embedd will be the embedding of <video_pad>, rather than the embedding obtained from pixel_values through the vision_tower. In this case, the reference model does not see any visual information, leading to an erroneous reference.
2. Impact on KL Loss:
The KL loss is affected because the KL divergence calculation in grpo relies on the formula KL($$\pi_{\theta},\pi_{ref}$$). Since the logps output from the reference model ($$\pi_{ref}$$) is incorrect, the KL divergence becomes problematic. Specifically, during initialization, the parameters of the model and the reference model are identical, meaning that $$\pi_{\theta}$$ and $$\pi_{ref}$$ should have the same values. Therefore, the correct initial value of KL divergence should be 0. However, in R1-Video, due to the incorrect logp from the reference model, the initial value of KL divergence is not 0.
![Fig4](assert/fig4.jpg)
<center>Figure 4</center>
   
## The fixed version
To resolve the issue, you should add **prompt_inputs in the get_per_token_logps method for the reference model. This will fix the bug as of February 22, 2025.
```python
per_token_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
# Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
per_token_logps = per_token_logps[:, prompt_length - 1 :]

with torch.inference_mode():
   if self.ref_model is not None:
       ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
   else:
       with self.accelerator.unwrap_model(model).disable_adapter():
           ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
```


