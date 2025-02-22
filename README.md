# R1-Video-fixbug

最近有很多开源项目致力于将Deepseek-R1/GPRO引入到多模态任务中，其中[Open-R1-Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video/)就是应用在Video Understanding的一个工作。我们在复现它的过程中，发现存在一个bug, 导致参考模型在get_probs时出现了问题，导致KL散度项起到负面的作用。我们发现不止是Open-R1-Video存在这个问题，open-r1-multimodal也存在的这个问题，让我怀疑到底是不是真的有这个bug，于此我们做了一些探索。

## What is the bug?
在Open-R1-Video/src/open_r1_video/trainer/grpo_trainer.py的444-453行中，当前模型model$\pi_{\theta}$和ref_model$\pi_{\ref}$都需要过一下get_per_token_logps函数来跑一次model的forward获得logp。在GRPO中，对于$\pi_{\theta}$和$\pi_{\ref}$来说只是参数不一样，但是它们的输入是一样的，如图1。但是在代码的实现中，这两个的输入是不一样的。我们进一步check了一下[R1-V], 发现它是一样的。直觉上，R1-Video和R1-multimodel是错的，R1-V才是对的，那么这个bug有什么影响呢？

## What does the bug affect?
1. Ref_model的ref_embeeding有问题。代码中的**prompt_inputs主要包含了两个key, ["pixel_values_videos"] 和 "video_grid_thw"，这两个变量代表了视频的输入（在R1-multimodel中这个是image）. 这个输入进入get_per_token_logps后，会进入Qwen2VL的model.forward (Qwen2VLForConditionalGeneration.forward) 的1667-1703。如果pixel_values_videos和pixel_values都是None,那么Inputs_embedd的输入就是<Video_pad>的embedding而不是pixel_values经过vision_tower得到的embedding。这就到时ref_model根本没有看到视觉信息，起到错误的参考意义。
2. 导致了Kl_loss有问题。由于grpo中需要计算kl(model, ref_model), 由于ref_model输出的logp存在问题，那么这时kl就出现问题。特别地，初始化时R1-Video正确的kl散度的初始值应该是0。而在R1-Video中由于ref_model的logp是错的，导致kl散度的初始值不是0.
   
## The fixed version

```

```



