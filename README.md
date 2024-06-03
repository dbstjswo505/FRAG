# FRAG: Frequency Adaptive Group for Diffusion Video Editing, ICML 2024

[![arXiv](https://img.shields.io/badge/arXiv-FRAG-b31b1b.svg)](https://arxiv.org/abs/2307.10373) 

[teaser](./asset/frag_demo.mp4)

**FRAG** is a framework that enhances the quality of edited videos by effectively preserving high-frequency components.

[//]: # (### Abstract)
>In video editing, the hallmark of a quality edit lies in its consistent and unobtrusive adjustment. Modification, when integrated, must be smooth and subtle, preserving the natural flow and aligning seamlessly with the original vision. Therefore, our primary focus is on overcoming the current challenges in high quality edit to ensure that each edit enhances the final product without disrupting its intended essence. However, quality deterioration such as blurring and flickering is routinely observed in recent diffusion video editing systems. We confirm that this deterioration often stems from high-frequency leak: the diffusion model fails to accurately synthesize high-frequency components during denoising process. To this end, we devise Frequency Adapting Group (FRAG) which enhances the video quality in terms of consistency and fidelity by introducing a novel receptive field branch to preserve high-frequency components during the denoising process. FRAG is performed in a model-agnostic manner without additional training and validates the effectiveness on video editing benchmarks (i.e., TGVE, DAVIS).

## Environment
```
conda create -n frag python=3.9
conda activate frag
pip install -r requirements.txt
```
## DDIM inversion

Preprocess you video by running using the following command:
```
python ddim_inversion.py
```
## Editing
```
python frag.py
```

## Acknowledgement

This code is implemented on top of following contributions: [TAV](https://github.com/showlab/Tune-A-Video), [TokenFlow](https://github.com/omerbt/TokenFlow), [HuggingFace](https://github.com/huggingface/transformers), [FLATTEN](https://github.com/yrcong/flatten), [FateZero](https://github.com/ChenyangQiQi/FateZero), [Prompt-to-prompt](https://github.com/google/prompt-to-prompt) 

We thank the authors for open-sourcing these great projects and papers!

This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).
