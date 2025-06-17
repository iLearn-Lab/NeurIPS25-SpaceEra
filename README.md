<h1 align="center"> <a href=''>Spatial Understanding from Videos: <br>Structured Prompts Meet Simulation Data</a></h2>

> **TL;DR:** A framework that integrates spatial prompting and fine-tuning to enhance the 3D spatial understanding of VLMs from scanning videos.

## 📖 Abstract
Visual-spatial understanding, the ability to infer object relationships and layouts from visual input, is fundamental to downstream tasks such as robotic navigation and embodied interaction. However, existing methods face spatial uncertainty and data scarcity, limiting the 3D spatial reasoning capability of pre-trained vision-language models (VLMs). To address these challenges, we present a unified framework for enhancing 3D spatial reasoning in pre-trained VLMs without modifying their architecture. This framework combines SpatialMind, a structured prompting strategy that decomposes complex scenes and questions into interpretable reasoning steps, with ScanForgeQA, a scalable question-answering dataset built from diverse 3D simulation scenes through an automated construction process designed for fine-tuning. Extensive experiments across multiple benchmarks demonstrate the individual and combined effectiveness of our prompting and fine-tuning strategies, and yield insights that may inspire future research on visual-spatial understanding.

## 📝 TODO List
- [] Release of process documentation to support dataset construction
- [] Release of the constructed dataset
- [] Release of the source code
- [x] Release of the [arXiv preprint](https://arxiv.org/abs/2506.03642)

## 🔧 SpatialMind Prompting Strategy
We propose a **SpatialMind** prompting strategy that enhances the spatial reasoning capabilities of VLMs without the need for fine-tuning. It consists of two main components: 1) Scene Decomposition, where the 3D scene depicted in the video is transformed into multiple different representations; and 2) Question Decomposition, in which the question is brokendown into a sequence of fine-grained reasoning steps.

![](asset/prompting.png)

## 🎯 ScanForgeQA Dataset Construction
In addition to the prompting strategy, we also construct a novel dataset based on simulated scenarios, **ScanForgeQA**, to further enhance the spatial understanding capabilities of VLMs through training. The construction of the ScanForgeQA dataset involves a three-stage pipeline, including: 1) Scene Construction, where single-room 3D environments are created; 2) Scan Creation, in which egocentric videos are simulated by scanning through the constructed scenes; and 3) QA Generation, where textual question-answering pairs are automatically generated based on object annotations and the spatial layout of each scene.

![](asset/data.png)

## 🏆 Evaluation
We have validated our approach through extensive experiments across multiple benchmarks. Experimental results validate the effectiveness and generalizability of both SpatialMind  and ScanForgeQA, with their combination achieving further gains and providing valuable insights for future research.

![](asset/result.png)

## 🎓 Citation
If our work is helpful to you, please cite our paper.

```
@article{zhang2025spatial,
  title={Spatial Understanding from Videos: Structured Prompts Meet Simulation Data},
  author={Zhang, Haoyu and Liu, Meng and Li, Zaijing and Wen, Haokun and Guan, Weili and Wang, Yaowei and Nie, Liqiang},
  journal={arXiv preprint arXiv:2506.03642},
  year={2025}
}
```

## 🙏 Acknowledgements
We thank the authors of [VSI-Bench](https://github.com/vision-x-nyu/thinking-in-space) for releasing their evaluation benchmark, and we also acknowledge the authors from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for providing an easy-to-use and efficient platform for training and fine-tuning VLMs.

## 🔖 License
[MIT License]()

