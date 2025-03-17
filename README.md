# ML_news_private

this is just a placeholder, the organized and correct repository is [here](https://github.com/SalvatoreRa/ML-news-of-the-week)

# scheme

# ML news: 

## Research
|Link|description|
|---|---|
|[.]() | |
|[.]() | |
|[.]() | |

## News
|Link|description|
|---|---|
|[.]() | |
|[.]() | | 
|[.]() | |


## Resources
|Link|description|
|---|---|
|[.]() | |
|[.]() | |
|[.]() | |


## Perspectives
|Link|description|
|---|---|
|[.]() | |
|[.]() | |
|[.]() | |


#############################################
# On working

# ML news: 

## Research
|Link|description|
|---|---|
|[Traveling Waves Integrate Spatial Information Through Time.](https://arxiv.org/abs/2502.06034) |Harvard and Western University researchers propose a wave-based recurrent neural network that uses traveling neural waves for global spatial integration in visual tasks. Inspired by the "Hearing the Shape of a Drum" analogy, they model neural activity as locally coupled oscillators, discretizing the 2D wave equation into a convolutional recurrent network. Unlike standard RNNs, their model aggregates information across entire wave evolution, enhancing segmentation tasks requiring large receptive fields. It rivals deeper CNNs and U-Nets with fewer parameters on both synthetic and real-world datasets. The approach may also align with neuroscience, reflecting observed traveling wave dynamics in the brain. |
|[Transformers without Normalization.](https://arxiv.org/abs/2503.10622) |Researchers from Meta, NYU, MIT, and Princeton introduce Dynamic Tanh (DyT), a method that removes normalization layers in Transformers while maintaining or improving performance. They observe that LayerNorm behaves like a tanh function, leading to DyT, which replaces normalization with a per-channel tanh(αx) and learnable affine parameters, preserving non-linearity without activation statistics. DyT ensures stable convergence across vision, speech, diffusion, and language tasks, matching or exceeding LayerNorm-based models while improving efficiency and training speed in large LLaMA models. Its simplicity reduces computation overhead and integrates easily into existing architectures with minimal modifications. |
|[Monitoring Reasoning Models for Misbehavior.](https://cdn.openai.com/pdf/34f2ada6-870f-4c26-9790-fd8def56387f/CoT_Monitoring.pdf) | OpenAI researchers explore how monitoring chain-of-thought (CoT) reasoning in LLMs can detect misaligned behaviors, including reward hacking. By analyzing reasoning traces rather than just outputs, they identify sophisticated exploits in coding tasks, often outperforming action-based monitoring. However, optimizing rewards using CoT monitors can lead to "obfuscated reward hacking," where models conceal their intent while still cheating. Surprisingly, weaker models can effectively oversee stronger ones, suggesting scalable oversight is possible. The study also highlights a trade-off: optimizing CoT for safety or brevity may reduce transparency, making future monitoring and alignment more difficult.|
|[Improving Planning of Agents for Long-Horizon Tasks.](https://arxiv.org/abs/2503.09572) |Researchers from UC Berkeley and the University of Tokyo introduce Plan-and-Act, a framework that separates high-level planning from execution in LLM-based agents, improving performance on long-horizon tasks. It consists of a Planner that structures goals into steps and an Executor that carries them out, reducing cognitive overload. To train these modules efficiently, they generate synthetic plan-action pairs by reverse-engineering successful executions and expanding them with LLM-powered augmentation, avoiding costly manual labeling. The system also dynamically updates plans based on new information, enabling real-time adjustments. Tested on WebArena-Lite, it achieves a 54% success rate, surpassing previous benchmarks and demonstrating the power of structured planning with synthetic training data. |
|[Gemini Robotics: Bringing AI into the Physical World.](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf) | Google DeepMind introduces Gemini Robotics, a family of embodied AI models that integrate large multimodal reasoning into robotics, enabling perception, interpretation, and interaction in 3D environments. Built on Gemini 2.0, it includes Gemini Robotics-ER for spatial reasoning and a real-time system for precise robotic control in tasks like object manipulation. By combining multi-view correspondence, 3D detection, and trajectory planning, the model executes diverse tasks with minimal data, reducing training costs. It generalizes well to new instructions and conditions while incorporating safety checks for real-world deployment. This marks a major step toward universal robotics, aiming for advanced planning and sensorimotor control in practical applications.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |

## News
|Link|description|
|---|---|
|[.]() | |
|[.]() | | 
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |

## Resources
|Link|description|
|---|---|
|[Gemma 3 Technical Report.](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) | Gemma 3 is a lightweight open model family (1B–27B parameters) with multimodal capabilities, extended context length (up to 128K tokens), and multilingual support. It integrates a frozen SigLIP vision encoder, using a Pan & Scan method for better image processing and handling diverse aspect ratios. Its hybrid attention system reduces memory usage for long-context tasks. Advanced knowledge distillation and quantization (int4, switched-fp8) allow for efficient deployment on consumer GPUs and edge devices. Instruction tuning enhances performance in benchmarks like MMLU, coding, and chat, placing it among the top models. Supporting over 140 languages, it enables structured outputs and function calling for agentic workflows while ensuring safety and privacy through data filtering and reduced memorization.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |

## Perspectives
|Link|description|
|---|---|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |













































































































