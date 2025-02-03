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
|[Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs.](https://arxiv.org/abs/2501.18585) |This work examines the "thinking" patterns of o1-like LLMs in greater detail. Recent papers have highlighted issues related to overthinking, but now a new phenomenon, called underthinking, has been identified. What is it? The authors observe that o1-like LLMs often shift between different reasoning paths without fully exploring the most promising ones, which can hinder reaching the correct solution. |
|[Diverse Preference Optimization.](https://arxiv.org/abs/2501.18101) | Diverse Preference Optimization (DivPO) is a new training method that enhances the diversity of language model outputs without sacrificing quality. Unlike traditional approaches like RLHF, which often result in similar responses, DivPO selects diverse training pairs by comparing a highly diverse response with a less diverse one. It measures diversity using various criteria, such as model probability or word frequency. In tests on persona generation and creative writing, DivPO significantly increased output diversity while maintaining similar quality to existing methods.|
|[Challenges in Ensuring AI Safety in DeepSeek-R1 Models: The Shortcomings of Reinforcement Learning Strategies.](https://arxiv.org/abs/2501.17030) |This paper offers a collection of guidelines for effectively prompting the DeepSeek-R1 model. Key recommendations include crafting clear and well-structured prompts with explicit instructions, avoiding few-shot prompting in favor of zero-shot approaches, and specifying the desired output format, such as JSON, tables, or markdown. For reasoning tasks, requesting step-by-step explanations is advised. Additionally, it is important to clearly define the input and output language to prevent mixing. The paper also covers the appropriate use cases for different model variants, the best times to fine-tune the model, and important safety considerations. |
|[Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning.](https://arxiv.org/abs/2501.15228) |This work approaches RAG as a multi-agent cooperative task to enhance answer generation quality. It treats components like query rewriting, document selection, and answer generation as reinforcement learning agents collaborating to produce accurate answers. Multi-Agent Proximal Policy Optimization (MAPPO) is used to optimize all agents together, with a shared reward based on answer quality. In addition to improvements on well-known benchmarks, the framework demonstrates strong generalization in out-of-domain scenarios and remains effective across various RAG system configurations. |
|[TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs.](https://arxiv.org/abs/2501.15674) |This framework introduces a method for compressing MHA through a multi-head tensorization process and Tucker decomposition. It achieves a compression rate of up to approximately 250x in MHA weights, without the need for additional data, training, or fine-tuning. |
|[TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space.](https://arxiv.org/abs/2501.12224) |TokenVerse, introduced by Google DeepMind and collaborators, presents a new technique for generating images from learned concepts in a specific configuration. It enables multi-concept personalization by utilizing a pre-trained text-to-image diffusion model to separate and extract complex visual concepts from multiple images. Operating within the modulation space of DiTs, TokenVerse learns a personalized modulation vector for each text token in an input caption. This method provides flexible and localized control over distinct concepts like objects, materials, lighting, and poses. The learned token modulations can be combined in innovative ways to create new images that integrate multiple personalized concepts, all without the need for additional segmentation masks. |
|[AI to revolutionise fundamental physics and ‘could show how universe will end’.](https://www.theguardian.com/science/2025/feb/03/ai-to-revolutionise-fundamental-physics-and-could-show-how-universe-will-end) |Cern’s next director general Mark Thomson says AI is paving the way for huge advances in particle physics |
|[Was this the week DeepSeek started the slow unwinding of the AI bet?](https://www.theguardian.com/technology/2025/feb/01/was-this-the-week-deepseek-started-the-slow-unwinding-of-the-ai-bet) |The cheap Chinese chatbot has stunned tech giants – and opened up the possibility that other countries, not just China, could now afford to enter the AI race |
|[A Controlled Study on Long Context Extension and Generalization in LLMs.](https://github.com/leooyii/lceg) | This study examines how language models manage long-document contexts by evaluating different extension methods through a controlled analysis. It emphasizes that perplexity continues to be a crucial performance metric, while approximate attention techniques face challenges with longer contexts.|
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
|[Inside the “Virtual Lab” where AIs and humans collaborate.](https://www.freethink.com/artificial-intelligence/virtual-lab-interdisciplinary-research) |Stanford's "Virtual Lab" employs AI agents as partners in scientific research, with the goal of addressing complex challenges through interdisciplinary collaboration. Researchers showcase its capabilities in projects such as creating COVID-19 treatments by simulating expert interactions among AI agents. This framework enables scientists to build AI-driven expertise, presenting a fresh approach to collaborative research and innovation. |
|[Alibaba’s Qwen team releases AI models that can control PCs and phones.](https://techcrunch.com/2025/01/27/alibabas-qwen-team-releases-ai-models-that-can-control-pcs-and-phones/) |Chinese AI lab DeepSeek might be getting the bulk of the tech industry’s attention this week. But one of its top domestic rivals, Alibaba, isn’t sitting idly by. |
|[Quartz has been quietly publishing AI-generated news articles.](https://techcrunch.com/2025/01/27/quartz-has-been-quietly-publishing-ai-generated-news-articles/) | Quartz has been employing AI to create articles by aggregating content from sources such as CNN and TechCrunch through its "Quartz Intelligence Newsroom."|
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
|[OpenAI o3-mini.](https://cdn.openai.com/o3-mini-system-card.pdf) |OpenAI has introduced o3-mini, their latest cost-effective reasoning model, now available in ChatGPT and via API. This model excels in STEM tasks, particularly in science, math, and coding, while retaining the low cost and reduced latency of its predecessor, o1-mini. It also introduces important developer features such as function calling, Structured Outputs, and developer messages, ensuring it's production-ready from the start. o3-mini offers varying levels of reasoning effort (low, medium, and high) and enhances performance across a wide range of tasks. It provides responses 24% faster than o1-mini and has shown strong results in competition math, PhD-level science queries, and software engineering challenges. |
|[Qwen2.5-1M.](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf) |Qwen has released two open-source LLMs, Qwen2.5-7B-Instruct-1M and Qwen2.5-14B-Instruct-1M, capable of handling context lengths up to 1 million tokens. These models use a progressive training strategy, beginning with 4K tokens and gradually increasing to 256K tokens, before applying length extrapolation methods to achieve 1M tokens. They also offer an inference framework based on vLLM, which processes long inputs 3-7 times faster using sparse attention techniques. The models perform well on both long-context and short-text tasks. The 14B version surpasses GPT-4o-mini on several long-context datasets, while maintaining comparable results on shorter tasks. |
|[Janus-Pro.](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf) |An upgraded version of the previous Janus model for multimodal understanding and generation has been released. This new model includes three major improvements: optimized training strategies with longer initial training and targeted fine-tuning, expanded training data with 90 million new samples for understanding and 72 million synthetic aesthetic samples for generation, and scaling up to larger model sizes of up to 7B parameters. Janus-Pro delivers notable enhancements in both multimodal understanding and text-to-image generation. It outperforms existing models across several benchmarks, scoring 79.2 on MMBench for understanding tasks and achieving 80% accuracy on GenEval for text-to-image generation. These advancements also improve image generation stability and quality, particularly for short prompts and intricate details, though the current 384x384 resolution limits performance for some tasks. |
|[Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion.](https://github.com/DS4SD/docling) | Docling is an open-source toolkit designed to convert various popular document formats into a unified, richly structured representation.|
|[PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides.](https://arxiv.org/abs/2501.03936v1) | PPTAgent offers presentation generation through a two-stage, edit-based approach inspired by human workflows.|
|[1.58-bit FLUX.](https://arxiv.org/abs/2412.18653) | The 1.58-bit FLUX effectively quantizes the FLUX.1-dev text-to-image model with minimal weights, preserving its performance. This technique works without image data, depending on self-supervision. It greatly decreases model storage and memory usage, while enhancing inference speed.|
|[Phi-4.](https://huggingface.co/microsoft/phi-4) | Microsoft has released the benchmark topping synthetic data models on Hugging Face for commercial use due to the MIT license|
|[LLMs' Guardrails.](https://github.com/yueliu1999/guardreasoner) |GuardReasoner presents a reasoning-driven safeguard for LLMs, enhancing explainability and generalizability in safety-sensitive applications. It surpasses GPT-4o+CoT and LLaMA Guard 3 in various benchmarks. The training data, models, and code have been released to the public. |
|[aiMotive 3D Traffic Light and Traffic Sign Dataset.](https://github.com/aimotive/aimotive_tl_ts_dataset) |This project introduces a novel method for creating precise 3D bounding box annotations for traffic lights and road signs, which are essential for self-driving vehicles. |
|[OpenThoughts Dataset.](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |A comprehensive synthetic reasoning dataset from R1, containing 114k examples of reasoning tasks, which can be utilized to train powerful reasoners through distillation or serve as a starting point for RL cold start. |
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
|[Top AI Investor Says Goal Is to Crash Human Wages.](https://futurism.com/the-byte/ai-investor-goal-crash-human-wages) | Marc Andreessen proposes that AI should "crash" wages to create an economic utopia, focusing on productivity improvements and lower consumer prices. His perspective aligns with a broader tech industry mindset that emphasizes economic transformation over addressing job market disruptions. Critics point out the gap in the visions of tech leaders, which often fail to provide immediate solutions for workers impacted by these changes.|
|[Will DeepSeek Burst VC’s AI Bubble?](https://news.crunchbase.com/ai/chinas-deepseek-tech-openai-nvda/) |The launch of DeepSeek, a Chinese AI app that asserts better performance at lower costs, led to notable declines in tech stocks, including Nvidia. This development raises worries about the U.S. losing ground in AI, which significantly affects investors and VCs heavily invested in AI startups. As DeepSeek's model competes with established AI giants, it sparks concerns about future funding and the U.S.'s competitiveness in the global AI race. |
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


































































































