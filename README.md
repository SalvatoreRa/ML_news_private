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


































































































