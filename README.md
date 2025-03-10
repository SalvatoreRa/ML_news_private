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
|[The First Few Tokens Are All You Need.](https://arxiv.org/abs/2503.02875) |Researchers from Tencent AI Lab and The Chinese University of Hong Kong, Shenzhen propose a method to enhance reasoning in large language models (LLMs) by fine-tuning only the first few tokens of generated solutions. This approach focuses on Prefix Self-Consistency, where the initial tokens often share core reasoning steps, making fine-tuning on these prefixes effective. It uses Minimal Token Training, which reduces computational cost by up to 16 times compared to full-chain fine-tuning while maintaining reasoning structure. Despite being unsupervised, this method performs as well as or better than more computationally intensive methods. It works across various LLM architectures and can scale from small to large datasets, with the option to incorporate ground-truth checks to improve accuracy. |
|[Cognitive Behaviors that Enable Self-Improving Reasoners.](https://arxiv.org/abs/2503.01307) |Researchers from Stanford University and collaborators examine why some language models excel in reinforcement learning (RL)-based self-improvement, while others plateau. They identify four key cognitive behaviors—verification, backtracking, subgoal setting, and backward chaining—that are crucial for problem-solving in both humans and models. The study finds that models exhibiting these behaviors, like Qwen-2.5-3B, perform better in RL tasks than those that don't, like Llama-3.2-3B. Introducing cognitive behaviors through priming also boosts performance, with reasoning patterns playing a significant role. Curating pretraining data to emphasize these behaviors can enhance model performance, even for those initially underperforming. These cognitive behaviors also generalize to other reasoning tasks, suggesting that targeted priming and pretraining modifications can greatly improve a model's ability for self-improvement. |
|[Forecasting Rare Language Model Behaviors.](https://arxiv.org/abs/2502.16797) | A team from Anthropic and collaborators developed a method to predict rare failures that may only emerge at deployment scale, allowing developers to address issues early. They estimate the risk of undesirable behavior by sampling multiple outputs and measuring the likelihood of harmful responses, even from seemingly safe prompts. The study reveals that the probability of worst-case failures increases with the number of queries sampled, enabling prediction of extreme risks from smaller-scale tests. They introduce metrics like worst-query risk, behavior frequency, and aggregate risk, which can be extrapolated to larger-scale deployments. The approach also improves red-teaming by identifying which models or sampling strategies are most effective at uncovering potential failures, optimizing resources before models face billions of queries.|
|[Differentiable Logic Cellular Automata.](https://google-research.github.io/self-organising-systems/difflogic-ca/?hn) |A team from Google’s Paradigms of Intelligence introduces a discrete version of Neural Cellular Automata (NCA) by replacing floating-point layers with Differentiable Logic Gate Networks. This approach uses binary vectors for each cell's state, updated by learned logic circuits, enabling interpretable local rules and end-to-end differentiable training. Unlike traditional NCAs that rely on continuous neurons, this model uses learnable AND/OR/XOR gates, converted to binary gates for inference. The system successfully replicates Conway’s Game of Life and can generate complex patterns like checkerboards and images. It also demonstrates fault tolerance and supports asynchronous updates. This discrete, interpretable framework shows promise for robust, flexible computing in areas like programmable matter. |
|[How Well do LLMs Compress Their Own Chain-of-Thought?](https://arxiv.org/abs/2503.01141) |This paper explores how large language models (LLMs) balance the length of chain-of-thought (CoT) reasoning with accuracy. It introduces the concept of token complexity, which represents the minimum number of tokens required to solve a problem correctly. The study shows that various CoT "compression prompts," like "use bullet points" or "remove grammar," follow the same universal accuracy-length trade-off curve, indicating that reasoning length, not formatting, primarily influences accuracy. The authors also highlight that if the CoT falls below the token complexity threshold, the model fails. They propose that CoT compression can be seen as a "lossy coding" problem, with current prompting methods far from theoretical limits, leaving room for improvement. The optimal approach would involve adapting CoT length based on problem difficulty, using fewer tokens for easier tasks and more detailed reasoning for complex ones. |
|[LADDER: Self-Improving LLMs Through Recursive Problem Decomposition.](https://arxiv.org/abs/2503.00735) | LADDER is a framework that enables LLMs to recursively generate and solve simpler versions of complex problems, improving math integration accuracy. It allows models to autonomously create easier problem variants and use reinforcement learning with a verifier, establishing a self-guided learning process without needing human feedback or curated datasets. The framework introduces Test-Time Reinforcement Learning (TTRL), where problem variants are generated during inference, refining solutions on simpler tasks to increase final accuracy (e.g., improving from 73% to 90% on the MIT Integration Bee). LADDER uses generalizable numeric verification, allowing its application in fields like code testing or theorem proving, where straightforward checks are available.|
|[Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems.](https://arxiv.org/abs/2502.19328) |This paper introduces Agentic Reward Modeling, a new reward framework that combines human preference models with "verifiable correctness" signals to provide more reliable rewards for training and evaluating LLMs. The framework uses a modular system, REWARDAGENT, which includes a router to determine necessary checks (e.g., factual accuracy, adherence to instructions), specialized verification agents for factual checks, and a judger that merges these signals with human preference scores. The system improves factual accuracy by comparing candidate responses and verifying differences through evidence, reducing costs. It also ensures instructions are followed by using auto-generated Python scripts for constraint checking, penalizing violations. REWARDAGENT outperforms existing models on challenging benchmarks and real-world tasks, offering significant improvements in accuracy and reliability when used for best-of-n search or DPO training. |
|[Fractal Generative Models.](https://arxiv.org/abs/2502.17437) |Researchers from MIT CSAIL and Google DeepMind introduce a fractal-based framework for generative modeling, where atomic generative modules are used recursively. This approach, which abstracts autoregressive models into modular units, efficiently handles high-dimensional data like raw pixels. The fractal method achieves state-of-the-art performance on ImageNet 64×64, outperforming previous methods, and can generate high-quality 256×256 images. It also allows for tasks like inpainting and semantic replacement. The design reduces computational costs, making pixel-by-pixel generation feasible at larger resolutions, and is open-sourced for wider use. |
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
|[.]() | |
|[.]() | |
|[.]() | |

## Resources
|Link|description|
|---|---|
|[LLM Post-Training: A Deep Dive into Reasoning Large Language Models.](https://arxiv.org/abs/2502.21321) | This survey examines methods for improving LLMs post-pretraining, including fine-tuning, reinforcement learning, and optimizing inference techniques. It also addresses challenges such as catastrophic forgetting, reward manipulation, and ethical concerns, providing a guide for developing more reliable and advanced AI systems.|
|[Crossing the uncanny valley of
conversational voice.](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) |Researchers from Sesame introduce a multimodal TTS approach designed for natural, context-aware speech in real-time conversational AI. Unlike traditional TTS, which lacks contextual awareness, this method addresses the "one-to-many" problem by conditioning on conversation history, speaker identity, and prosodic cues. The end-to-end model uses Residual Vector Quantization (RVQ) tokens and two autoregressive transformers for efficiency and expressivity, with a lightweight decoder to reduce computational load. Despite training on only a fraction of frames, the model maintains high fidelity. Evaluations show near-human accuracy in word error rates and speaker similarity, with scaling improving speech realism. However, challenges remain in capturing nuanced human prosody in conversational contexts. The team plans to release their models open-source and expand to more languages while refining conversational dynamics. |
|[Applications of Large Models in Medicine.](https://arxiv.org/abs/2502.17132) |Medical AI is progressing beyond basic diagnostics, with large models reshaping healthcare. A recent paper categorizes Medical Large Models (MedLMs) into clinical text analysis, medical imaging, anatomical representation, and multimodal systems. It also explores Large Graph Models, which can interpret complex biomedical relationships, offering significant potential. These models are improving diagnostic accuracy and transforming treatment planning and drug discovery. While the medical field has been cautious about AI, these advancements suggest we may be nearing a tipping point where their clinical value becomes undeniable. |
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

## Perspectives
|Link|description|
|---|---|
|[Are AI-generated video games really on the horizon?](https://www.theguardian.com/games/2025/mar/10/are-ai-generated-video-games-microsoft-muse-google-gamengen) | Microsoft and Google have both recently released new generative AI models that simulate video game worlds – with notable limitations. What can they do?|
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












































































































