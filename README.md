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












































































































