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
|[s1: Simple test-time scaling.](https://arxiv.org/abs/2501.19393) | Researchers from Stanford, UW, and others introduced s1, a method to enhance LLM performance by using additional compute during inference ("test-time scaling"). Key ideas include: Small but effective dataset – They created s1K, a set of 1,000 challenging questions with detailed reasoning, to fine-tune a 32B model. Despite the small size, it provides valuable reasoning examples. "Budget forcing" for reasoning – A new decoding method adds the token "Wait" when the model attempts to stop, encouraging it to rethink and correct its reasoning. It also limits excessive reasoning to control inference time. Significant improvements over OpenAI’s o1 – The fine-tuned model (s1-32B), based on Qwen2.5-32B-Instruct, outperforms OpenAI's o1-preview by up to 27% on math competitions (MATH & AIME24). Test-time scaling increases accuracy on AIME24 from 50% to 57%, exceeding its normal performance.|
|[OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models.](https://arxiv.org/abs/2502.01061) |ByteDance AI Lab introduced OmniHuman-1, a diffusion-transformer model that creates realistic human videos from a single image and motion input (audio or video). Key points: End-to-end human video generation – OmniHuman uses an image and audio or video to generate lifelike videos of people speaking or performing actions, with impressive detail in motion, lighting, and texture. Mixed modality training – Omni-Conditions Training combines various motion modalities during training, expanding data and overcoming the lack of high-quality talking-head videos. The model handles diverse inputs like speech, song, and complex poses. Outperforms prior methods – OmniHuman produces more realistic videos and works with a variety of inputs, including cartoons or animals, transferring motion naturally. Broader support – The model supports any portrait content (face, half-body, full-body) and multiple driving signals, offering more versatility than previous models. |
|[LIMO: Less is More for Reasoning.](https://arxiv.org/abs/2502.03387) |The LIMO paper challenges the need for large fine-tuning datasets in complex reasoning tasks, showing that a small set of carefully curated examples can be highly effective. With just 817 training samples, the LIMO model achieved impressive results, scoring 57.1% on the AIME math competition and 94.8% on MATH, far surpassing earlier models that required much more data. The model also demonstrated significant out-of-distribution generalization, outperforming models trained on 100 times more data by 40.5% on various benchmarks. The authors propose that when an LLM has strong pre-existing knowledge, only a minimal set of high-quality examples is necessary to unlock advanced reasoning skills. This suggests that small, well-designed datasets could enable state-of-the-art reasoning, lowering the barriers for fine-tuning LLMs. |
|[CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning.](https://arxiv.org/abs/2502.02390) |CoAT introduces a “slow thinking” inference framework that enhances LLM reasoning by allowing it to explore and update its thoughts more like a human. The system combines Monte Carlo Tree Search (MCTS) with associative memory, enabling the model to explore different reasoning branches and dynamically add relevant information as needed. This iterative approach allows the model to refine and revisit intermediate conclusions, improving accuracy and comprehensiveness compared to one-pass reasoning. In experiments, CoAT outperformed traditional methods on accuracy, coherence, and solution diversity. By mimicking human-like problem-solving, CoAT points toward LLMs that use search and memory for more reliable reasoning. |
|[Syntriever: How to Train Your Retriever with Synthetic Data from LLMs.](https://arxiv.org/abs/2502.03824) |Syntriever introduces a two-stage framework to build a high-quality text retriever without relying on large labeled datasets or access to an LLM’s internals. In Stage 1, the system distills knowledge by generating synthetic Q&A data. A powerful LLM (e.g., GPT-4) is prompted to create relevant and incorrect passages, with chain-of-thought ensuring variety. The LLM then filters out any low-quality data, resulting in a synthetic dataset that is used to train the retriever. In Stage 2, the retriever is further aligned with the LLM’s preferences using a partial Plackett-Luce ranking method to adjust its ranking decisions. Syntriever achieves state-of-the-art results on several retrieval benchmarks without needing any real training queries, all training data is generated synthetically by the LLM. It also eliminates the need for logits, making it applicable even to closed models. |
|[Demystifying Long Chain-of-Thought Reasoning in LLMs.](https://arxiv.org/abs/2502.03373) | This study examines how LLMs develop extended chain-of-thought (CoT) reasoning, focusing on reinforcement learning (RL) and compute scaling. It finds that supervised fine-tuning (SFT) improves accuracy by using long CoT sequences, and introduces a cosine length-scaling reward with repetition penalties to stabilize RL and prevent unnecessary reasoning lengthening. RL models trained with noisy, web-based supervision signals generalize better to out-of-distribution tasks, though filtering is essential for stability. Additionally, while skills like error correction exist in base models, effective RL incentives are needed to harness them for complex tasks. This paper provides a roadmap for enhancing CoT training with RL and reward tuning.|
|[Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?](https://arxiv.org/abs/2502.00674) |This paper explores whether mixing different LLMs in an ensemble (Mixture-of-Agents, MoA) improves performance or if using a single top model’s outputs is more effective. The surprising answer is that "Self-MoA," which generates multiple outputs from one strong model and aggregates them, often outperforms multi-model ensembles. Extensive tests show that Self-MoA yields better results, with a +6.6% score improvement on the AlpacaEval 2.0 benchmark compared to MoA, and +3.8% on tasks like MMLU, CRUX, and MATH. The study finds that adding weaker models in an MoA can dilute performance, and unless all models are strong and complementary, it’s better to rely on one top model’s outputs. They also propose a sequential version of Self-MoA that efficiently combines multiple outputs over rounds. |
|[Multi-agent Architecture Search via Agentic Supernet.](https://arxiv.org/abs/2502.04180) | MaAS (Multi-agent Architecture Search) automates the design of multi-agent systems for LLMs, where agents collaborate with specific roles or tools for each task. Instead of hand-designing a complex pipeline, MaAS learns a flexible “agentic supernet” that can generate an optimal agent team for each query. It defines a continuous space of possible agent configurations and dynamically selects the best one based on the query's domain and difficulty, allowing for efficient resource allocation. MaAS outperforms traditional multi-agent systems in accuracy by 0.5–11.8%, while using only 6–45% of the inference cost. Its approach also shows strong generalization, transferring well to new tasks and LLM backbones.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[gambling firms secretly sharing users’ data with Facebook without permission.](https://www.theguardian.com/society/2025/feb/08/gambling-firms-secretly-shared-users-data-with-facebook-without-permission) | Meta accounts of those affected flooded with ads for casinos and betting sites|
|[From Dogecoin to $Trump: everything you need know about the wild world of meme coins.](https://www.theguardian.com/technology/2025/feb/09/from-dogecoin-to-trump-everything-you-need-know-about-the-wild-world-of-meme-coins) | Are they the same as crypto, why has the US president launched one, and who’s really coining it in? Here’s a complete guide to the latest digital money mania|
|[Google Maps changed the way we get around. It all began in a spare bedroom in Sydney.](https://www.theguardian.com/technology/2025/feb/09/google-maps-turns-20-anniversary-feature) |This weekend the mapping platform turns 20 – and Stephen Ma is writing himself and his friends back into its origin story |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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




































































































