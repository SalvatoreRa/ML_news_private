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
|[Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.](https://arxiv.org/abs/2502.05171) |This work introduces a latent recurrent-depth transformer, a model that enhances reasoning efficiency at test time without generating additional tokens. Instead of increasing the context window or relying on Chain-of-Thought (CoT) fine-tuning, it enables iterative latent space reasoning, achieving performance comparable to a 50B parameter model with only 3.5B parameters. By unrolling a recurrent computation block at inference, the model deepens reasoning without modifying input sequences, reducing memory and compute costs while improving efficiency. Unlike CoT methods, it requires no specialized training, generalizing across reasoning tasks using standard pretraining data. Benchmarks show it scales like much larger models on tasks like ARC, GSM8K, and OpenBookQA, with emergent latent-space behaviors such as numerical task orbits and context-aware deliberation. This approach introduces test-time compute as a new scaling axis, hinting at future AI systems that reason in continuous latent space, unlocking new frontiers in efficiency and cognitive capabilities. |
|[Brain-to-Text Decoding: A Non-invasive Approach via Typing.](https://ai.meta.com/research/publications/brain-to-text-decoding-a-non-invasive-approach-via-typing/) |Meta AI’s Brain2Qwerty model translates brain activity into text by decoding non-invasive EEG/MEG signals while users type, marking a breakthrough in brain-computer interfaces (BCIs) without surgical implants. Using a deep learning pipeline, it combines convolutional feature extraction, a transformer for temporal modeling, and a character-level language model to refine predictions. MEG-based decoding achieved a 32% character error rate (CER)—a significant improvement over 67% with EEG—with the top participant reaching 19% CER, demonstrating rapid progress over previous non-invasive methods. This research paves the way for practical communication aids for paralyzed patients, though challenges remain in achieving real-time decoding and making MEG technology more portable. |
|[On the Emergence of Thinking in LLMs I: Searching for the Right Intuition.](https://arxiv.org/abs/2502.06773) | Researchers introduce Reinforcement Learning via Self-Play (RLSP) as a framework to train LLMs to "think" by generating and rewarding their own reasoning steps, mimicking algorithmic search. The three-phase training process starts with supervised fine-tuning, followed by exploration rewards to encourage diverse solutions, and concludes with an outcome verifier to ensure correctness. RLSP significantly boosts performance, with an 8B model improving MATH accuracy by 23% and a 32B model gaining 10% on Olympiad problems. Trained models exhibit emergent reasoning behaviors, such as backtracking and self-verification, suggesting that scaling this approach can enhance LLM problem-solving abilities.|
|[Competitive Programming with Large Reasoning Models.](https://arxiv.org/abs/2502.06807) |OpenAI’s latest study compares a specialized coding AI to a scaled-up general model on competitive programming tasks, highlighting the trade-off between efficiency and specialization. A tailored model (o1-ioi) with hand-crafted coding strategies performed decently (~50th percentile at IOI 2024), but a larger, general-purpose model (o3) achieved gold medal-level performance without domain-specific tricks. Both improved with reinforcement learning (RL) fine-tuning, yet the scaled model matched elite human coders on platforms like Codeforces, outperforming the expert-designed system. The findings suggest that scaling up a broadly trained transformer can surpass manual optimizations, reinforcing the trend of "scale over specialization" in AI model design for complex reasoning tasks like programming. |
|[Training Language Models to Reason Efficiently.](https://arxiv.org/abs/2502.04463) |A new RL approach trains large reasoning models to allocate compute efficiently, adjusting Chain-of-Thought (CoT) length based on problem difficulty. Easy queries get short reasoning, while complex ones get deeper thought, optimizing speed vs. accuracy. The model, rewarded for solving tasks with minimal steps, learns to avoid “overthinking” while maintaining performance. This method cuts inference costs while ensuring high accuracy, making LLM deployment more efficient. Acting as both “thinker” and “controller,” the model self-optimizes reasoning, mimicking expert decision-making on when to stop analyzing. |
|[LM2: Large Memory Models.](https://arxiv.org/abs/2502.06049) | Large Memory Models (LM2) enhance transformer architectures with an external memory module, enabling superior long-term reasoning and handling of extended contexts. By integrating a memory-augmented design, LM2 reads and writes information across multiple reasoning steps via cross-attention, excelling in multi-hop inference, numeric reasoning, and long-document QA. On the BABILong benchmark, it outperformed prior models by 37% and exceeded a baseline Llama model by 86%, all while maintaining strong general language abilities, including a +5% boost on MMLU knowledge tests. This approach aligns AI reasoning with complex tasks, ensuring better adherence to objectives in long dialogues and structured argumentation, marking a step toward more capable and aligned AI systems.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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






































































































