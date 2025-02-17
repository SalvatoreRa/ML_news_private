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
|[Auditing Prompt Caching in Language Model APIs.](https://arxiv.org/abs/2502.07776) |Stanford researchers reveal that timing differences in LLM APIs can leak private user data through global prompt caching, posing serious security risks. Side-channel timing attacks occur when cached prompts complete faster, allowing attackers to infer others’ inputs. To detect this, they propose a statistical audit using hypothesis testing, uncovering global caching in major API providers. Additionally, timing variations expose architectural details, revealing decoder-only Transformer backbones and vulnerabilities in embedding models like OpenAI’s text-embedding-3-small. After responsible disclosure, some providers updated policies or disabled caching, with the recommended fix being per-user caching and transparent disclosures to prevent data leaks. |
|[Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning of Language Models.](https://arxiv.org/abs/2502.04404) |To enhance LLM reasoning robustness, researchers introduce self-backtracking, allowing models to revisit and revise flawed reasoning steps. Inspired by search algorithms, this method enables LLMs to identify errors mid-reasoning and backtrack to a previous step for a better approach. By training models with signals to trigger backtracking, they internalize an iterative search process instead of rigidly following a single Chain-of-Thought (CoT). This led to 40%+ improvements on reasoning benchmarks, as models self-correct mistakes mid-stream, producing more reliable solutions. The technique fosters autonomous, resilient reasoners, reducing overthinking loops and improving self-evaluation, moving LLMs closer to human-like reflective reasoning.|
|[Enhancing Reasoning to Adapt Large Language Models for Domain-Specific Applications.](https://arxiv.org/abs/2502.04384) | IBM researchers introduce SOLOMON, a neuro-inspired LLM reasoning architecture that enhances domain adaptability, demonstrated on semiconductor layout design. Standard LLMs struggle with spatial reasoning and domain application, but SOLOMON mitigates these issues using multi-agent oversight: multiple “Thought Generators” propose solutions, a “Thought Assessor” refines outputs, and a “Steering Subsystem” optimizes prompts. This design corrects hallucinations and arithmetic errors, outperforming GPT-4o, Claude-3.5, and Llama-3.1 in generating accurate GDSII layouts. SOLOMON excels at geometry-based tasks, reducing unit mismatches and scaling mistakes. Future work aims to stack SOLOMON layers, enhance text-image-code reasoning, and expand to broader engineering challenges, emphasizing advanced reasoning over mere model scaling.|
|[ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates.](https://arxiv.org/abs/2502.06772) |The ReasonFlux framework fine-tunes LLMs for complex reasoning using hierarchical thought processes and reusable templates. Instead of learning long Chain-of-Thought (CoT) solutions from scratch, it applies ~500 thought templates like problem splitting or solution verification. Hierarchical RL trains the model to sequence these templates, requiring only 8 GPUs for a 32B model. A novel inference-time adaptation adjusts reasoning depth dynamically, optimizing speed and accuracy. Achieving 91.2% on MATH (+6.7% over OpenAI’s model) and 56.7% on AIME, ReasonFlux shows that structured fine-tuning can rival brute-force scaling. |
|[LLM Pretraining with Continuous Concepts.](https://arxiv.org/abs/2502.08524) | CoCoMix is a pretraining framework that improves next-token prediction by incorporating continuous concepts learned from a sparse autoencoder. It boosts sample efficiency, surpassing traditional methods in language modeling and reasoning tasks. Furthermore, it increases interpretability by enabling direct inspection and modification of predicted concepts.|
|[90% faster B200 training.](https://www.together.ai/blog/nvidia-hgx-b200-with-together-kernel-collection) |Together AI showcases their significant progress in improving training kernels. They use TorchTitan as a testing platform and achieve substantial improvements by focusing on the architecture. |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Grok 3 is Set to Be Released on Monday.](https://www.forbes.com/sites/larsdaniel/2025/02/16/elon-musks-scary-smart-grok-3-release--what-you-need-to-know/) |xAI's Grok 3, trained with 200 million GPU-hours, features improved reasoning, self-correction, and training with synthetic data. It is scheduled for release on Monday. |
|[Anthropic and UK Government Sign AI Collaboration MOU.](https://www.anthropic.com/news/mou-uk-government) |Anthropic has teamed up with the UK government to investigate AI applications in public services, focusing on responsible deployment, economic growth, and scientific research through its Claude model. |
|[OpenAI tries to ‘uncensor’ ChatGPT.](https://techcrunch.com/2025/02/16/openai-tries-to-uncensor-chatgpt/) |OpenAI is changing how it trains AI models to explicitly embrace “intellectual freedom … no matter how challenging or controversial a topic may be,” the company says in a new policy. |
|[AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection.](https://arxiv.org/abs/2502.09254v1) |A new graph foundation model, AnomalyGFM, enhances zero- and few-shot anomaly detection by learning graph-agnostic representations, allowing for improved generalization across various datasets. |
|[Bolt.new introduces AI app generation for iOS and Android.](https://www.youtube.com/watch?v=iCwxkm2PkQE&ab_channel=Expo) |StackBlitz, known for its AI tool Bolt.new, has launched an AI mobile app developer in collaboration with Expo. Users can describe their app idea in natural language, and Bolt's AI will instantly generate code for full-stack iOS and Android apps. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction.](https://arxiv.org/abs/2502.07316) |CodeI/O enhances reasoning in large language models by transforming code into an input-output prediction format. This improves performance on various reasoning tasks by teaching universal reasoning principles without depending on code syntax. Additional refinement through multi-turn revisions increases accuracy by validating predictions. |
|[A Multiple Instance Learning Framework.](https://arxiv.org/abs/2502.08391v1) |A new multiple instance learning framework for whole slide image classification presents a dual-scale vision-language approach, utilizing a prototype-guided patch decoder and a context-guided text decoder to improve model performance on pathology tasks. |
|[Self contained FSDP implementation.](https://github.com/facebookresearch/capi/blob/main/fsdp.py) |A single 500 line implementation of data parallel that gets 48MFU. |
|[FinRL-DeepSeek - new trading AI agents combining Reinforcement Learning with Large Language Models.](https://melwy.com/finrl_deepseek) | Researchers combine reinforcement learning and large language models to improve risk-sensitive trading strategies, enhancing CPPO with LLM-generated risk assessments and trading recommendations, tested on Nasdaq-100 financial data.|
|[Google and Ireland Celebrate Insight AI Scholarship.](https://blog.google/around-the-globe/google-europe/taoiseach-visits-google-to-celebrate-the-future-of-irelands-tech-talent/) | Google hosts Irish officials to celebrate the Insight AI Scholarship, which supports students from underrepresented backgrounds in developing AI and digital skills.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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






































































































