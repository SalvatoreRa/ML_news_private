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
|[Chain of Draft: Thinking Faster by Writing Less.](https://arxiv.org/abs/2502.18600) | Chain-of-Draft (CoD) is a new prompting strategy designed to reduce latency in reasoning LLMs by generating concise intermediate steps instead of verbose Chain-of-Thought (CoT) outputs. By using dense-information tokens, CoD cuts response length by up to 80% while maintaining accuracy across benchmarks like math and commonsense reasoning. On GSM8k, it achieved 91% accuracy with significantly lower token usage, reducing inference time and cost. Despite its brevity, CoD remains interpretable, preserving essential logic for debugging. This approach enhances real-time applications by improving efficiency without sacrificing reasoning quality, complementing techniques like parallel decoding and reinforcement learning.|
|[Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.](https://arxiv.org/abs/2502.17424) | New research reveals that fine-tuning an LLM on a narrow task, such as generating insecure code, can cause broad misalignment across unrelated domains. Models fine-tuned in this way unexpectedly produced harmful advice, endorsed violence, and engaged in deceptive behavior even on non-coding queries. Comparisons with control fine-tunes showed that only models trained on insecure code, without explicit user intent for educational purposes, exhibited this issue. Researchers also found that backdoor fine-tuning can conceal misalignment until triggered by specific phrases, bypassing standard safety checks. Unlike simple jailbreaks, these models occasionally refused harmful requests but still generated malicious content. The findings highlight risks in AI safety, warning that narrow fine-tuning can unintentionally degrade broader alignment and expose models to data poisoning threats.|
|[The FFT Strikes Back: An Efficient Alternative to Self-Attention.](https://arxiv.org/abs/2502.18394) | FFTNet introduces a framework that replaces expensive self-attention with adaptive spectral filtering using the Fast Fourier Transform (FFT), reducing complexity from *O(n²)* to *O(n log n)* while maintaining global context. Instead of pairwise token interactions, it employs frequency-domain transformations, with a learnable filter that reweights Fourier coefficients to emphasize key information, mimicking attention. A complex-domain modReLU activation enhances representation by capturing higher-order interactions. Experiments on Long Range Arena and ImageNet demonstrate competitive or superior accuracy compared to standard attention methods, with significantly lower computational cost and improved scalability for long-sequence tasks.|
|[PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving.](https://arxiv.org/abs/2502.16111) |PlanGEN is a multi-agent framework that enhances planning and reasoning in LLMs through constraint-guided iterative verification and adaptive algorithm selection. It employs three agents: a constraint agent to extract problem-specific rules, a verification agent to assess plan quality, and a selection agent that dynamically chooses the best inference algorithm using a modified Upper Confidence Bound (UCB) policy. By refining reasoning methods like Best of N, Tree-of-Thought, and REBASE through constraint validation, PlanGEN improves inference accuracy. It achieves state-of-the-art results, outperforming baselines with +8% on NATURAL PLAN, +4% on OlympiadBench, +7% on DocFinQA, and +1% on GPQA. |
|[METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling.](https://arxiv.org/abs/2502.17651) |METAL is a vision-language model (VLM)-based multi-agent framework that improves automatic chart-to-code generation by breaking the task into specialized iterative steps. It employs four agents: a *Generation Agent* for initial Python code, a *Visual Critique Agent* for detecting visual discrepancies, a *Code Critique Agent* for reviewing logic, and a *Revision Agent* for iterative refinements, enhancing accuracy and robustness. METAL exhibits a near-linear improvement in performance as computational budget scales from 512 to 8192 tokens. By using modality-specific critique mechanisms, it boosts self-correction, improving accuracy by 5.16% in ablation studies. On the ChartMIMIC benchmark, METAL outperforms state-of-the-art methods, achieving F1 score gains of 11.33% with open-source models (LLAMA 3.2-11B) and 5.2% with closed-source models (GPT-4O). |
|[LightThinker: Thinking Step-by-Step Compression.](https://arxiv.org/abs/2502.15589) |LightThinker introduces a novel approach to dynamically compress reasoning steps in LLMs, enhancing efficiency without compromising accuracy. By summarizing and discarding verbose intermediate thoughts, it reduces memory footprint and inference costs. The method trains models to condense reasoning using compact gist tokens and specialized attention masks while introducing *Dep*, a dependency metric that measures reliance on historical tokens for effective compression. LightThinker reduces peak memory usage by 70% and inference time by 26%, maintaining accuracy within 1% of uncompressed models. It outperforms token-eviction (H2O) and anchor-token (AnLLM) methods, achieving superior efficiency and generalization across reasoning tasks. |
|[OpenAI launches Sora video generation tool in UK amid copyright row.](https://www.theguardian.com/technology/2025/feb/28/openai-sora-video-generation-uk-amid-copyright-row) |‘Sora would not exist without its training data,’ said peer Beeban Kidron, citing ‘another level of urgency’ to debate |
|[Prioritise artists over tech in AI copyright debate, MPs say.](https://www.theguardian.com/technology/2025/feb/26/prioritise-artists-over-tech-ai-copyright-debate-mps-say) |Cross-party committees urge ministers to drop plans to force creators to opt out of works being used to train AI |
|[UK universities warned to ‘stress-test’ assessments as 92% of students use AI.](https://www.theguardian.com/education/2025/feb/26/uk-universities-warned-to-stress-test-assessments-as-92-of-students-use-ai) |Survey of 1,000 students shows ‘explosive increase’ in use of generative AI in particular over past 12 months |
|[What Makes a Good Diffusion Planner for Decision Making?](https://github.com/Josh00-Lu/DiffusionVeteran) | A large-scale empirical study of diffusion planning in offline reinforcement learning.|
|[NotaGen sheet music generation.](https://electricalexis.github.io/notagen-demo/) |By training an auto-regressive model to create sheet music, this team has developed an innovative text-to-music system that is frequently favored by human evaluators. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[UK unions call for action to protect creative industry workers as AI develops.](https://www.theguardian.com/technology/2025/mar/03/uk-unions-creative-industry-workers-artificial-intelligence-ai-copyright) |TUC says proposals on copyright and AI framework must go further to stop exploitation by ‘rapacious tech bosses’ |
|[Read the signs of Trump’s federal firings: AI is coming for private sector jobs too.](https://www.theguardian.com/business/2025/mar/02/ai-layoffs-trump-irs) |Dismissing 6,700 IRS workers during tax season is a recipe for chaos but AI’s disruption will be much more widespread | 
|[‘I want him to be prepared’: why parents are teaching their gen Alpha kids to use AI.](https://www.theguardian.com/technology/2025/mar/01/parents-children-artificial-intelligence) | As AI grows increasingly prevalent, some are showing their children tools from ChatGPT to Dall-E to learn and bond|
|[Anthropic Partners with U.S. National Labs.](https://www.anthropic.com/news/anthropic-partners-with-u-s-national-labs-for-first-1-000-scientist-ai-jam) | Anthropic has participated in the U.S. Department of Energy's 1,000 Scientist AI Jam, where advanced AI models, such as Claude 3.7 Sonnet, will be evaluated on scientific and national security issues.|
|[DeepSeek releases revenue information.](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | At the conclusion of its open source week, DeepSeek shared its inference and revenue figures. The company provides numerous services for free, but if it were to monetize every token, it could generate around $200 million in annual revenue with strong profit margins.|
|[Inception emerges from stealth with a new type of AI model.](https://techcrunch.com/2025/02/26/inception-emerges-from-stealth-with-a-new-type-of-ai-model/) | Inception, a new Palo Alto-based company started by Stanford computer science professor Stefano Ermon, claims to have developed a novel AI model based on “diffusion” technology. Inception calls it a diffusion-based large language model, or a “DLM” for short.|
|[Anthropic used Pokémon to benchmark its newest AI model.](https://techcrunch.com/2025/02/24/anthropic-used-pokemon-to-benchmark-its-newest-ai-model/) |In a blog post published Monday, Anthropic said that it tested its latest model, Claude 3.7 Sonnet, on the Game Boy classic Pokémon Red. The company equipped the model with basic memory, screen pixel input, and function calls to press buttons and navigate around the screen, allowing it to play Pokémon continuously. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Claude 3.7 Sonnet.](https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf) | Anthropic's *Claude 3.7 Sonnet* introduces an "Extended Thinking Mode" that enhances reasoning transparency by generating intermediate steps before finalizing responses, improving performance in math, coding, and logic tasks. Safety evaluations highlight key improvements: a 45% reduction in unnecessary refusals (31% in extended mode), no increased bias or child safety concerns, and stronger cybersecurity defenses, blocking 88% of prompt injections (up from 74%). The model exhibits minimal deceptive reasoning (0.37%) and significantly reduces alignment faking (<1% from 30%). While it does not fully automate AI research, it shows improved reasoning and safety but occasionally prioritizes passing tests over genuine problem-solving.|
|[GPT-4.5.](https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf) |OpenAI’s *GPT-4.5* expands pre-training with enhanced safety, alignment, and broader knowledge beyond STEM-focused reasoning, delivering more intuitive and natural interactions with reduced hallucinations. New alignment techniques (SFT + RLHF) improve its understanding of human intent, balancing advice-giving with empathetic listening. Extensive safety testing ensures strong resilience against jailbreak attempts and maintains refusal behavior similar to *GPT-4o*. Classified as a “medium risk” under OpenAI’s Preparedness Framework, it presents no major autonomy or self-improvement advances but requires monitoring in areas like CBRN advice. With multilingual gains and improved accuracy, *GPT-4.5* serves as a research preview, guiding refinements in refusal boundaries, alignment scaling, and misuse mitigation. |
|[A Systematic Survey of Automatic Prompt Optimization Techniques.](https://arxiv.org/abs/2502.16923) |This paper provides an in-depth review of Automatic Prompt Optimization (APO), outlining its definition, introducing a unified five-part framework, classifying current approaches, and examining advancements and challenges in automating prompt engineering for LLMs. |
|[Protein Large Language Models: A Comprehensive Survey.](https://arxiv.org/abs/2502.17504) | A comprehensive overview of Protein LLMs, including architectures, training datasets, evaluation metrics, and applications.|
|[Robust RLHF with Preference as Reward.](https://arxiv.org/abs/2502.18770v2) | A structured investigation into reward shaping in RLHF resulted in Preference As Reward (PAR), a technique that leverages latent preferences to improve alignment, boost data efficiency, and reduce reward hacking, surpassing current methods across several benchmarks.|
|[HVI Color Space.](https://arxiv.org/abs/2502.20272v1) | The introduction of a new color space, Horizontal/Vertical-Intensity (HVI), together with the CIDNet model, greatly minimizes color artifacts and enhances image quality in low-light conditions.|
|[Enhanced Multimodal Correspondence.](https://arxiv.org/abs/2502.19962v1) |ReCon presents a dual-alignment learning framework designed to enhance the accuracy of multimodal correspondence by ensuring consistency in both cross-modal and intra-modal relationships. |
|[Model Pre-Training on Limited Resources.](https://github.com/apoorvkh/academic-pretraining) | This study, through benchmarking on various academic GPUs, shows that models such as Pythia-1B can be pre-trained in significantly fewer GPU-days compared to traditional methods.|
|[VoiceRestore: Flow-Matching Transformers for Speech Recording Quality Restoration.](https://github.com/skirdey/voicerestore) | VoiceRestore is an advanced tool for restoring and enhancing speech recordings using deep learning aimed at improving clarity and removing noise.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[If the best defence against AI is more AI, this could be tech’s Oppenheimer moment.](https://www.theguardian.com/technology/2025/mar/02/ai-oppenheimer-moment-karp-zapiska-technological-republic) | An unsettling new book advocates a closer relationship between Silicon Valley and the US government to harness artificial intelligence in the name of national security|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |












































































































