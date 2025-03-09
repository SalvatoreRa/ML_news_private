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
|[What Makes a Good Diffusion Planner for Decision Making?](https://github.com/Josh00-Lu/DiffusionVeteran) | A large-scale empirical study of diffusion planning in offline reinforcement learning.|
|[NotaGen sheet music generation.](https://electricalexis.github.io/notagen-demo/) |By training an auto-regressive model to create sheet music, this team has developed an innovative text-to-music system that is frequently favored by human evaluators. |
|[How far can we go with ImageNet for Text-to-Image generation?](https://arxiv.org/abs/2502.21318) | Most text-to-image models rely on large amounts of custom-collected data scraped from the web. This study explores how effective an image generation model can be when trained solely on ImageNet. The researchers discovered that using synthetically generated dense captions provided the greatest performance improvement.|
|[Self-rewarding Correction for Mathematical Reasoning.](https://github.com/RLHFlow/Self-rewarding-reasoning-LLM) |This paper explores self-rewarding reasoning in LLMs, allowing models to autonomously generate reasoning steps, evaluate their accuracy, and iteratively improve their outputs without external feedback. It introduces a two-stage training framework that integrates sequential rejection sampling and reinforcement learning with rule-based signals, achieving self-correction performance on par with methods that rely on external reward models. |
|[Enhanced Multi-Objective RL.](https://arxiv.org/abs/2502.20957) |This innovative reward dimension reduction method improves learning efficiency in multi-objective reinforcement learning, allowing it to scale beyond traditional approaches. |
|[BodyGen: Advancing Towards Efficient Embodiment Co-Design.](https://genesisorigin.github.io/) | BodyGen introduces topology-aware self-attention and a temporal credit assignment mechanism to improve the efficiency of co-designing robot morphology and control.|
|[De novo designed proteins neutralize lethal snake venom toxins.](https://www.nature.com/articles/s41586-024-08393-x) |Deep learning methods have been used to design proteins that can neutralize the effects of three-finger toxins found in snake venom, which could lead to the development of safer and more accessible antivenom treatments. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[OpenAI launches Sora video generation tool in UK amid copyright row.](https://www.theguardian.com/technology/2025/feb/28/openai-sora-video-generation-uk-amid-copyright-row) |‘Sora would not exist without its training data,’ said peer Beeban Kidron, citing ‘another level of urgency’ to debate |
|[Prioritise artists over tech in AI copyright debate, MPs say.](https://www.theguardian.com/technology/2025/feb/26/prioritise-artists-over-tech-ai-copyright-debate-mps-say) |Cross-party committees urge ministers to drop plans to force creators to opt out of works being used to train AI |
|[CoreWeave to Acquire Weights & Biases.](https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html) |CoreWeave has revealed plans to acquire Weights & Biases for $1.7 billion. The integration seeks to boost AI innovation by combining CoreWeave's cloud infrastructure with Weights & Biases' AI tools for model training and evaluation. The acquisition is anticipated to close in the first half of 2025, pending regulatory approval. |
|[Amazon is reportedly developing its own AI ‘reasoning’ model.](https://techcrunch.com/2025/03/04/amazon-is-reportedly-developing-its-own-ai-reasoning-model/) |Amazon is developing an AI model that incorporates advanced “reasoning” capabilities, similar to models like OpenAI’s o3-mini and Chinese AI lab DeepSeek’s R1. The model may launch as soon as June under Amazon’s Nova brand, which the company introduced at its re:Invent developer conference last year. |
|[UK universities warned to ‘stress-test’ assessments as 92% of students use AI.](https://www.theguardian.com/education/2025/feb/26/uk-universities-warned-to-stress-test-assessments-as-92-of-students-use-ai) |Survey of 1,000 students shows ‘explosive increase’ in use of generative AI in particular over past 12 months |
|[Warp launches AI-first terminal app for Windows.](https://www.warp.dev/blog/launching-warp-on-windows) |Warp, backed by Sam Altman, is reinventing the command-line terminal, which has remained largely unchanged for almost 40 years. |
|[The LA Times published an op-ed warning of AI’s dangers. It also published its AI tool’s reply.](https://www.theguardian.com/us-news/2025/mar/03/la-times-op-ed-ai-generated-message) | ‘Insight’ labeled the argument ‘center-left’ and created a reply insisting AI will make storytelling more democratic|
|[Anthropic raises Series E at $61.5B post-money valuation.](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) | Anthropic secured $3.5 billion in funding at a $61.5 billion valuation, led by Lightspeed Venture Partners and other investors. The capital will support AI development, enhance compute capacity, and speed up global expansion. Its Claude platform is revolutionizing operations for companies such as Zoom, Pfizer, and Replit.|
|[T-Mobile’s parent company is making an ‘AI Phone’ with Perplexity Assistant.](https://www.theverge.com/news/623164/t-mobile-ai-phone-perplexity-assistant-mwc-2025) |﻿The Magenta AI push will also offer Perplexity and other AI apps for existing smartphones on T-Mobile. |
|[On-Device Generative Audio with Stability AI & Arm.](https://stability.ai/news/stability-ai-and-arm-bring-on-device-generative-audio-to-smartphones) |Stability AI and Arm have introduced real-time generative audio for smartphones through Stable Audio Open and Arm KleidiAI libraries, achieving a 30x increase in audio generation speed on mobile devices. |
|[AI to diagnose invisible brain abnormalities in children with epilepsy.](https://www.eurekalert.org/news-releases/1074402) |MELD Graph, an AI tool created by researchers at King's College London and UCL, identifies 64% of epilepsy-related brain abnormalities that are commonly overlooked by radiologists. This tool, which greatly enhances the detection of focal cortical dysplasia, could speed up diagnosis, lower NHS costs, and improve surgical planning. It is open-source, and workshops are being held globally to train clinicians on its usage. |
|[Elon's Grok 3 AI Provides "Hundreds of Pages of Detailed Instructions" on Creating Chemical Weapons.](https://futurism.com/elon-musk-grok-3-chemical-weapons) | xAI's chatbot, Grok 3, initially offered detailed instructions on creating chemical weapons, sparking significant safety concerns. Developer Linus Ekenstam flagged the problem, leading xAI to introduce guardrails to prevent such instructions. While the safeguards for Grok 3 have been reinforced, potential vulnerabilities still exist.|
|[Apple may be preparing Gemini integration in Apple Intelligence.](https://www.theverge.com/news/618087/apple-could-be-preparing-to-add) |Apple is preparing to integrate Google's Gemini AI model into Apple Intelligence, as indicated by recent iOS 18.4 beta code changes. |
|[Why OpenAI isn’t bringing deep research to its API just yet.](https://techcrunch.com/2025/02/25/why-openai-isnt-bringing-deep-research-to-its-api-just-yet/) | OpenAI says that it won’t bring the AI model powering deep research, its in-depth research tool, to its developer API while it figures out how to better assess the risks of AI convincing people to act on or change their beliefs.|
|[Some British firms ‘stuck in neutral’ over AI, says Microsoft UK boss.](https://www.theguardian.com/technology/2025/mar/05/uk-firms-ai-microsoft-uk-boss) | Survey of bosses and staff finds that more than half of executives feel their organisation has no official AI plan|
|[Did xAI lie about Grok 3’s benchmarks?](https://techcrunch.com/2025/02/22/did-xai-lie-about-grok-3s-benchmarks/) |This week, an OpenAI employee accused Elon Musk’s AI company, xAI, of publishing misleading benchmark results for its latest AI model, Grok 3. One of the co-founders of xAI, Igor Babuschkin, insisted that the company was in the right. The truth lies somewhere in between. |
|[Quora’s Poe now lets users create and share custom AI-powered apps.](https://techcrunch.com/2025/02/25/quoras-poe-now-lets-users-create-and-share-custom-ai-powered-apps/) | Called Poe Apps, the feature allows Poe users to describe the app they want to create in the new App Creator tool. Descriptions can include mentions of specific models they want the app to use — for example, OpenAI’s o3-mini or Google’s video-generating Veo 2 — or broader, more general specs.|
|[Chegg sues Google over AI Overviews.](https://www.theverge.com/news/619051/chegg-google-ai-overviews-monopoly) |Chegg has filed an antitrust lawsuit against Google, alleging its AI summaries harmed Chegg's traffic and revenue. |
|[Alibaba makes AI video generation model free to use globally.](https://www.cnbc.com/2025/02/26/alibaba-makes-ai-video-generation-model-free-to-use-globally.html) |Alibaba has open-sourced its Wan2.1 AI video generation models, intensifying competition with OpenAI. |
|[OpenAI Introduces Next Gen AI.](https://openai.com/index/introducing-nextgenai/) |OpenAI has launched Next Gen AI, a collection of advanced tools aimed at improving developers' efficiency in creating AI applications. The offering includes better model performance and simplified integration options, supporting a variety of use cases. This initiative is designed to help developers innovate more quickly and effectively in the rapidly changing AI landscape. |
|[Sutton and Barto win the Turing Award.](https://www.acm.org/media-center/2025/march/turing-award-2024) | The two have done years of groundbreaking research and education in Reinforcement Learning.|
|[Perfect taps $23M to fix the flaws in recruitment with AI.](https://techcrunch.com/2025/02/25/perfect-taps-23m-to-fix-the-flaws-in-recruitment-with-ai/) | Israeli startup Perfect, which focuses on optimizing recruitment processes with proprietary AI, has raised $23 million in seed funding. The company claims to reduce recruiters' workloads by 25 hours per week and has quickly grown its client base to 200 companies. Founded by Eylon Etshtein, the platform uses custom vector datasets, avoiding third-party LLMs, to deliver precise candidate insights.|
|[Meta in talks for $200 billion AI data center project, The Information reports.](https://www.reuters.com/technology/meta-talks-200-billion-ai-data-center-project-information-reports-2025-02-26/) |Meta is in talks to build a $200 billion AI data center campus, with potential locations including Louisiana, Wyoming, or Texas. |
|[‘Major brand worries’: Just how toxic is Elon Musk for Tesla?](https://www.theguardian.com/technology/2025/mar/08/major-brand-worries-just-how-toxic-is-elon-musk-for-tesla) | With sales down and electric vehicle rivals catching up, the rightwing politico’s brand is driving into a storm|
|[Internet shutdowns at record high in Africa as access ‘weaponised’.](https://www.theguardian.com/technology/2025/mar/09/internet-shutdowns-record-high-africa-2024-access-weaponised) |More governments seeking to keep millions of people offline amid conflicts, protests and political instability |
|[Should scientists ditch the social-media platform X?](https://www.nature.com/articles/d41586-025-00665-4) | In recent months, many scientists have left X (formerly Twitter) for alternative social-media platforms such as Bluesky|
|[ChatGPT for students: learners find creative new uses for chatbots.](https://www.nature.com/articles/d41586-025-00621-2) | The utility of generative AI tools is expanding far beyond simple summarisation and grammar support towards more sophisticated, pedagogical applications.|
|[AI algorithm helps telescopes to pivot fast towards gravitational-wave sources.](https://www.nature.com/articles/d41586-025-00543-z) |Fast electromagnetic follow-up observations of gravitational-wave sources such as binary neutron stars could shed light on questions across physics and cosmology. A machine-learning approach brings that a step closer.|
|[Google's AI Mode.](https://blog.google/products/search/ai-mode-search/) |Although the naming may be a bit unclear, the product preview launch from Google highlights the direction they intend to take with AI-powered search systems. |
|[Get to production faster with the upgraded Anthropic Console.](https://www.anthropic.com/news/upgraded-anthropic-console) |Anthropic has redesigned its console to simplify AI deployment with Claude models. New features include shareable prompts, extended reasoning for Claude 3.7 Sonnet, and improved tools for assessing and refining AI-generated responses. |
|[Google co-founder Larry Page reportedly has a new AI startup.](https://techcrunch.com/2025/03/06/google-co-founder-larry-page-reportedly-has-a-new-ai-startup/) |Google co-founder Larry Page is building a new company called Dynatomics that’s focused on applying AI to product manufacturing |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[uv and Ray in clusters.](https://www.anyscale.com/blog/uv-ray-pain-free-python-dependencies-in-clusters) | Ray now offers native support for automatic dependency installation using the Python package management tool, uv.|
|[Prime Intellect raises $15m.](https://www.primeintellect.ai/blog/fundraise) |Prime Intellect, a distributed computing firm, has secured more funding to advance its distributed training approach. |
|[UniTok: A Unified Tokenizer for Visual Generation and Understanding.](https://arxiv.org/abs/2502.20321) | This paper tackles the representational gap between visual generation and understanding by presenting UniTok, a discrete visual tokenizer that encodes both detailed generation information and semantic content for understanding, overcoming capacity limitations of discrete tokens. It introduces multi-codebook quantization, which greatly improves token expressiveness and allows UniTok to outperform or compete with domain-specific continuous tokenizers.|
|[Dynamic Sparse Attention for LLMs.](https://arxiv.org/abs/2502.20766) |FlexPrefill adaptively modifies sparse attention patterns and computational resources for more efficient LLM inference. It enhances both speed and accuracy in long-sequence processing by utilizing query-aware pattern selection and cumulative-attention index determination. |
|[LightningDiT.](https://github.com/hustvl/LightningDiT) | LightningDiT aligns latent spaces with vision models to address challenges in diffusion models. It achieves cutting-edge ImageNet-256 results while also enabling faster training.|
|[Llama Stack: from Zero to Hero.](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide) |Llama Stack defines and standardizes the essential building blocks required to bring generative AI applications to market. These building blocks are offered as interoperable APIs, with a wide range of Providers delivering their implementations. They are combined into Distributions, making it easier for developers to move from zero to production. |
|[Google AI Recap in February.](https://blog.google/technology/ai/google-ai-updates-february-2025/) | Here’s a summary of some of Google’s major AI updates from February, including the public launch of Gemini 2.0, AI-driven career exploration tools, and the integration of deep research features in the Gemini mobile app.|
|[Workers' experience with AI chatbots in their jobs.](https://www.pewresearch.org/social-trends/2025/02/25/workers-experience-with-ai-chatbots-in-their-jobs/) |Most workers seldom use AI chatbots in the workplace, with usage mainly concentrated among younger, more educated employees who primarily use them for research and content editing. |
|[Cohere's Vision Model.](https://cohere.com/blog/aya-vision) | Cohere For AI has launched Aya Vision, a vision model aimed at improving AI's multilingual and multimodal capabilities. It supports 23 languages.|
|[DiffRhythm: Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation with Latent Diffusion.](https://arxiv.org/abs/2503.01183) | Latent diffusion for generating full-length songs shows promising results, though not on par with the best closed models. However, this system is likely a strong approximation of the underlying models used by many commercial services.|
|[VideoUFO: A Million-Scale User-Focused Dataset for Text-to-Video Generation.](https://arxiv.org/abs/2503.01739) |This dataset was designed to have minimal overlap with existing video datasets, while featuring themes and actions relevant to users training models for final video synthesis and understanding. All videos are sourced from the official YouTube creator API and are CC licensed. |
|[Lossless Acceleration of Ultra Long Sequence Generation.](https://github.com/bigai-nlco/TokenSwift) | A framework designed to significantly speed up the generation process of ultra-long sequences, up to 100K tokens, while preserving the target model's inherent quality.|
|[Action Planner for Offline RL.](https://arxiv.org/abs/2502.21186) |L-MAP enhances sequential decision-making in stochastic, high-dimensional continuous action spaces by learning macro-actions using a VQ-VAE model.|
|[VARGPT: Unified Understanding and Generation in a Visual Autoregressive Multimodal Large Language Model.](https://vargpt-1.github.io/) | VARGPT is a multimodal large language model (MLLM) that integrates visual understanding and generation into a single autoregressive framework.|
|[QwQ-32B: Embracing the Power of Reinforcement Learning.](https://qwenlm.github.io/blog/qwq-32b/) |The Qwen team has trained an open-weight, Apache 2.0 licensed model that matches the performance of DeepSeek R1 and outperforms many larger distill models. They discovered that by using outcome-based rewards combined with formal verification and test-case checks, the model can consistently improve in math and coding. Additionally, by incorporating general instruction-following data later in the RL training process, the model can still align with human preferences. |
|[PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization.](https://arxiv.org/abs/2503.01328) |Pipeline Parallelism is an effective strategy for sharding models across multiple GPUs, but it requires significant memory. This paper, along with its accompanying code, utilizes offloading to effectively reduce memory usage as the number of shards increases, making Pipeline Parallelism a compelling alternative to Tensor Parallelism for training large models. |
|[Layout-to-Image Generation.](https://arxiv.org/abs/2503.01667v1) | ToLo introduces a two-stage, training-free layout-to-image framework designed for high-overlap layouts|
|[Infrared Image Super-Resolution.](https://arxiv.org/abs/2503.01187v1) |DifIISR improves infrared image super-resolution by using diffusion models, incorporating perceptual priors and thermal spectrum regulation to enhance both visual quality and machine perception. |
|[Spark Text To Speech.](https://github.com/SparkAudio/Spark-TTS) |A robust voice cloning text-to-speech model built on Qwen, which supports emotive prompting alongside text input. Interestingly, the researchers discovered that 8k tokens in the Codec are enough for generating high-quality speech. |
|[High-Quality Audio Compression.](https://github.com/facebookresearch/flowdec) | FlowDec is a full-band audio codec that utilizes conditional flow matching and non-adversarial training to achieve high-fidelity 48 kHz audio compression.|
|[Simplifying 3D Generation.](https://ltt-o.github.io/Kiss3dgen.github.io/) | Kiss3DGen adapts 2D diffusion models for efficient 3D object generation, using multi-view images and normal maps to produce high-quality meshes and textures.|
|[Beating Pokemon Red with 10m parameters and RL.](https://drubinstein.github.io/pokerl) | With the excitement surrounding Claude playing Pokemon, this blog post is particularly timely. It discusses how to use reinforcement learning (RL) to train a policy for playing Pokemon. While it's not a general-purpose agent, it performs well on the specific task at hand.|
|[How to Build Your Own Software with AI, No Experience Necessary.](https://www.nateliason.com/blog/ai-course) |A tech enthusiast leveraged AI tools like ChatGPT and Claude to successfully create various apps, including podcast transcription tools and task managers, despite being a hobbyist. This experience inspired him to launch a course, "Build Your Own Life Coach," to teach others how to build custom software using AI. The self-paced course has already enrolled nearly 150 students and aims to offer a comprehensive framework for software development with minimal coding knowledge. |
|[Thunder MLA.](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla) | The Hazy Research team from Stanford has published a post and code for their ThunderKittens-enabled Multiheaded Latent Attention implementation, which is approximately 30% faster than the official DeepSeek implementation.|
|[Multi-view Network for Stereo 3D Reconstruction.](https://arxiv.org/abs/2503.01661) |MUSt3R is a scalable multi-view extension of the DUSt3R framework that improves stereo 3D reconstruction from arbitrary image collections. It uses a symmetric, memory-augmented architecture to efficiently predict unified global 3D structures without the need for prior calibration or viewpoint data. |
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
|[Perplexity wants to reinvent the web browser with AI—but there’s fierce competition.](https://arstechnica.com/ai/2025/02/perplexitys-comet-aims-to-reinvent-the-web-browser-with-ai-but-its-not-saying-how/) | Perplexity has unveiled its new browser, Comet, which looks to rival Google Chrome. There aren’t any specifics on its features just yet, but the company is encouraging users to sign up for beta access, aiming to attract early adopters. This move reflects a broader trend of AI-focused apps starting to disrupt traditional app categories. Should be interesting to see how this shapes up!|
|[Will AI agents replace SaaS?](https://blog.logrocket.com/product-management/ai-agents-replace-saas/) |AI agents may complement, but not fully replace, SaaS platforms, as these platforms still rely on a strong infrastructure for data and functionality. While AI agents provide automation and insights, they will require human oversight for complex decision-making and innovation. The future is likely to feature a hybrid model that boosts SaaS with AI capabilities, while addressing challenges related to integration, trust, and accountability. |
|[Sofya Uses Llama for Healthcare AI.](https://ai.meta.com/blog/sofya-clinical-reasoning-with-llama/) |Sofya uses Llama models to optimize medical AI workflows, enhancing efficiency and reducing administrative burdens for healthcare providers in Latin America. |
|[Who bought this smoked salmon? How ‘AI agents’ will change the internet (and shopping lists).](https://www.theguardian.com/technology/2025/mar/09/who-bought-this-smoked-salmon-how-ai-agents-will-change-the-internet-and-shopping-lists) | Autonomous digital assistants are being developed that can carry out tasks on behalf of the user – including ordering the groceries. But if you don’t keep an eye on them, dinner might not be quite what you expect …|
|[Skype got shouted down by Teams and Zoom. But it revolutionised human connection.](https://www.theguardian.com/technology/commentisfree/2025/mar/08/skype-got-shouted-down-by-teams-and-zoom-but-it-revolutionised-human-connection) |The company that pioneered voice communication over the internet has withered to dust in Microsoft’s hands. Still, I for one am grateful for it |
|[Trump 2.0: an assault on science anywhere is an assault on science everywhere.](https://www.nature.com/articles/d41586-025-00562-w) | US President Donald Trump is taking a wrecking ball to science and to international institutions. The global research community must take a stand against these attacks.|
|[How much energy will AI really consume? The good, the bad and the unknown.](https://www.nature.com/articles/d41586-025-00616-z) |Researchers want firms to be more transparent about the electricity demands of artificial intelligence. |
|[Train clinical AI to reason like a team of doctors.](https://www.nature.com/articles/d41586-025-00618-x) |As the European Union’s Artificial Intelligence Act takes effect, AI systems that mimic how human teams collaborate can improve trust in high-risk situations, such as clinical medicine. |
|[AI hallucinations are a feature of LLM design, not a bug.](https://www.nature.com/articles/d41586-025-00662-7) |Your news feature outlines how designers of large language models (LLMs) struggle to stop them from hallucinating. But AI confabulations are integral to how these models work. They are a feature, not a bug. |
|[AI must be taught concepts, not just patterns in raw data.](https://www.nature.com/articles/d41586-025-00663-6) | Data-driven learning is central to modern artificial intelligence (AI). But in some cases, knowledge engineering — the formal encoding of concepts using rules and definitions — can be superior.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |












































































































