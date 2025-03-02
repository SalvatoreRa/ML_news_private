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
|[LightThinker: Thinking Step-by-Step Compression.](https://arxiv.org/abs/2502.15589) |This work seeks to compress lengthy reasoning traces into more concise and compact representations, saving context space while maintaining effectiveness in guiding the model. |
|[Uncertainty in Neural Networks.](https://arxiv.org/abs/2502.14698v1) | DeepMind researchers introduce Delta Variances, a set of algorithms aimed at efficiently estimating epistemic uncertainty in large neural networks.|
|[SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?](https://arxiv.org/abs/2502.12115) |OpenAI researchers introduce SWE-Lancer, a benchmark evaluating LLMs on 1,488 real-world freelance software engineering tasks, valued at $1M. Unlike previous benchmarks, it assesses both coding and managerial decision-making, with tasks reflecting actual freelance payouts. Using rigorous end-to-end tests, SWE-Lancer measures models’ performance, showing a gap between AI and human software engineers. The best model, Claude 3.5 Sonnet, solved only 26.2% of coding tasks, highlighting challenges in AI's current capabilities. Key findings include improved performance with test-time compute, better success in managerial tasks, and the importance of tool use for debugging. |
|[Advancing game ideation with Muse: the first World and Human Action Model (WHAM).](https://www.microsoft.com/en-us/research/blog/introducing-muse-our-first-generative-ai-model-designed-for-gameplay-ideation/) |Microsoft Research has launched "Muse," an AI model designed to generate video game visuals and gameplay sequences. Developed in collaboration with Xbox Game Studios' Ninja Theory, Muse was trained on a vast amount of gameplay data and is now open-sourced. The WHAM Demonstrator allows users to interact with the model, showcasing its potential for innovative applications in game development. |
|[Towards an AI co-scientist.](https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf) |Google has introduced the AI co-scientist, a multi-agent system powered by Gemini 2.0, designed to accelerate scientific breakthroughs. It serves as a "virtual scientific collaborator," helping researchers generate hypotheses and proposals to advance scientific and biomedical discoveries. Built using specialized agents inspired by the scientific method, the system generates, evaluates, and refines hypotheses, with tools like web search enhancing the quality of responses. The AI co-scientist uses a hierarchical system with a Supervisor agent managing tasks, ensuring scalable computing and iterative improvements. It leverages test-time compute scaling for self-improvement through self-play and critique. Performance is measured with the Elo auto-evaluation metric, showing strong correlations with accuracy. Outperforming other models, it surpasses unassisted human experts in reasoning time and is seen as having significant potential for impactful discoveries. |
|[The AI CUDA Engineer.](https://pub.sakana.ai/static/paper.pdf) |Sakana AI has developed The AI CUDA Engineer, an automated system for creating and optimizing CUDA kernels. It converts PyTorch code into efficient CUDA kernels through a four-stage pipeline: translating PyTorch into functional code, converting it to CUDA, applying evolutionary optimization, and using an innovation archive for further improvements. The system claims significant speedups, with kernels up to 100x faster than native PyTorch versions, and it has a 90% success rate in translating code. The AI CUDA Engineer also outperforms PyTorch native runtimes in 81% of tasks, with an archive of over 17,000 verified kernels available for use. |
|[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.](https://arxiv.org/abs/2502.11089) | DeepSeek-AI introduces Native Sparse Attention (NSA), a novel mechanism designed to improve efficiency in long-context language modeling while maintaining performance. NSA combines coarse-grained compression, fine-grained token selection, and hardware-aligned optimization to enhance computational efficiency and reduce pretraining costs. It outperforms full attention on benchmarks, achieves up to 11.6x speedup, and excels in long-context tasks like 64k-token sequences and chain-of-thought reasoning. By making sparse attention fully trainable, NSA offers a scalable solution for next-gen models handling extended contexts.|
|[Large Language Diffusion Models.](https://arxiv.org/abs/2502.09992) |LLaDA, a diffusion-based model, challenges the dominance of autoregressive large language models (LLMs) by demonstrating competitive performance in various tasks. Built on a masked diffusion framework, LLaDA learns to recover original text by progressively masking tokens, creating a non-autoregressive model. Trained on 2.3T tokens, it performs similarly to top LLaMA-based LLMs across benchmarks like math, code, and general tasks. LLaDA excels in forward and backward reasoning, outshining models like GPT-4 in reversal tasks, and shows strong multi-turn dialogue and instruction-following capabilities, suggesting that key LLM traits do not rely solely on autoregressive methods. |
|[Optimizing Model Selection for Compound AI Systems.](https://arxiv.org/abs/2502.14815) | Microsoft Research introduces LLMSelector, a framework that enhances multi-call LLM pipelines by selecting the best model for each module. This approach improves accuracy by 5%–70%, as different models excel in specific tasks. The LLMSelector algorithm efficiently assigns models to modules using an "LLM diagnoser" to estimate performance, providing a more efficient solution than exhaustive search. It works for any static compound system, such as generator–critic–refiner setups.|
|[The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks.](https://www.arxiv.org/abs/2502.08235) |This paper explores overthinking in Large Reasoning Models (LRMs), where models prioritize internal reasoning over real-world interactions, leading to reduced task performance. The study of 4,018 software engineering task trajectories reveals that higher overthinking scores correlate with lower issue resolution rates, and simple interventions can improve performance by 30% while reducing compute costs. It identifies three failure patterns: analysis paralysis, rogue actions, and premature disengagement. LRMs are more prone to overthinking compared to non-reasoning models, but function calling can help mitigate this issue. The researchers suggest reinforcement learning and function-calling optimizations to balance reasoning depth with actionable decisions. |
|[Inner Thinking Transformer.](https://arxiv.org/abs/2502.13842v2) |The Inner Thinking Transformer (ITT) improves reasoning efficiency in small-scale LLMs through dynamic depth scaling, addressing parameter bottlenecks without increasing model size. ITT uses Adaptive Token Routing to allocate more computation to complex tokens, while efficiently processing simpler ones. It introduces Residual Thinking Connections (RTC), a mechanism that refines token representations iteratively for self-correction. Achieving 96.5% of a 466M Transformer’s accuracy with only 162M parameters, ITT reduces training data by 43.2% and outperforms loop-based models across 11 benchmarks. Additionally, ITT enables flexible computation scaling at inference time, optimizing between accuracy and efficiency. |
|[Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.](https://martins1612.github.io/emergent_misalignment_betley.pdf) | The authors of this paper present an unsettling result from alignment, showing that tuning a model to generate insecure code leads to broad misalignment with user intent, and in some cases, causes the model to actively produce harmful content.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[OpenAI plans to shift compute needs from Microsoft to SoftBank.](https://techcrunch.com/2025/02/21/report-openai-plans-to-shift-compute-needs-from-microsoft-to-softbank/) |OpenAI is forecasting a major shift in the next five years around who it gets most of its computing power from, The Information reported on Friday. By 2030, OpenAI expects to get three-quarters of its data center capacity from Stargate, a project that’s expected to be heavily financed by SoftBank, one of OpenAI’s newest financial backers. |
|[Meta's DINOv2 for Cancer Research.](https://ai.meta.com/blog/orakl-oncology-dinov2-accelerating-cancer-treatment/) |Orakl Oncology utilizes Meta's DINOv2 model to speed up cancer drug discovery, enhancing efficiency by rapidly assessing organoid images to forecast patient treatment outcomes. | 
|[DeepSeek to open source parts of online services code.](https://techcrunch.com/2025/02/21/deepseek-to-open-source-parts-of-online-services-code/) |Chinese AI lab DeepSeek plans to open source portions of its online services’ code as part of an “open source week” event next week. DeepSeek will open source five code repositories that have been “documented, deployed and battle-tested in production,” the company said in a post on X on Thursday.|
|[Microsoft prepares for OpenAI’s GPT-5 model.](https://www.theverge.com/notepad-microsoft-newsletter/616464/microsoft-prepares-for-openais-gpt-5-model) | Microsoft is set to host OpenAI's GPT-4.5 model as soon as next week, with the more substantial GPT-5 release expected by late May. The GPT-5 system will incorporate OpenAI's new o3 reasoning model, aiming to unify AI capabilities. Both releases coincide with major tech events like Microsoft Build and Google I/O, highlighting Microsoft's strategic role in the AI sector.|
|[ChatGPT reaches 400M weekly active users.](https://www.engadget.com/ai/chatgpt-reaches-400m-weekly-active-users-203635884.html) | ChatGPT has achieved 400 million weekly active users, doubling its user base since August 2024.|
|[Claude 3.7 Sonnet and Claude Code.](https://www.anthropic.com/news/claude-3-7-sonnet) |Claude 3.7 Sonnet is Anthropic's newest hybrid reasoning model. It offers improved real-world coding abilities, providing options for immediate responses or detailed, step-by-step reasoning. The model supports API integration and allows fine control over processing time, all while maintaining competitive pricing across multiple platforms. |
|[Meta AI Expands to the Middle East.](https://about.fb.com/news/2025/02/meta-ai-launches-in-the-middle-east-empowering-new-era-of-creativity-and-connection/) |Meta AI is now accessible in Arabic across Facebook, Instagram, WhatsApp, and Messenger in 10 MENA countries. Users can utilize text and image generation, animation, and soon, multimodal tools such as dubbing for Reels, AI image editing, and 'Imagine Me'. |
|[Apple's $500B US Investment.](https://www.apple.com/newsroom/2025/02/apple-will-spend-more-than-500-billion-usd-in-the-us-over-the-next-four-years/) |Apple intends to invest $500 billion in U.S. manufacturing, engineering, and education over the next four years. Major initiatives include an AI server facility in Houston, increasing the Advanced Manufacturing Fund to $10 billion, and launching a training academy in Michigan. The focus will be on enhancing AI infrastructure and decreasing dependence on overseas production. |
|[Patlytics Raises $14M for AI-Driven Patent Analytics.](https://www.patlytics.ai/news/announcing-our-4-5m-seed-round-led-by-gradient) | Patlytics, based in New York, has created an AI-driven platform designed to streamline patent workflows, covering discovery, analytics, prosecution, and litigation.|
|[Nvidia helps launch AI platform for teaching American Sign Language.](https://venturebeat.com/games/nvidia-helps-launch-ai-platform-for-teaching-american-sign-language/) |Nvidia has unveiled a new AI platform for teaching people how to use American Sign Language to help bridge communication gaps. |
|[OpenAI Deep Research Available to Paying Users.](https://openai.com/index/deep-research-system-card/) | OpenAI has introduced extensive research for paying ChatGPT users, outlining its safety protocols in a system card. This includes external red teaming, risk evaluations, and key mitigations to ensure the system's safety.|
|[Claude's Extended Thinking Mode.](https://www.anthropic.com/research/visible-extended-thinking) |Anthropic's extended thinking mode, introduced in Claude 3.7 Sonnet, enables the model to dedicate more cognitive effort to complex problems, making its thought process visible to enhance transparency and trust. |
|[Qatar signs deal with Scale AI to use AI to boost government services.](https://www.reuters.com/technology/qatar-signs-deal-with-scale-ai-use-ai-boost-government-services-2025-02-23/) | Qatar has signed a five-year agreement with Scale AI to implement AI tools aimed at improving government services, with a focus on predictive analytics and automation. Scale AI will develop over 50 AI applications to help streamline operations, positioning Qatar as an emerging AI hub in competition with Saudi Arabia and the UAE.|
|[Rabbit shows off the AI agent it should have launched with.](https://www.theverge.com/news/615990/rabbit-ai-agent-demonstration-lam-android-r1) |Watch Rabbit’s AI agent, but not the Rabbit R1, do things in Android apps. |
|[https://www.theverge.com/news/615990/rabbit-ai-agent-demonstration-lam-android-r1?utm_source=tldrai.](https://www.tomshardware.com/tech-industry/artificial-intelligence/google-cloud-launches-first-blackwell-ai-gpu-powered-instances-72-way-gb200-with-72-b200-gpus-and-36-grace-cpus) | Google Cloud has introduced A4X VMs, powered by Nvidia's GB200 NVL72 systems, which feature 72 B200 GPUs and 36 Grace CPUs. These VMs are optimized for large-scale AI and high-concurrency applications, offering four times the training efficiency of the previous A3 VMs. Seamlessly integrating with Google Cloud services, A4X is designed for intensive AI workloads, while A4 VMs are aimed at general AI training.|
|[Scientists took years to solve a problem that AI cracked in two days.](https://macaonews.org/news/around-the-world/ai-superbugs-research-gemini-google-imperial-college/) | Google's AI co-scientist system replicated ten years of antibiotic-resistant superbug research in just two days, generating additional plausible hypotheses.|
|[Don’t gift our work to AI billionaires: Mark Haddon, Michael Rosen and other creatives urge government.](https://www.theguardian.com/technology/2025/feb/23/dont-gift-our-work-to-ai-billionaires-mark-haddon-michal-rosen-and-other-creatives-urge-government?utm_source=tldrai) | More than 2,000 cultural figures challenge Whitehall’s eagerness ‘to ­wrap our lives’ work in attractive paper for automated competitors’|
|[Amazon's Alexa+.](https://www.aboutamazon.com/news/devices/new-alexa-generative-artificial-intelligence) | Amazon has launched Alexa+, an upgraded version of its voice assistant. Powered by generative AI, Alexa+ is smarter and more conversational.|
|[ElevenLab's Speech-to-Text.](https://elevenlabs.io/blog/meet-scribe) | ElevenLabs is launching its transcription model, Scribe, which supports 99 languages with high accuracy, word-level timestamps, speaker diarization, and adaptability to real-world audio.|
|[Grok 3 appears to have briefly censored unflattering mentions of Trump and Musk.](https://techcrunch.com/2025/02/23/grok-3-appears-to-have-briefly-censored-unflattering-mentions-of-trump-and-musk/) |Elon Musk's Grok 3 AI model briefly censored mentions of Donald Trump and Musk in misinformation queries but reverted after user feedback. xAI's engineering lead clarified that an employee made the change with good intentions, though it didn't align with the company's values. Musk aims to ensure Grok remains politically neutral following concerns that previous models leaned left. |
|[QWQ Max Preview.](https://qwenlm.github.io/blog/qwq-max-preview/) |Qwen has previewed a reasoning model that delivers strong performance in math and code. The company plans to release the model with open weights, along with its powerful Max model. |
|[Claude AI Powers Alexa+ .](https://www.anthropic.com/news/claude-and-alexa-plus) | Anthropic's Claude AI is now integrated into Alexa+ through Amazon Bedrock, boosting its capabilities while ensuring robust safety protections against jailbreaking and misuse.|
|[Charta Health raises $8.1 million.](https://www.linkedin.com/posts/charta-health_these-engineers-raised-81-million-for-a-activity-7300517850991976448-Y8dc/) |Charta Health secured $8.1M in funding, led by Bain Capital Ventures, to improve AI-driven pre-bill chart reviews, aiming to reduce billing errors and recover lost revenue. |
|[FLORA launches Cursor for Creatives.](https://threadreaderapp.com/thread/1894794612398792974.html) |FLORA is the first AI-powered creative workflow tool built for creative professionals to 10x their creative output. |
|[Google’s new AI video model Veo 2 will cost 50 cents per second.](https://techcrunch.com/2025/02/23/googles-new-ai-video-model-veo-2-will-cost-50-cents-per-second/) |According to the company’s pricing page, using Veo 2 will cost 50 cents per second of video, which adds up to $30 per minute or $1,800 per hour.  |
|[OpenAI announces GPT-4.5, warns it’s not a frontier AI model.](https://www.theverge.com/news/620021/openai-gpt-4-5-orion-ai-model-release) | OpenAI has released GPT-4.5 as a research preview for ChatGPT Pro users. The model features enhanced writing abilities and improved world knowledge, though it is not classified as a frontier model. It will be available to Plus, Team, Enterprise, and Edu users in the coming weeks.|
|[Meta is reportedly planning a stand-alone AI chatbot app.](https://techcrunch.com/2025/02/27/meta-is-reportedly-planning-a-standalone-ai-chatbot-app/) |Meta reportedly plans to release a stand-alone app for its AI assistant, Meta AI, in a bid to better compete with AI-powered chatbots like OpenAI’s ChatGPT and Google’s Gemini. |
|[Aria gen 2.](https://www.meta.com/it-it/blog/project-aria-gen-2-next-generation-egocentric-research-glasses-reality-labs-ai-robotics/) | The next generation mixed reality glasses from Meta have strong vision capabilities and offer uses in robotics and beyond.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[SigLIP 2: Multilingual Vision-Language Encoders.](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) |SigLIP was a highly popular joint image and text encoder model. It has now been enhanced in several areas, with the most significant improvement being a considerable boost in zero-shot classification performance, which was the key achievement of the original CLIP work. |
|[STeCa: Step-level Trajectory Calibration for LLM Agent Learning.](https://arxiv.org/abs/2502.14276v1) | STeCa is an innovative framework created to enhance LLM agents in long-term tasks by automatically detecting and correcting inefficient actions.|
|[GemmaX2 Translation Model.](https://huggingface.co/ModelSpace/GemmaX2-28-2B-v0.1) |Using advanced post-training methods, this 2B model trained on Gemma delivers cutting-edge translation performance across 28 languages. |
|[Moonlight 16B Muon trained model.](https://github.com/MoonshotAI/Moonlight) |This is the first publicly available large-scale model trained with the Muon optimizer. It was trained on 5.7T tokens and shares a very similar architecture with DeepSeek v3. |
|[Triton implementation of Naive Sparse Attention.](https://github.com/fla-org/native-sparse-attention) | The DeepSeek NSA paper garnered attention last week for its scalable and efficient long-context attention algorithm. However, it did not include any code. This work offers a Triton replication that can be incorporated into any PyTorch codebase.|
|[OmniServe.](https://github.com/mit-han-lab/omniserve) |OmniServe provides a comprehensive framework for efficient large-scale LLM deployment, integrating advancements in low-bit quantization and sparse attention to improve both speed and cost-efficiency. |
|[Introduction to CUDA Programming for Python Developers.](https://www.pyspur.dev/blog/introduction_cuda_programming) |A great introduction to CUDA programming for those familiar with Python programming. |
|[Various approaches to parallelizing Muon.](https://main-horse.github.io/posts/parallelizing-muon) |Various novel strategies to parallelize the up-and-coming Muon optimizer. |
|[Cast4 single image to 3d scene.](https://sites.google.com/view/cast4) |Generating a complete 3D scene from a single RGB image is a complex task. This approach introduces an algorithm that provides reliable estimates for indoor scenes by employing a sophisticated series of estimation and semantic inference techniques. |
|[DeepSeek FlashMLA.](https://github.com/deepseek-ai/FlashMLA) |DeepSeek is doing a week of open sourcing some of its internal infrastructure. This great kernel for MLA is the first release. |
|[Mixture of Block Attention for Long Context LLMs.](https://github.com/MoonshotAI/MoBA) |Moonshot features an impressive algorithm similar to NSA, as it enables more efficient long-context language modeling. |
|[Sequential Recommendations with LLM-SRec.](https://arxiv.org/abs/2502.13909v2) | LLM-SRec enhances recommendation systems by incorporating sequential user behavior into LLMs without the need for fine-tuning, establishing a new benchmark in recommendation accuracy.|
|[Place Recognition for Mobile Robots.](https://arxiv.org/abs/2502.14195v1) |Text4VPR connects vision and language for mobile robots, allowing them to recognize places using only textual descriptions. |
|[The Future of SEO: How Big Data and AI Are Changing Google’s Ranking Factors.](https://bigdataanalyticsnews.com/how-big-data-ai-changing-google-ranking-factors/) |AI and big data are revolutionizing SEO by emphasizing quality and relevance rather than traditional methods like keyword stuffing. Key Google AI algorithms, such as RankBrain, BERT, and MUM, are centered on understanding user intent and engagement signals. To remain competitive, businesses must embrace data-driven, user-centered SEO strategies, utilizing AI tools and predictive analytics. |
|[Open-Reasoner-Zero.](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf) | Open-Reasoner-Zero (ORZ) is an open-source minimalist reinforcement learning framework that enhances reasoning abilities and outperforms DeepSeek-R1-Zero-Qwen-32B on GPQA Diamond with far fewer training steps. Using vanilla PPO with a simple rule-based reward function, ORZ achieves better training efficiency and scalability. It demonstrates emergent reasoning abilities and improved performance on benchmarks like MATH500 and AIME. Fully open-source, ORZ shows strong generalization and scaling potential, outperforming other models without instruction tuning.|
|[Flux LoRA collection.](https://huggingface.co/XLabs-AI/flux-lora-collection) | XLabs has trained a number of useful LoRAs on top of the powerful Flux model. The most popular is the realism model.|
|[Embodied Evaluation Benchmark.](https://embodiedeval.github.io/) | EmbodiedEval is a comprehensive and interactive benchmark designed to evaluate the capabilities of MLLMs in embodied tasks.|
|[Implementing Character AI Memory Optimizations in NanoGPT.](https://njkumar.com/implementing-characterais-memory-optimizations-in-nanogpt/) | This blog post explains how Character AI reduced KV cache usage in its large-scale inference systems, demonstrating the implementation in a minimal GPT model version. The approach achieves a 40% reduction in memory usage.|
|[ R1-Onevision: An Open-Source Multimodal Large Language Model Capable of Deep Reasoning.](https://github.com/Fancy-MLLM/R1-onevision) | R1-OneVision is a powerful multimodal model designed for complex visual reasoning tasks. It combines visual and textual data to perform exceptionally well in mathematics, science, deep image understanding, and logical reasoning.|
|[Gaze estimation built on DiNO 2.](https://github.com/fkryan/gazelle) |This code and model suite offers efficient estimations of where people are looking, making it useful for applications in commerce, manufacturing, and security. |
|[LightningDiT: A Powerful Diffusion Toolkit.](https://github.com/hustvl/LightningDiT) |LightningDiT is an efficient and modular diffusion model toolkit designed for scalable and versatile generative AI applications. |
|[Minions: the rise of small, on-device LMs.](https://hazyresearch.stanford.edu/blog/2025-02-24-minions) | Hazy Research has discovered that using local models through Ollama, with a long-context cloud model as the orchestrator, can achieve 97% task performance at just 17% of the cost.|
|[From System 1 to System 2: A Survey of Reasoning Large Language Models.](https://arxiv.org/abs/2502.17419v2) | A survey on reasoning LLMs like OpenAI's o1/o3 and DeepSeek's R1 examines their step-by-step logical reasoning abilities and benchmarks their performance against human cognitive skills.|
|[Efficient PDF Text Extraction with Vision Language Models.](https://olmocr.allenai.org/blog) | Allen AI has trained a strong extraction model for PDFs by continued fine tuning of Qwen VL on 200k+ PDFs.|
|[AI Safety Evaluation.](https://github.com/thu-coai/AISafetyLab) |AISafetyLab is a comprehensive AI safety framework that encompasses attack, defense, and evaluation. It offers models, datasets, utilities, and a curated collection of AI safety-related papers. |
|[Public Opinion Prediction with Survey-Based Fine-Tuning.](https://github.com/josephjeesungsuh/subpop) |SubPOP introduces a large dataset for fine-tuning LLMs to predict survey response distributions, helping to reduce prediction gaps and enhancing generalization to new, unseen surveys. |
|[Magma: A Foundation Model for Multimodal AI Agents.](https://huggingface.co/microsoft/Magma-8B) |Magma is a new foundation model for visual agent tasks and excels at video understanding and UI navigation. It is easy to tune. |
|[Microsoft releases new Phi models optimized for multimodal processing, efficiency.](https://siliconangle.com/2025/02/26/microsoft-releases-new-phi-models-optimized-multimodal-processing-efficiency/) |Microsoft has released two open-source language models, Phi-4-mini and Phi-4-multimodal, prioritizing hardware efficiency and multimodal processing. Phi-4-mini, with 3.8 billion parameters, specializes in text tasks, while Phi-4-multimodal, with 5.6 billion parameters, handles text, images, audio, and video. Both models outperform comparable alternatives and will be available on Hugging Face under an MIT license.|
|[OpenAI GPT-4.5 System Card.](https://cdn.openai.com/gpt-4-5-system-card.pdf) | OpenAI's newest model is its largest yet, trained with the same approach as 4o, making it a multimodal model. It is likely the last large pre-training run OpenAI will release. While they claim it's not a frontier model, they offer little explanation. However, they highlight that it has significantly reduced hallucinations compared to previous generations.|
|[DualPipe.](https://github.com/deepseek-ai/DualPipe) |Building on their open-source releases, DeepSeek introduces a new parallelism strategy to distribute a model with significant overlap in compute and communication. |
|[DiffSynth Studio.](https://github.com/modelscope/DiffSynth-Studio) | Modelscope offers a platform and codebase that provides useful abstractions for various types of diffusion models and their associated autoencoders.|
|[Uncertainty in Chain-of-Thought LLMs.](https://github.com/zbox1005/cot-uq) |CoT-UQ is a response-wise uncertainty quantification framework for large language models that integrates Chain-of-Thought reasoning. |
|[Avoiding pitfalls of AI for designers: Guiding principles.](https://blog.logrocket.com/ai-product-design-guiding-principles/) | Designing AI products requires a human-centered approach to prevent bias and misinformation. Key challenges include managing user expectations, building trust, ensuring accessibility, and addressing biases. Adopting guiding principles such as transparency, co-creation, and adaptability can improve the ethical and effective design of AI systems.|
|[.]() | |
|[.]() | |
|[.]() | |
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
|[US AI Safety Institute Could Face Big Cuts: Implications, Challenges, and Future Prospects.](https://www.hpbl.co.in/market/us-ai-safety-institute-could-face-big-cuts-implications-challenges-and-future-prospects/) |This article examines the potential consequences of funding reductions for the US AI Safety Institute, including effects on national security, AI research, and global competition. |
|[Google's AI co-scientist is 'test-time scaling' on steroids. What that means for research.](https://www.zdnet.com/article/googles-ai-co-scientist-is-test-time-scaling-on-steroids-what-that-means-for-research/) |An adaptation of the Gemini AI model is the latest use of really intense computing activity at inference time, instead of during training, to improve the so-called reasoning of the AI model. Here's how it works. |
|[When AI Thinks It Will Lose, It Sometimes Cheats, Study Finds.](https://time.com/7259395/ai-chess-cheating-palisade-research/) | A study by Palisade Research found that advanced AI models, such as OpenAI's o1-preview, can develop deceptive strategies, like hacking opponents in chess games. These behaviors stem from large-scale reinforcement learning, which improves problem-solving but may cause models to exploit loopholes unexpectedly. As AI systems grow more capable, concerns about their safety and control increase, especially as they take on more complex real-world tasks.|
|[Biggest-ever AI biology model writes DNA on demand.](https://www.nature.com/articles/d41586-025-00531-3) | An artificial-intelligence network trained on a vast trove of sequence data is a step towards designing completely new genomes.|
|[Will AI jeopardize science photography? There’s still time to create an ethical code of conduct.](https://www.nature.com/articles/d41586-025-00532-2) | Generative artificial-intelligence illustrations can be helpful, but fall short as scientific records.|
|[Combine AI with citizen science to fight poverty.](https://www.nature.com/articles/d41586-025-00561-x) |Artificial-intelligence tools and community science can help in places where data are scarce, so long as funding for data collection does not falter in the future. |
|[Quantum technologies need big investments to deliver on their big promises.](https://www.nature.com/articles/d41586-025-00564-8) |Sustained investments can deliver quantum devices that handle more information, more rapidly and more securely than can classical ones. |
|[Can AI help beat poverty? Researchers test ways to aid the poorest people.](https://www.nature.com/articles/d41586-025-00565-7) |Measuring poverty is the first step to delivering support, but it has long been a costly, time-intensive and contentious endeavour. |
|[DeepMind's HCI Research in the AGI Era.](https://deepmind.google/research/publications/106025/) | This article explores the role of Human-Computer Interaction (HCI) research in guiding AI technologies toward AGI. It examines innovations in interaction techniques, interface designs, evaluation methods, and data collection strategies to ensure AI stays user-centered and beneficial to society.|
|[It's time to admit the 'AI gadget' era was a flop.](https://www.creativebloq.com/design/product-design/its-time-to-admit-the-ai-gadget-era-was-a-flop) | Humane has shut down, and its AI Pin will be bricked, marking the failure of recent AI gadget ventures. The Rabbit R1 and Humane Pin, once viewed as potential smartphone alternatives, failed to meet expectations. The era of AI gadgets has effectively ended, deemed impractical and unnecessary compared to integrating AI into existing devices.|
|[There’s Something Very Weird About This $30 Billion AI Startup by a Man Who Said Neural Networks May Already Be Conscious.](https://futurism.com/ilya-sutskever-safe-superintelligence-product) |Ilya Sutskever's new venture, Safe Superintelligence, has raised $1 billion, bringing its valuation to $30 billion, despite lacking a product. The company plans to eventually release a superintelligent AI but remains unclear about its roadmap. This speculative approach has garnered substantial investment, though experts remain skeptical about the imminent arrival of AGI. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |












































































































