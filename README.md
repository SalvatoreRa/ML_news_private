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
|[Visual RFT.](https://arxiv.org/abs/2503.01785) | One of the trends is the use of simple verifiable rewards and scaled reinforcement learning. This paper successfully applies that approach to vision-language models.|
|[Stability AI hires Unity's Ryan Ellis as SVP, Head of Product.](https://stability.ai/news/introducing-our-new-svp-head-of-product-ryan-ellis) | Ryan Ellis, formerly with Unity, has joined Stability AI to lead product development, bringing his expertise in real-time 3D engines and AI-driven content creation.|
|[Podcasting platform Podcastle launches a text-to-speech model with more than 450 AI voices.](https://techcrunch.com/2025/03/03/podcasting-platform-podcastle-launches-a-text-to-speech-model-with-more-than-450-ai-voices/) |Podcastle has released Asyncflow v1.0, an AI text-to-speech model featuring more than 450 AI voices and affordable training options. |
|[Generalized discrete diffusion.](https://www.arxiv.org/abs/2503.04482) | This work expands diffusion on discrete data, like text, by introducing a generalized denoising process and a slightly enhanced masking scheme. This combination improves training efficiency and enables the model to correct its own output.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Stability AI Secures Investment for AI-Driven Content.](https://stability.ai/news/stability-ai-announces-investment-from-wpp-and-new-partnership-to-shape-the-future-of-media-and-entertainment-production) | Stability AI has revealed a strategic collaboration and investment from WPP, with the goal of incorporating generative AI into advertising and media creation.|
|[The US Army Is Using ‘CamoGPT' to Purge DEI From Training Materials.](https://www.wired.com/story/the-us-army-is-using-camogpt-to-purge-dei-from-training-materials/) |The US Army's TRADOC is utilizing an AI tool, CamoGPT, to detect and remove DEIA references from training materials in accordance with an executive order from President Trump. Developed by the Army's AI Integration Center, CamoGPT scans documents for certain keywords and has roughly 4,000 users. This initiative is part of a broader government effort to remove DEIA content, using AI to improve efficiency while aligning with national security goals. | 
|[OpenAI’s ex-policy lead criticizes the company for ‘rewriting’ its AI safety history.](https://techcrunch.com/2025/03/06/openais-ex-policy-lead-criticizes-the-company-for-rewriting-its-ai-safety-history/) |A high-profile ex-OpenAI policy researcher, Miles Brundage, took to social media on Wednesday to criticize OpenAI for “rewriting the history” of its deployment approach to potentially risky AI systems. |
|[CoreWeave signs $11.9 billion contract with OpenAI ahead of IPO.](https://www.investing.com/news/stock-market-news/coreweave-signs-119-billion-contract-with-openai-ahead-of-ipo-93CH-3918610) | CoreWeave has secured a five-year, $11.9 billion cloud computing agreement with OpenAI ahead of its IPO, with OpenAI set to acquire a stake in the Nvidia-supported AI startup.|
|[Microsoft appears to be working on 3D gaming experiences for Copilot.](https://techcrunch.com/2025/03/10/microsoft-appears-to-be-working-on-3d-gaming-experiences-for-copilot/) |Microsoft appears to be working on 3D gaming experiences for Copilot, its AI-powered chatbot platform, according to a new job listing. |
|[DeepSeek isn’t taking VC money yet — here are 3 reasons why.](https://techcrunch.com/2025/03/10/deepseek-isnt-taking-vc-money-yet-here-are-3-reasons-why/) | DeepSeek's founder, Liang Wenfeng, is steering clear of external investments to maintain control, using profits from his hedge fund, High-Flyer, for funding. Despite its success, DeepSeek faces obstacles such as stringent Chinese data laws and chip import restrictions imposed by U.S. export controls. While the company has managed to avoid outside capital so far, there is a possibility of future investment as DeepSeek begins to shift towards monetization.|
|[Smalldiffusion.](https://github.com/yuanchenyang/smalldiffusion) |A minimal, readable, and performant toolkit for training and sampling from diffusion models. |
|[Musk may still have a chance to thwart OpenAI’s for-profit conversion.](https://techcrunch.com/2025/03/09/musk-may-still-have-a-chance-to-thwart-openais-for-profit-conversion/) |A federal judge denied Elon Musk's request to stop OpenAI's shift to a for-profit model, though the ruling raised concerns about the transition. An expedited trial is scheduled for 2025 to resolve disputes over the restructuring. OpenAI's move towards a for-profit model is facing regulatory scrutiny and possible challenges from legal and AI safety viewpoints. |
|[Gmail's “Add to calendar” Feature Powered by Gemini.](https://workspaceupdates.googleblog.com/2025/03/add-events-to-google-calendar-using-gemini-in-gmail.html) |Google's Gemini in Gmail now identifies calendar-related content in emails and provides an "Add to calendar" button for easy scheduling. |
|[Spotify is trumpeting big paydays for artists – but only a tiny fraction of them are actually thrivin.](https://www.theguardian.com/music/2025/mar/12/spotify-is-trumpeting-big-paydays-for-artists-but-only-a-tiny-fraction-of-them-are-actually-thriving-loud-and-clear-report) | The company’s latest Loud & Clear report – a relatively transparent look into a closed-off industry – shows just how skewed financial success is in music|
|[OpenAI Introduces New Tools for AI Agents.](https://openai.com/index/new-tools-for-building-agents/) |OpenAI has introduced new APIs and an Agents SDK to simplify the process of building AI agents for developers. The toolkit includes web and file search, computer usage features, and observability tools to enhance agent orchestration and task automation. |
|[Reka's New Reasoning Model.](https://www.reka.ai/news/introducing-reka-flash) | Reka has open-sourced Reka Flash 3, a 21B parameter general-purpose model designed for reasoning, chat, coding, and instruction following. It competes well with proprietary models and offers a 32k context length, making it ideal for low-latency and on-device applications.|
|[Lopsided AI Revenues.](https://tomtunguz.com/ai-hardware-software/) | NVIDIA's data center business leads the AI market with Q4 revenues of $31 billion and margins exceeding 70%. Microsoft and IBM follow with $3.25 billion and $2 billion in AI revenues, respectively. Hardware continues to dominate over software and services in the AI sector.|
|[Google releases SpeciesNet, an AI model designed to identify wildlife.](https://techcrunch.com/2025/03/03/google-releases-speciesnet-an-ai-model-designed-to-identify-wildlife/) | Google has open sourced an AI model, SpeciesNet, designed to identify animal species by analyzing photos from camera traps. Researchers around the world use camera traps — digital cameras connected to infrared sensors — to study wildlife populations. But while these traps can provide valuable insights, they generate massive volumes of data that take days to weeks to sift through.|
|[Microsoft’s new Dragon Copilot is an AI assistant for healthcare.](https://www.theverge.com/news/622528/microsoft-dragon-copilot-ai-healthcare-assistant) | Microsoft has announced Microsoft Dragon Copilot, an AI system for healthcare that can, among other things, listen to and create notes based on clinical visits. The system combines voice-dictating and ambient listening tech created by AI voice company Nuance, which Microsoft bought in 2021.|
|[Google upgrades Colab with an AI agent tool.](https://techcrunch.com/2025/03/03/google-upgrades-colab-with-an-ai-agent-tool/) | Google Colab, Google’s cloud-based notebook tool for coding, data science, and AI, is gaining a new “AI agent” tool, Data Science Agent, to help Colab users quickly clean data, visualize trends, and get insights on their uploaded data sets.|
|[Apple to appeal against UK government data demand at secret high court hearing.](https://www.theguardian.com/technology/2025/mar/12/apple-to-appeal-against-uk-government-data-demand-at-secret-high-court-hearing) | Guardian understands tech company’s appeal against Home Office request for encrypted data is to be heard by tribunal on Friday|
|[Transforming Game Asset Creation With Genies' AIGC-Powered System - Genies.](https://genies.com/blog/transforming-game-asset-creation-with-genies-aigc-powered-system) | Game Art Forge introduces AI-generated templates to simplify game asset creation, boosting speed, scalability, and creative control for developers. It offers customization while preserving consistency and caters to both indie developers and larger teams. By merging AI efficiency with human creativity, it provides high-quality, flexible workflows for game development.|
|[AI Scientist's First Publication.](https://sakana.ai/ai-scientist-first-publication/) |The AI scientist from Sakana Labs has its first reviewed and accepted publication. It generated the idea, conducted experiments, and wrote a groundbreaking paper that was accepted to an ICLR workshop (with full consent from the conference organizers). |
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
|[Deriving Muon.](https://jeremybernste.in/writing/deriving-muon) | Adam has been the leading optimizer in deep learning for years. However, recently, the community has discovered that Muon could be a promising alternative. It achieves many of the same results as muP without needing any changes to the model. This post outlines some of the theoretical reasons behind the optimizer.|
|[Optimal Hyperparameter Scaling Law in Large Language Model Pretraining.](https://arxiv.org/abs/2503.04715) |Step Law is a comprehensive optimal hyperparameter scaling law that applies to various model structures, architectures, and data distributions. This allows predictions about how models are likely to perform before the training process even begins. |
|[Time-Series Forecasting.](https://arxiv.org/abs/2503.02836v1) | SeqFusion is a framework that sequentially chooses and combines pre-trained models for zero-shot forecasting. In contrast to traditional methods, it reduces data usage to improve privacy while still delivering strong accuracy across various temporal patterns.|
|[Distractor Aware SAM .](https://github.com/jovanavidenovic/DAM4SAM/) | Segment Anything (SAM) is a top-tier model for visual analysis and segmentation. However, it can struggle when two similar-looking objects appear in a video. This new approach addresses these "distractors" by incorporating extra memory augmentation and training.|
|[Autoregressive Streaming Text-to-Speech Model for Any LLM.](https://github.com/mbzuai-oryx/LLMVoX) | A compact 30 million parameter model designed to enhance any language model, enabling it to comprehend and generate speech in response to general queries. Importantly, it doesn't require adjustments to the base model, making it easily transferable across different models.|
|[Federated Learning for Neural Feedforward Control.](https://github.com/j-cap/FL-based-neural-FF-control) | This project presents a federated learning-based method for neural feedforward control, enabling multi-agent systems to enhance tracking performance while maintaining data privacy.|
|[Gemini Embedding Model.](https://developers.googleblog.com/en/gemini-embedding-text-model-now-available-gemini-api) |The Gemini team has developed and released an outstanding embedding model for text. It leads in benchmark performance, is cost-effective, and offers excellent speed.|
|[Token-Efficient Long Video Understanding for Multimodal LLMs.](https://research.nvidia.com/labs/lpr/storm/) |Most video understanding models process individual frames, which makes addressing temporal questions challenging. STORM, which leverages Mamba adapters, introduces temporal attention operations. This post compares it with Qwen models. |
|[Video Painter.](https://yxbian23.github.io/project/video-painter) |A new video inpainting model, VideoPainter, effectively integrates background information, supports videos of any length, and utilizes a dedicated dataset and benchmark for training and evaluation. Its design goes beyond basic inpainting, offering potential for advanced video manipulation and the generation of related training data. |
|[Detecting misbehavior in frontier reasoning models.](https://openai.com/index/chain-of-thought-monitoring/) |This report from OpenAI discusses monitoring the chain of thought in advanced reasoning models. Frontier reasoning models take advantage of loopholes when possible. It demonstrates that an LLM can be used to detect these exploits in their chains of thought. Penalizing their "bad thoughts" doesn't eliminate most misbehavior—it simply causes the models to conceal their intentions. |
|[Flying Safer: Obstacle Avoidance for Fast Drones.](https://github.com/ch9397/fixedwing-monoppo) |This repository includes the implementation of a lightweight deep reinforcement learning-based collision avoidance system for fixed-wing unmanned aerial vehicles (UAVs), using AirSim and JSBSim. |
|[Teaching Language Models to Solve Sudoku Through Reinforcement Learning.](https://hrishbh.com/teaching-language-models-to-solve-sudoku-through-reinforcement-learning/) |This research investigates training AI language models to solve Sudoku puzzles using reinforcement learning, specifically applying Group Relative Policy Optimization (GRPO) to models like Qwen 2.5, without the need for external data or larger model distillation. A multi-faceted reward system was developed, focusing on correct answer formatting, proper grid structure, and accurate solutions, to help the models learn the logical rules and spatial reasoning required for Sudoku, transforming them from text predictors to structured problem-solvers. |
|[Hugging Face Expanding LeRobot Platform.](https://huggingface.co/blog/lerobot-goes-to-driving-school) |Hugging Face and Yaak have released L2D, the largest open-source multimodal dataset for automotive AI. It features driving policies from both experts and students collected from driving schools, along with natural language instructions to improve spatial intelligence models. |
|[MovieAgent: Automated Movie Generation via Multi-Agent CoT Planning.](https://weijiawu.github.io/MovieAgent/) |This system combines multiple generative modalities and employs persona-based prompting to promote consistency and accuracy. It then utilizes the Stable Diffusion video model to generate and assemble frames. This process could likely be enhanced with the use of Wan. |
|[Large Language and Vision Embedding Models with Hardness-Weighted Contrastive Learning.](https://arxiv.org/abs/2503.04812) |Building embedding models for vision and language tasks using a contrastive loss often causes these models to struggle with hard negative pairs. This work introduces a regularization strategy and reports significant improvement in challenging retrieval tasks. The method also scales effectively for zero-shot video retrieval. |
|[YoloE: real-time open vocabulary detection.](https://github.com/THU-MIG/yoloe) |Small vision models can be prompted in various ways for open vocabulary detection. This allows the use of classes, images, and text to guide the model on what to detect. Notably, it operates at 300fps, making it suitable for real-time applications. |
|[Perception efficient reconstruction.](https://github.com/hujiecpp/PE3R) |Another approach combines textual query capabilities with 3D reconstruction from images. This specific system utilizes a feed-forward model for fast reconstruction. |
|[DeepMind's Image-Text Model.](https://github.com/google-deepmind/tips) | DeepMind has unveiled TIPS, an innovative image-text model designed for dense and global vision tasks. By combining contrastive learning with masked image modeling and leveraging synthetic captions, it demonstrates strong spatial awareness and surpasses existing methods in several benchmarks.|
|[The TechCrunch AI glossary.](https://techcrunch.com/2025/03/02/the-techcrunch-ai-glossary/) | This article defines key AI terminology, such as "AI agents," chain-of-thought reasoning, deep learning, and large language models (LLMs). Deep learning is explained as a subset of machine learning inspired by neural pathways in the human brain, while LLMs are described as neural networks that power AI assistants like ChatGPT. The article also covers fine-tuning and the role of weights in optimizing AI models.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[It begins: Pentagon to give AI agents a role in decision making, ops planning.](https://www.theregister.com/2025/03/05/dod_taps_scale_to_bring/) |The US military has granted a major contract to Scale AI and its partners, including Anduril and Microsoft, to incorporate AI agents into military operations for decision-making in workflows. The Thunderforge project seeks to improve the speed and accuracy of strategic planning while ensuring human oversight. The Pentagon intends to eventually implement this AI system across all of its combatant commands. |
|[AI Market Summary & Conclusions .](https://klaothongchan.medium.com/ai-market-summary-conclusions-march-2-2025-4196a23a5a68) |The AI industry is advancing quickly, with OpenAI's GPT-4.5 concentrating on refinement, Google's Gemini 2.0 encountering adoption challenges, and China's DeepSeek and Kimi pushing the boundaries of cost-effective, specialized AI. Meta is expanding AI into wearables and robotics, while XAI's Grok 3 remains a niche contender. The AI race is intensifying as specialization and real-world integration become increasingly important, posing a challenge to established players like OpenAI and Google. |
|[Why are proponents of ‘smart cities’ neglecting research?](https://www.nature.com/articles/d41586-025-00727-7) |Despite the buzz surrounding smart cities in urban-policy circles, studies are lacking on the evidence for what works, what doesn’t — and who benefits. |
|[For more reliable AI, academics should edit Wikipedia.](https://www.nature.com/articles/d41586-025-00715-x) |Wikipedia, the “encyclopedia anyone can edit”, is a fundamental training and reference data set for large language models (LLMs) |
|[Artificial intelligence speaks up.](https://www.science.org/doi/10.1126/science.adu1567) |An AI safety specialist confronts fears about the future of large language models |
|[Cutting AI down to size.](https://www.science.org/content/article/what-s-tinyml-global-south-s-alternative-power-hungry-pricey-ai) |Many artificial intelligence models are power hungry and expensive. Researchers in the Global South are increasingly embracing low-cost, low-power alternatives |
|[Evaluating animal consciousness.](https://www.science.org/doi/10.1126/science.adp4990) |An emerging field shows how animal feelings can be studied scientifically |
|[High-performance computing at a crossroads.](https://www.science.org/doi/10.1126/science.adu0801) |Long-term plans and comprehensive vision are needed |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |












































































































