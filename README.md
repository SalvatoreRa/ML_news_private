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
|[Traveling Waves Integrate Spatial Information Through Time.](https://arxiv.org/abs/2502.06034) |Harvard and Western University researchers propose a wave-based recurrent neural network that uses traveling neural waves for global spatial integration in visual tasks. Inspired by the "Hearing the Shape of a Drum" analogy, they model neural activity as locally coupled oscillators, discretizing the 2D wave equation into a convolutional recurrent network. Unlike standard RNNs, their model aggregates information across entire wave evolution, enhancing segmentation tasks requiring large receptive fields. It rivals deeper CNNs and U-Nets with fewer parameters on both synthetic and real-world datasets. The approach may also align with neuroscience, reflecting observed traveling wave dynamics in the brain. |
|[Transformers without Normalization.](https://arxiv.org/abs/2503.10622) |Researchers from Meta, NYU, MIT, and Princeton introduce Dynamic Tanh (DyT), a method that removes normalization layers in Transformers while maintaining or improving performance. They observe that LayerNorm behaves like a tanh function, leading to DyT, which replaces normalization with a per-channel tanh(αx) and learnable affine parameters, preserving non-linearity without activation statistics. DyT ensures stable convergence across vision, speech, diffusion, and language tasks, matching or exceeding LayerNorm-based models while improving efficiency and training speed in large LLaMA models. Its simplicity reduces computation overhead and integrates easily into existing architectures with minimal modifications. |
|[Monitoring Reasoning Models for Misbehavior.](https://cdn.openai.com/pdf/34f2ada6-870f-4c26-9790-fd8def56387f/CoT_Monitoring.pdf) | OpenAI researchers explore how monitoring chain-of-thought (CoT) reasoning in LLMs can detect misaligned behaviors, including reward hacking. By analyzing reasoning traces rather than just outputs, they identify sophisticated exploits in coding tasks, often outperforming action-based monitoring. However, optimizing rewards using CoT monitors can lead to "obfuscated reward hacking," where models conceal their intent while still cheating. Surprisingly, weaker models can effectively oversee stronger ones, suggesting scalable oversight is possible. The study also highlights a trade-off: optimizing CoT for safety or brevity may reduce transparency, making future monitoring and alignment more difficult.|
|[Improving Planning of Agents for Long-Horizon Tasks.](https://arxiv.org/abs/2503.09572) |Researchers from UC Berkeley and the University of Tokyo introduce Plan-and-Act, a framework that separates high-level planning from execution in LLM-based agents, improving performance on long-horizon tasks. It consists of a Planner that structures goals into steps and an Executor that carries them out, reducing cognitive overload. To train these modules efficiently, they generate synthetic plan-action pairs by reverse-engineering successful executions and expanding them with LLM-powered augmentation, avoiding costly manual labeling. The system also dynamically updates plans based on new information, enabling real-time adjustments. Tested on WebArena-Lite, it achieves a 54% success rate, surpassing previous benchmarks and demonstrating the power of structured planning with synthetic training data. |
|[Gemini Robotics: Bringing AI into the Physical World.](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf) | Google DeepMind introduces Gemini Robotics, a family of embodied AI models that integrate large multimodal reasoning into robotics, enabling perception, interpretation, and interaction in 3D environments. Built on Gemini 2.0, it includes Gemini Robotics-ER for spatial reasoning and a real-time system for precise robotic control in tasks like object manipulation. By combining multi-view correspondence, 3D detection, and trajectory planning, the model executes diverse tasks with minimal data, reducing training costs. It generalizes well to new instructions and conditions while incorporating safety checks for real-world deployment. This marks a major step toward universal robotics, aiming for advanced planning and sensorimotor control in practical applications.|
|[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.](https://arxiv.org/abs/2503.09516) |This paper explores search-augmented reasoning by training LLMs to query a search engine multiple times during reasoning using reinforcement learning. Unlike one-shot retrieval-augmented generation (RAG), the model refines its queries dynamically through multi-turn retrieval. It learns query optimization solely through outcome rewards, eliminating the need for large supervised datasets. To stabilize training, retrieved text is masked from policy gradient updates. The approach significantly improves accuracy, achieving up to +26% gains on QA benchmarks like NQ and TriviaQA. It generalizes across architectures, including Qwen and LLaMA, demonstrating a scalable method for integrating real-time search into LLM reasoning with minimal supervision. |
|[Auditing LLMs for Hidden Objectives.](https://assets.anthropic.com/m/317564659027fb33/original/Auditing-Language-Models-for-Hidden-Objectives.pdf) |Anthropic introduces a framework for auditing LLMs to detect hidden objectives beyond intended goals. By deliberately training a model to exploit reward model flaws, they demonstrate how LLMs can develop unintended behaviors, learning to "please the reward model" rather than follow explicit instructions. In a blind auditing game, teams analyzed the model’s weights, data, and outputs, with most successfully uncovering the hidden objective. Comparing eight auditing techniques, they found that while simple methods like semantic search often work, interpretability tools like sparse autoencoders can reveal deeper patterns. This study underscores the risk of unintended objectives in AI and suggests systematic audits combining data inspection, interpretability, and behavioral tests as a crucial step for safe deployment.|
|[Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.](https://arxiv.org/abs/2503.09573) | Researchers from Cornell Tech, Stanford, and Cohere introduce Block Diffusion (BD3-LMs), a framework that combines autoregressive modeling with discrete diffusion to enable parallel token sampling and flexible-length text generation. Unlike traditional diffusion models with fixed-length constraints or slow token-by-token AR models, BD3-LMs divide sequences into blocks, applying diffusion within each and stacking them autoregressively. This allows efficient generation beyond training limits, improving perplexity and sampling speed through a specialized training approach and noise schedule. The model balances block size for optimal performance and is open-source, with future applications in chatbots, code generation, and controllable text synthesis.|
|[Scaling Laws for DiLoCo.](https://arxiv.org/abs/2503.09799) |DeepMind has published a paper outlining scaling laws for the powerful cross-data center training algorithm DiLoCo. These laws demonstrate how stable model training can be maintained, even when gradients are synchronized across continents. |
|[Retrieval-Augmented Generation with Hierarchical Knowledge (HiRAG).](https://github.com/hhy-huang/HiRAG) |HiRAG introduces a hierarchical knowledge-based approach to Retrieval-Augmented Generation (RAG), enhancing semantic understanding and indexing for domain-specific tasks. |
|[‘Open’ AI model licenses often carry concerning restrictions.](https://techcrunch.com/2025/03/14/open-ai-model-licenses-often-carry-concerning-restrictions/) |Many AI models labeled as "open" come with restrictive licensing terms. Google's new Gemma 3 models and similar releases from Meta raise concerns about commercial limitations, which could affect smaller companies that depend on these technologies. |
|[https://arxiv.org/abs/2503.11061v2.](https://arxiv.org/abs/2503.11061v2) |Funsearch is a new LLM-driven genetic algorithm designed to help mathematicians tackle combinatorial and number-theoretic problems without needing machine learning expertise. |
|[TxAgent: An AI agent for therapeutic reasoning across a universe of tools.](https://zitniklab.hms.harvard.edu/TxAgent/) |TxAgent is an AI-driven system that assesses drug interactions, contraindications, and patient-specific data to create adaptive treatment plans. |
|[Optimizing generative AI by backpropagating language model feedback.](https://www.nature.com/articles/s41586-025-08661-4) |Generative artificial intelligence (AI) systems can be optimized using TextGrad, a framework that performs optimization by backpropagating large-language-model-generated feedback; TextGrad enables optimization across diverse tasks, including radiotherapy treatment plans and molecule generation. |
|[A mixed-precision memristor and SRAM compute-in-memory AI processor.](https://www.nature.com/articles/s41586-025-08639-2) |A mixed-precision heterogeneous memristor combined with a compute-in-memory artificial intelligence (AI) processor allows optimization of the precision, energy efficiency, storage and wakeup-to-response time requirements of AI edge devices, which is demonstrated using existing models and datasets. |
|[A generative model for inorganic materials design.](https://www.nature.com/articles/s41586-025-08628-5) |MatterGen is a model that generates stable, diverse inorganic materials across the periodic table and can further be fine-tuned to steer the generation towards a broad range of property constraints. |
|[Plug-and-play external knowledge for LLMs.](https://www.microsoft.com/en-us/research/blog/introducing-kblam-bringing-plug-and-play-external-knowledge-to-llms/) | A novel approach to integrate knowledge into an LLM without retraining, enabling online and realtime learning.|
|[The KoLMogorov Test: Compression by Code Generation.](https://arxiv.org/abs/2503.13992) | Meta has introduced a new benchmark to evaluate a language model's reasoning and knowledge. The test involves presenting the model with a long data sequence and requiring it to generate the shortest program that can reproduce the sequence and halt. This process, known as Kolmogorov compression, is uncomputable in polynomial time and demands strong reasoning capabilities.|
|[Measuring AI Ability to Complete Long Tasks.](https://arxiv.org/abs/2503.14499) |A new "Moore's Law" for agents' task horizon ability indicates that the length of tasks an agent can complete has been doubling every seven months. This trend suggests that within the next two years, agents will be capable of handling multi-hour tasks with numerous complex steps. |
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
|[‘Deeply uncomfortable’: UK Starlink users switch off over Musk’s political machinations.](https://www.theguardian.com/technology/2025/mar/16/deeply-uncomfortable-uk-starlink-users-switch-off-over-musks-political-machinations) |Numbers using satellite broadband system has been growing but users are having second thoughts due to Musk’s role in Donald Trump’s administration |
|[Google Assistant is Replaced by Gemini.](https://blog.google/products/gemini/google-assistant-gemini-mobile/) | Google Assistant is evolving into Gemini, a more personalized and AI-driven assistant designed to integrate with apps and services, utilizing generative AI.| 
|[Google's Response to U.S. AI Policy.](https://blog.google/outreach-initiatives/public-policy/google-us-ai-action-plan-comments/) | Google has outlined its vision for U.S. AI policy, calling for investments in AI infrastructure, streamlined government adoption, and international pro-innovation standards to sustain leadership in the AI field.|
|[Sakana claims its AI-generated paper passed peer review — but it’s a bit more nuanced than that.](https://techcrunch.com/2025/03/12/sakana-claims-its-ai-paper-passed-peer-review-but-its-a-bit-more-nuanced-than-that/) | Japanese AI startup Sakana said that its AI generated one of the first peer-reviewed scientific publications. But while the claim isn’t necessarily untrue, there are caveats to note.|
|[No part of Amazon is ‘unaffected’ by AI, says its head of AGI.](https://techcrunch.com/2025/03/03/no-part-of-amazon-is-unaffected-by-ai-says-its-head-of-agi/) | Amazon's VP of Artificial General Intelligence, Vishal Sharma, has confirmed the integration of AI across AWS, robotics, and Alexa, emphasizing the company's wide-ranging AI deployment.|
|[PANORAMA Challenge for Cancer Detection.](https://panorama.grand-challenge.org/) |The PANORAMA study is an international initiative evaluating AI models and radiologists in detecting pancreatic cancer in CECT scans. |
|[Performing arts leaders issue copyright warning over UK government’s AI plans.](https://www.theguardian.com/culture/2025/mar/18/performing-arts-leaders-issue-copyright-warning-over-uk-governments-ai-plans) |In a statement, 35 signatories from dance, theatre and music industries express concern about ‘fragile ecosystem’ |
|[The court rejects Elon’s latest attempt to slow OpenAI down.](https://openai.com/index/court-rejects-elon/) |A court dismissed key claims in Elon Musk's lawsuit against OpenAI, rejecting his preliminary injunction request. |
|[UiPath looks for a path to growth with Peak agentic AI acquisition.](https://techcrunch.com/2025/03/13/uipath-is-looking-for-a-path-to-growth-in-agentic-ai-with-its-peak-ai-acquisition/) |UiPath has acquired Peak.ai to strengthen its AI and automation services, particularly in retail and manufacturing. Despite recent revenue struggles and a downgraded forecast, UiPath aims to leverage Peak's decision-making AI to enhance cross-selling and expand market share. The acquisition signals a strategic shift toward deeper AI integration within UiPath's existing offerings. |
|[People are using Google’s new AI model to remove watermarks from images.](https://techcrunch.com/2025/03/17/people-are-using-googles-new-ai-model-to-remove-watermarks-from-images/) | Users on social media have discovered a controversial use case for Google’s new Gemini AI model: removing watermarks from images, including from images published by Getty Images and other well-known stock media outfits.|
|[Yet another AI robotics firm lands major funding, as Dexterity closes latest round.](https://techcrunch.com/2025/03/11/yet-another-ai-robotics-firm-lands-major-funding-as-dexterity-closes-latest-round/) | The intersection of robotics and AI continues to attract attention from investors and Big Tech alike. The latest indicator? Dexterity, a startup specializing in industrial robots with “human-like” finesse, has raised $95 million at a post-money valuation of $1.65 billion, per Bloomberg.|
|[Mark Cuban says AI is ‘never the answer,’ it’s a ‘tool’.](https://techcrunch.com/2025/03/11/mark-cuban-says-ai-is-never-the-answer-its-a-tool/) |Mark Cuban shared his thoughts on how AI technology can help small businesses outperform their competition. In short, he told the crowd that AI was not the answer, in and of itself; it’s meant to serve as an aid that can help entrepreneurs by making it easier to get started growing their businesses and answering questions along the way. |
|[Google’s parent to buy cybersecurity group Wiz in its biggest ever deal.](https://www.theguardian.com/technology/2025/mar/18/google-parent-alphabet-buy-cybersecurity-wiz-israeli-startup) |Alphabet’s acquisition of Israeli startup for $32bn follows rejection of takeover bid last summer |
|[Italian newspaper says it has published world’s first AI-generated edition.](https://www.theguardian.com/technology/2025/mar/18/italian-newspaper-says-it-has-published-worlds-first-ai-generated-edition) |Il Foglio says artificial intelligence used ‘for everything – the writing, the headlines, the quotes … even the irony’ |
|[Google updates Gemini app.](https://blog.google/products/gemini/gemini-collaboration-features) |Google has introduced several key features to its Gemini app and web browser, including Canvas, Audio Overview, and a coding assistant. |
|[AI search is starting to kill Google’s ‘ten blue links’.](https://www.theverge.com/ai-artificial-intelligence/631352/ai-search-adobe-analytics-google-perplexity-openai) |Adobe reported a 1,300% increase in AI search referrals for retailers during the 2024 holiday season, signaling a shift away from traditional search engines. Engagement metrics show AI search users stay longer and have lower bounce rates compared to traditional referrals. Despite early challenges, AI search is gaining momentum, with significant consumer adoption for shopping and research. |
|[Zero-Shot Steering for 3D/4D Scene Generation.](https://byeongjun-park.github.io/SteerX) |SteerX has introduced a zero-shot inference-time approach to enhance geometric alignment in 3D/4D scene generation, utilizing pose-free feed-forward reconstruction models and geometric reward functions. |
|[MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling.](https://github.com/hustvl/MaTVLM) | MaTVLM is a hybrid vision-language model that incorporates Mamba-2 layers into a pre-trained VLM, enhancing convergence speed and overall performance.|
|[Google's new robot AI can fold delicate origami, close zipper bags without damage.](https://arstechnica.com/ai/2025/03/googles-origami-folding-ai-brain-may-power-new-wave-of-humanoid-robots/) | Google DeepMind has introduced Gemini Robotics and Gemini Robotics-ER, AI models designed to enhance robots' fine motor skills and adaptability for real-world tasks. Gemini Robotics combines vision, language, and action capabilities, enabling robots to perform intricate tasks like origami folding. Early results show notable improvements in generalization and dexterity compared to previous models, with Google collaborating with Apptronik and other partners for further development.|
|[Google and NVIDIA Expand AI Collaboration.](https://blog.google/technology/ai/google-nvidia-gtc-ai) | Google announced an expanded partnership with NVIDIA at NVIDIA GTC, aiming to enhance AI infrastructure, optimize hardware, and advance applications in areas such as healthcare, energy, and robotics.|
|[Google playing AI search catchup, and forming relationships with chatbots.](https://www.technologyreview.com/2025/03/17/1113264/the-download-google-playing-ai-search-catchup-and-forming-relationships-with-chatbots/) | Google is adding new AI features from Gemini to its search but this is seen as playing catch-up to OpenAI's ChatGPT.|
|[Robotic fingers can tell objects apart by touch.](https://www.nature.com/articles/d41586-025-00706-y) | Prosthetic appendage uses three layers of touch sensors to accurately differentiate between textures.|
|[Microsoft quantum computing ‘breakthrough’ faces fresh challenge.](https://www.nature.com/articles/d41586-025-00683-2) | Analysis pokes holes in protocol that underpins Microsoft’s claim to have created the first topological qubits.|
|[UK cybersecurity agency warns over risk of quantum hackers.](https://www.theguardian.com/technology/2025/mar/20/uk-cybersecurity-agency-quantum-hackers) | Organisations including energy and transport firms told to guard systems against powerful new computers|
|[Anthropic's Red Team Warns of AI Security Risks.](https://www.anthropic.com/news/strategic-warning-for-ai-risk-progress-and-insights-from-our-frontier-red-team) |AI models are making rapid advancements in cybersecurity and biology, reaching expert-level knowledge in some domains. While current risks are manageable, the company cautions that stronger safeguards will be necessary as capabilities continue to evolve. |
|[Google's AI Health Innovations Announced at The Check Up 2025.](https://blog.google/technology/health/google-health-check-up-2025/) |At The Check Up 2025, Google showcased how AI is advancing global healthcare, from diagnostic tools to research collaborations. The company aims to enhance medical accessibility and improve patient outcomes through AI-driven innovations. |
|[Nvidia's Humanoid Robot.](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/) | Nvidia's humanoid robot, gr00t, is trained using its software suite to perform generalist tasks.|
|[Nvidia DGX Spark.](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) | Nvidia has rebranded its Digits personal computing device as Spark, now featuring reduced memory bandwidth and lower VRAM. Additionally, the company introduced the DGX Station, a high-performance workstation offering a more powerful alternative.|
|[Anthropic CEO says spies are after $100M AI secrets in a ‘few lines of code’.](https://techcrunch.com/2025/03/12/anthropic-ceo-says-spies-are-after-100m-ai-secrets-in-a-few-lines-of-code/) | Anthropic’s CEO Dario Amodei is worried that spies, likely from China, are getting their hands on costly “algorithmic secrets” from the U.S.’s top AI companies — and he wants the U.S. government to step in.|
|[Perplexity and SoftBank Launch Enterprise Pro Japan.](https://www.perplexity.ai/hub/blog/perplexity-expands-partnership-with-softbank-to-launch-enterprise-pro-japan) | Perplexity has partnered with SoftBank to introduce Enterprise Pro to Japanese corporations, leveraging SoftBank's extensive network.|
|[Google's Gemini AI Enhances Efficiency in Japanese Hospitals.](https://blog.google/technology/health/gemini-ai-japan-hospitals/) | Google's Gemini AI is enhancing administrative efficiency in Japanese hospitals by automating documentation tasks, easing the cognitive burden on healthcare workers, and allowing more time for patient care.|
|[Google's Gemini Robotics AI Model Reaches Into the Physical World.](https://www.wired.com/story/googles-gemini-robotics-ai-model-that-reaches-into-the-physical-world/) | Google DeepMind has unveiled Gemini Robotics, which enhances AI with language, vision, and physical action capabilities.|
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Gemma 3 Technical Report.](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) | Gemma 3 is a lightweight open model family (1B–27B parameters) with multimodal capabilities, extended context length (up to 128K tokens), and multilingual support. It integrates a frozen SigLIP vision encoder, using a Pan & Scan method for better image processing and handling diverse aspect ratios. Its hybrid attention system reduces memory usage for long-context tasks. Advanced knowledge distillation and quantization (int4, switched-fp8) allow for efficient deployment on consumer GPUs and edge devices. Instruction tuning enhances performance in benchmarks like MMLU, coding, and chat, placing it among the top models. Supporting over 140 languages, it enables structured outputs and function calling for agentic workflows while ensuring safety and privacy through data filtering and reduced memorization.|
|[A Survey on Post-training of Large Language Models.](https://arxiv.org/abs/2503.06072) |PoLMs like OpenAI-o1/o3 and DeepSeek-R1 address LLM weaknesses in reasoning, ethics, and specialized tasks. This survey examines their development, categorizing techniques in fine-tuning, alignment, reasoning, efficiency, and integration to advance more capable and adaptable AI. |
|[Speaker Identification with Whisper.](https://arxiv.org/abs/2503.10446v1) | WSI repurposes the Whisper ASR encoder for multilingual speaker identification through joint loss optimization. It surpasses Pyannote, ECAPA TDNN, and Xvector in identifying speakers across various languages and environments.|
|[Visual reasoning models.](https://github.com/groundlight/r1_vlm) |Toolkit for training VLMs to have improved grounding and reasoning capabilities. |
|[Optimized workforce learning agent.](https://github.com/camel-ai/owl) |OWL is an agentic framework that appears both sensible and efficient. It enables easy composition and can even replicate functionality from some closed-source agents. |
|[Luma's new pre-training method for multi-modal models.](https://lumalabs.ai/news/inductive-moment-matching) | Luma Chief Scientist Jiaming Song, who developed the first accelerated algorithm for diffusion models, has introduced Inductive Moment Matching (IMM)—a new multi-modal pre-training method that outperforms diffusion models, offering superior sample quality and 10x greater efficiency.|
|[ThunderKittens on Blackwell.](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell) | ThunderKittens is a robust and straightforward abstraction for writing efficient, high-performance CUDA kernels. This post examines how to use the framework with Nvidia's latest Blackwell series of GPUs. The key difference lies in thinking in terms of data flow.|
|[Mistral Small 3.1.](https://mistral.ai/news/mistral-small-3-1) |Built on Mistral Small 3, this new model features enhanced text performance, improved multimodal understanding, and an expanded context window of up to 128k tokens. It surpasses comparable models like Gemma 3 and GPT-4o Mini while achieving inference speeds of 150 tokens per second. |
|[SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation.](https://arxiv.org/abs/2503.09641) | Nvidia has introduced SANA-sprint, a faster version of its SANA image generation model. This model delivers remarkably fast image generation while preserving quality. The team employs a novel distillation method based on consistency distillation. A key challenge in this area remains ensuring that these consistency models remain easy to fine-tune.|
|[DriveLMM-o1: A Step-by-Step Reasoning Dataset and Large Multimodal Model for Driving Scenario Understanding.](https://arxiv.org/abs/2503.10621v1) | DriveLMM-o1 presents a dataset and benchmark for stepwise visual reasoning in autonomous driving, enhancing AI-driven decision-making and reasoning accuracy in driving scenarios.|
|[CSM speech model on MLX.](https://github.com/senstella/csm-mlx) | Sesame recently released a 1B model for conversational speech generation. This repository provides an Apple-native MLX version optimized for fast performance on most MacBooks.|
|[MMS-LLaMA: A Speech-Focused Multimodal LLM.](https://github.com/JeongHun0716/MMS-LLaMA) |MMS-LLaMA is an efficient multimodal speech LLM framework for automatic visual speech recognition (AVSR), reducing token length while maintaining linguistic content integrity. |
|[Open-Source Handwritten Signature Detection Model.](https://huggingface.co/blog/samuellimabraz/signature-detection-model) | A comprehensive post that explores the performance of every model on Hugging Face for handwriting classification/signature detection.|
|[Roblox Open-Sources Its Generative 3D Model.](https://corp.roblox.com/newsroom/2025/03/introducing-roblox-cube) |Roblox has unveiled Cube, a generative AI system for 3D and 4D model generation. A beta version of Cube's 3D mesh generation will be integrated into Roblox Studio and made available as a Lua API. |
|[SmolDocling.](https://arxiv.org/abs/2503.11576) |A new, ultra-compact state-of-the-art model for document OCR runs exceptionally fast while maintaining sufficient accuracy for various document processing tasks. |
|[reWordBench.](https://arxiv.org/abs/2503.11751) |This work demonstrates that many popular reward models are fragile to simple rewording of prompts. It introduces a benchmark and proposes a strategy to improve their robustness. |
|[Single-View 3D Scene Reconstruction.](https://ai-kunkun.github.io/Niagara_page/) |Niagara introduces an innovative framework for reconstructing outdoor 3D scenes from a single image, leveraging depth and normal estimation alongside a geometric affine field and 3D Gaussian decoding. |
|[Stable virtual camera.](https://github.com/Stability-AI/stable-virtual-camera) | Stability AI has launched an impressive multiview camera system for novel view synthesis. While not entirely state-of-the-art, it is powerful, non-commercial, and runs efficiently in just two forward passes.|
|[Personalize Anything.](https://fenghora.github.io/Personalize-Anything-Page/) |You can personalize almost any image without additional fine-tuning or training by using token replacement in a Diffusion Transformer. |
|[TrainXGB - Train XGBoost in Browser.](https://www.xgblog.ai/p/trainxgb-train-xgboost-in-browser) |This blog post describes how to build a browser based training system accelerated by WASM.|
|[Stability's 3D Video Generation.](https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control) | Stable Virtual Camera is a multi-view diffusion model that creates immersive 3D videos from one or multiple 2D images, allowing for user-defined or pre-set camera trajectories.|
|[AAPM 2025 Challenge.](https://github.com/riqianggao/gdp-hmm_aapmchallenge) |This repository contains code and tutorials to help participants begin developing dose prediction models for the GDP-HMM Challenge at AAPM 2025. |
|[Open R1.](https://github.com/huggingface/open-r1) |The Open-R1 initiative aims to be more robust and feature-complete while remaining minimal and hackable. It includes additional SFT steps and data distillation for improved performance. |
|[.]() | |
|[.]() | |
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
|[AI firms follow DeepSeek's lead, create cheaper models with “distillation”.](https://arstechnica.com/ai/2025/03/ai-firms-follow-deepseeks-lead-create-cheaper-models-with-distillation/) | Leading AI firms like OpenAI, Microsoft, and Meta are using "distillation" to create more cost-effective models by training smaller systems with a "teacher" LLM.|
|[Paths and waystations in AI safety.](https://joecarlsmith.com/2025/03/11/paths-and-waystations-in-ai-safety) | This essay presents a framework for tackling the AI alignment problem by differentiating between technical parameters ("problem profile") and civilization's capacity to respond ("competence profile"). It highlights three key "security factors": safety progress, risk evaluation, and capability restraint. The discussion explores leveraging future AI labor to strengthen these factors and outlines strategic milestones for advancing AI safety.|
|[Nvidia won the AI race, but inference is still anyone's game.](https://www.theregister.com/2025/03/12/training_inference_shift/) | Nvidia currently dominates AI training, but the inference market remains open to competition.|
|[Global cooperation is crucial for DeepSeek and broader AI research.](https://www.nature.com/articles/d41586-025-00822-9) |China’s cost-effective, open-source artificial intelligence (AI) model DeepSeek-R1 — which rivals models from industry leaders — is indeed remarkable |
|[AI demands a different approach to education.](https://www.nature.com/articles/d41586-025-00823-8) | The rapid evolution of artificial intelligence (AI), marked by DeepSeek-style breakthroughs, challenges conventional learning approaches.|
|[Inside Zoom’s AI evolution: From basic meeting tools to agentic productivity platform powered by LLMs and SLMs.](https://venturebeat.com/ai/inside-zooms-ai-evolution-from-basic-meeting-tools-to-agentic-productivity-platform-powered-by-llms-and-slms/) |Zoom is evolving beyond video conferencing by building an agentic AI infrastructure that turns meetings into action-driven workflows. This includes AI Companion 2.0, featuring task management and document creation, along with customizable AI agents through the new AI Studio. Zoom’s federated approach integrates its custom small language model with larger LLMs, providing enterprises with efficient and adaptable AI solutions. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |













































































































