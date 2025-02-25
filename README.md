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
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[.]() | |
|[.]() | |
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
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |












































































































