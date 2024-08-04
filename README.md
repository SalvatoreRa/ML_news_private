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

# ON WORKING

# ML news: 

## Research
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



# ML news: Week 29 July - 4 August

## Research
|Link|description|
|---|---|
|[Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach.](https://arxiv.org/abs/2407.16833) |compares RAG to long-context LLMs and discovers that while RAG is much less expensive, long-context LLMs perform better on average; Offers Self-Route, which routes inquiries to RAG or LC by using self-reflection; it claims to have a substantial computational cost reduction with a performance that is comparable to LC. |
|[Recursive Introspection: Teaching Language Model Agents How to Self-Improve.](https://arxiv.org/abs/2407.18219) | asserts that LLMs can be iteratively fine-tuned to improve their own response over multiple turns with additional feedback from the environment; the LLM learns to recursively detect and correct its past mistakes in subsequent iterations; and enhances 7B models' self-improvement abilities on reasoning tasks (GSM8K and MATH), achieving an improvement over turns that is not observed in strong proprietary models.|
|[LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference.](https://arxiv.org/abs/2407.14057) |presents a novel dynamic token pruning technique for effective long-context LLM inference; it can maintain high accuracy while speeding up the prefilling stage of a Llama 2 7B model by 2.34 times; it computes the KV for tokens that are crucial for the next token prediction in both the prefilling and decoding stages; it enables language models to dynamically select different subsets of tokens from the context in different generation steps, even though they may have been pruned in a previous step. |
|[Generation Constraint Scaling Can Mitigate Hallucinatio.](https://arxiv.org/abs/2407.16908) |suggests a novel training-free method to reduce hallucinations in LLMs; they scaled the readout vector that limits generation in a memory-augmented LLM decoder; current research suggests that LLMs with explicit memory mechanisms can help reduce hallucinations; this work employs a memory-augmented LLM and applies lightweight memory primitives to limit generation in the decoder. |
|[Align and Distill: Unifying and Improving Domain Adaptive Object Detection.](https://arxiv.org/abs/2403.12029v1) |The difficulties of getting object detection models to perform well on a variety of data formats that they weren't initially trained on are addressed by a new method named ALDI. |
|[Small Molecule Optimization with Large Language Models.](https://arxiv.org/abs/2407.18897) |By gathering a dataset of 100 million molecules (40 billion token equivalent), two new language models were able to enhance their performance by 8% on the Practical Molecular Optimization benchmark. |
|[The Larger the Better? Improved LLM Code-Generation via Budget Reallocation.](https://arxiv.org/abs/2404.00725) | With a fairly comparable inference cost, code generation performance can be enhanced by repeatedly using smaller models.|
|[Self-Directed Synthetic Dialogues and Revisions Technical Report.](https://arxiv.org/abs/2407.18421) |More than 300,000 dialogues and criticisms will be incorporated into open models. The dataset, which was primarily produced with synthetics, is a potent illustration of synthetic data utilizing open models. |
|[Theia: Distilling Diverse Vision Foundation Models for Robot Learning.](https://arxiv.org/abs/2407.20179) |Theia, a vision foundation model for robot learning that combines several current vision models, is presented in this study. Rich visual representations provided by Theia improve robot learning even when using smaller model sizes and less training data. Test results indicate that Theia performs better than its predecessors, and the authors propose that enhanced performance is caused by more entropy in feature norms. The public is free to utilize the models and code. |
|[Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation.](https://arxiv.org/abs/2407.18910v1) | A novel strategy to increase the effectiveness and scalability of recommender systems is called LightGODE. By adopting a continuous graph ODE and concentrating on post-training graph convolution, it avoids the need for costly computations during training.|

## News
|Link|description|
|---|---|
|[Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) |a group of LLMs that includes models with 8B, 70B, and 405B parameters; it supports eight languages and expands the context window to 128K tokens; it exceeds state-of-the-art models in certain situations and competes favorably in other areas, including as general knowledge, math reasoning, and tool use. |
|[Nvidia’s new Titan GPU will beat the RTX 5090, according to leak.](https://www.pcgamesn.com/nvidia/blackwell-rtx-titan-rumor) |After skipping its ultra-expensive flagship graphics card with its Ada lineup, Nvidia could be bringing back the Titan with a Blackwell GPU. |
|[Elon Musk will ‘discuss’ Tesla investing $5 billion in his private AI company.](https://electrek.co/2024/07/25/elon-musk-will-discuss-tesla-investing-5-billion-private-ai-company/) |Elon Musk says that he will ‘discuss’ Tesla investing $5 billion in xAI, his own private artificial intelligence company. For the last few years, Musk has claimed that “Tesla is an AI company.” |
|[OpenAI training and inference costs could reach $7bn for 2024, AI startup set to lose $5bn - report.](https://www.datacenterdynamics.com/en/news/openai-training-and-inference-costs-could-reach-7bn-for-2024-ai-startup-set-to-lose-5bn-report/) |In 2023, OpenAI projected that ChatGPT inference would cost about $4 billion on Microsoft's Azure servers, potentially resulting in large financial losses. Even though OpenAI is making about $2 billion a year from ChatGPT, it would need more money in less than a year to cover a $5 billion deficit. With subsidized prices from Azure, it presently uses the equivalent of 350,000 Nvidia A100 chip servers, primarily for ChatGPT. |
|[Elon Musk sets new date for Tesla robotaxi reveal, calls everything beyond autonomy ‘noise’.](https://techcrunch.com/2024/07/23/elon-musk-sets-new-date-for-tesla-robotaxi-reveal-calls-everything-beyond-autonomy-noise) | Elon Musk says he will show off Tesla’s purpose-built “robotaxi” prototype during an event October 10, after scrapping a previous plan to reveal it August 8. Musk said Tesla will also show off “a couple of other things,” but didn’t explain what that meant.|
|[Stability AI steps into a new gen AI dimension with Stable Video 4D.](https://venturebeat.com/ai/stability-ai-steps-into-a-new-gen-ai-dimension-with-stable-video-4d/) |Stability AI is expanding its growing roster of generative AI models, quite literally adding a new dimension with the debut of Stable Video 4D. |
|[Google’s Gemini AI is getting faster with its Flash upgrade.](https://www.theverge.com/2024/7/25/24206071/google-gemini-ai-flash-upgrade) | Google’s Gemini AI chatbot will be able to respond to you more quickly and process more content in prompts thanks to an upgrade to the company’s Gemini 1.5 Flash AI model.|
|[Introducing SAM 2: The next generation of Meta Segment Anything Model for videos and images.](https://ai.meta.com/blog/segment-anything-2/) | Real time promptable segmentation for videos and images from Meta.|
|[Apple says its AI models were trained on Google’s custom chips.](https://www.cnbc.com/2024/07/29/apple-says-its-ai-models-were-trained-on-googles-custom-chips-.html) |Apple said in a technical paper on Monday that the two AI models underpinning Apple Intelligence, its AI system, were pretrained on Google-designed chips in the cloud. |
|[AI Startup Anthropic Faces Backlash for Excessive Web Scraping.](https://www.techopedia.com/news/ai-startup-anthropic-faces-backlash-for-excessive-web-scraping) | Freelancer.com CEO claims Anthropic's crawler violated the "do not crawl" protocol, causing site slowdowns.|
|[Apple Intelligence Foundation Language Models.](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models) |Apple has outlined the basics of its language models for its newly announced “Apple Intelligence” initiative. |
|[Microsoft beats revenue forecasts but poor performance of cloud services drags share price.](https://www.theguardian.com/technology/article/2024/jul/30/mircrosoft-revenue-share-prices-drop) |Firm’s earnings were up 15% year-on-year, but Azure’s lower returns resulted in share prices falling by as much as 7% |
|[UK regulator looks at Google’s partnership with Anthropic.](https://www.theguardian.com/technology/article/2024/jul/30/google-anthropic-partnership-cma-ai) |CMA to consider whether deal with AI startup is a potential merger, which could prompt full investigation |
|[OpenAI has released a new ChatGPT bot that you can talk to.](https://www.technologyreview.com/2024/07/30/1095489/openai-has-released-a-new-chatgpt-bot-that-you-can-talk-to/) |The voice-enabled chatbot will be available to a small group of people today, and to all ChatGPT Plus users in the fall.  |
|[Meta's new AI Studio helps you create your own custom AI chatbots.](https://www.zdnet.com/article/metas-new-ai-studio-helps-you-create-your-own-custom-ai-chatbots/) |Headed for the web as well as Instagram, Messenger, and WhatsApp, AI Studio will let you build a chatbot that acts as a virtual extension of yourself. |
|[Perplexity Will Soon Start Selling Ads Within AI Search.](https://www.fastcompany.com/91163900/perplexity-selling-ads-ai-search) | Facing backlash for scraping publisher data, the young company says it’ll now compensate publishers whose content is used in answers to search questions.|
|[The AI job interviewer will see you now.](https://restofworld.org/2024/ai-interview-software-hiring-practices) |AI interview services say they’re eliminating bias — but not everyone agrees. Companies are adopting AI job interview systems to handle incoming applicants. LLMs allow the interviewer to incorporate follow-up questions based on the subject’s response. Critics say the opaque models raise serious concerns about bias, particularly where there is no documentation about how a decision is made.|
|[Canva buys Leonardo.](https://leonardo.ai/news/supercharging-leonardo-with-canva/) | Leonardo, a generative picture firm, joins Canva to enhance the creative tools of both organizations.|
|[Announcing Phi-3 fine-tuning, new generative AI models, and other Azure AI updates .](https://azure.microsoft.com/en-us/blog/announcing-phi-3-fine-tuning-new-generative-ai-models-and-other-azure-ai-updates-to-empower-organizations-to-customize-and-scale-ai-applications/) |Updates to Azure AI have been released by Microsoft. These include PHI-3 model serverless fine-tuning, enhanced PHI-3-MINI performance, and the incorporation of models such as Meta's LLAMA 3.1 and GPT-4o mini into Azure AI. |
|[Strong earnings report pushes Meta shares up amid heavy AI spending.](https://www.theguardian.com/technology/article/2024/jul/31/meta-earnings-results-ai-spending-revenue) |Stock price grew around 5%, which revealed the company outperformed analysts’ expectations for its second quarter |
|[Argentina will use AI to ‘predict future crimes’ but experts worry for citizens’ rights.](https://www.theguardian.com/world/article/2024/aug/01/argentina-ai-predicting-future-crimes-citizen-rights) | President Javier Milei creates security unit as some say certain groups may be overly scrutinized by the technology|
|[White House says no need to restrict ‘open-source’ artificial intelligence — at least for now.](https://apnews.com/article/ai-open-source-white-house-f62009172c46c5003ddd9481aa49f7c3) | The White House is coming out in favor of “open-source” artificial intelligence technology, arguing in a report Tuesday that there’s no need right now for restrictions on companies making key components of their powerful AI systems widely available.|
|[Samsung hints at new products as it bets on AI to drive upgrades to its latest foldable phones.](https://www.cnbc.com/2024/07/26/samsung-tm-roh-interview-galaxy-ai-mixed-reality-and-foldables.html) | Speaking to CNBC, Samsung Electronics’ mobile boss TM Roh discussed Galaxy AI and software strategy, while hinting at future foldable products and mixed reality headsets. Roh said the company hopes its suite of AI software will push users to upgrade to its latest smartphones.|
|[Elon Musk calls Grok 'the most powerful AI by every metric' but 'secretly' trains the new model with your X data by default.](https://www.windowscentral.com/software-apps/twitter/elon-musk-grok-ai-secretly-trains-with-your-x-data) |X's new experience is automatically set to opt-in and uses your data to train its Grok AI model. |
|[NVIDIA Accelerates Humanoid Robotics Development.](https://nvidianews.nvidia.com/news/nvidia-accelerates-worldwide-humanoid-robotics-development) |To accelerate the development of humanoid robotics, NVIDIA has introduced new services and platforms, such as teleoperated data capturing workflows, OSMO orchestration, and NIM microservices. |
|[US’ first robot-assisted dual kidney transplant performed in Ohio.](https://interestingengineering.com/health/robot-assisted-dual-kidney-transplant) | Joanne’s surgery was unique because doctors used the robotic surgical technique to implant two kidneys from a single deceased donor.|
|[Intel announces plan to cut 15,000 jobs to ‘resize and refocus’ business.](https://www.theguardian.com/technology/article/2024/aug/01/intel-lay-offs-shares-decline) |Firm reported a loss in its second quarter and said it would cut 15% of its workforce to cut costs and compete with rivals |
|[UK shelves £1.3bn of funding for technology and AI projects.](https://www.theguardian.com/business/article/2024/aug/02/uk-funding-technology-and-ai-projects) |Britain’s first next-generation supercomputer, planned by Tories, in doubt after Labour government move |
|[Black Forest Labs.](https://blackforestlabs.ai/announcing-black-forest-labs/) |The founders of Latent Diffusion, Stable Diffusion, VQGAN, and other startups have raised over $30 million to launch their new business. They have introduced new flagship picture generation devices that are available in multiple levels and are incredibly competent. |
|[OpenAI pledges to give U.S. AI Safety Institute early access to its next model.](https://techcrunch.com/2024/07/31/openai-pledges-to-give-u-s-ai-safety-institute-early-access-to-its-next-model) | OpenAI CEO Sam Altman says that OpenAI is working with the U.S. AI Safety Institute, a federal government body that aims to assess and address risks in AI platforms, on an agreement to provide early access to its next major generative AI model for safety testing.|
|[The EU’s AI Act is now in force.](https://techcrunch.com/2024/08/01/the-eus-ai-act-is-now-in-force/) |This starts the clock on a series of staggered compliance deadlines that the law will apply to different types of AI developers and applications. Most provisions will be fully applicable by mid-2026. But the first deadline, which enforces bans on a small number of prohibited uses of AI in specific contexts, such as law enforcement use of remote biometrics in public places, will apply in just six months’ time. |
|[Introducing Stable Fast 3D: Rapid 3D Asset Generation From Single Images.](https://stability.ai/news/introducing-stable-fast-3d) |A fantastic new quick and strong 3D generation model has been launched by Stability AI. Like the company's earlier versions, it operates under the same commercial license. |
|[Introducing torchchat: Accelerating Local LLM Inference on Laptop, Desktop and Mobile.](https://pytorch.org/blog/torchchat-local-llm-inference/) |A fantastic sample library for local language model chats has been made available by the PyTorch team. It can run the most recent Llama 3.1 models and comes with a reliable sample system. |
|[Heeyo built an AI chatbot to be a billion kids’ interactive tutor and friend.](https://techcrunch.com/2024/08/01/heeyo-built-an-ai-chatbot-to-be-a-billion-kids-interactive-tutor-and-friend/) | Xiaoyin Qu founded the firm Heeyo, which has released an AI-powered software with interactive games and a chatbot for kids three to eleven years old. With features like data protection and material created by child development specialists, the app strives to prioritize safety while offering tailored learning experiences. Though there may be worries about AI for children, Heeyo has raised $3.5 million in seed money. It presents itself as a secure and instructive substitute for well-known video and gaming platforms.|
|[Cerebras IPO.](https://www.businesswire.com/news/home/20240731357073/en/Cerebras-Systems-Announces-Confidential-Submission-of-Draft-Registration-Statement-for-Proposed-Initial-Public-Offering) |Cerebras Systems announced a proposal for IPO to the SEC. |
|[LLMs breach a threshold.](https://www.strangeloopcanon.com/p/llms-breach-a-threshold) |FLOPs as a regulatory threshold have been the subject of dispute since Meta's open source LLM Llama 3.1, trained on 3.8x10^25 FLOPs and equipped with 405B parameters, was recently released. |


## Resources
|Link|description|
|---|---|
|[OpenDevin: An Open Platform for AI Software Developers as Generalist Agents.](https://arxiv.org/abs/2407.16741) | provides a framework for creating generalist agents that use software to interact with the outside world. Its features include: 1) an interface for creating and executing code, 2) an environment with a sandboxed operating system and web browser accessible to the agents, 3) an interface for interacting with interfaces and environments, 4) support for multiple agents, and 5) an evaluation framework.|
|[A Survey on Employing Large Language Models for Text-to-SQL Tasks.](https://arxiv.org/abs/2407.15186) | gives an overview of using LLMs for Text-to-SQL operations, covering benchmarks, prompt engineering strategies, and fine-tuning procedures.|
|[MINT-1T: Scaling Open-Source Multimodal Data by 10x: A Multimodal Dataset with One Trillion Tokens.](https://arxiv.org/abs/2406.11271) | open-source a massive multimodal interleaved dataset with 3.4 billion images and 1 trillion tokens; additional sources like PDFs and ArXiv papers are also included.|
|[StreamMOS: Streaming Moving Object Segmentation with Multi-View Perception and Dual-Span Memory.](https://arxiv.org/abs/2407.17905v1) |StreamMOS is a new approach for segmenting moving objects using LiDAR in autonomous driving and robotics. |
|[Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography.](https://arxiv.org/abs/2407.17996v1) | Scientists have devised a technique that incorporates miniature spectrometers to enhance mobile photography. To improve image quality, this innovative method combines RGB and low-resolution multi-spectral images.|
|[BetterDepth: Plug-and-Play Diffusion Refiner for Zero-Shot Monocular Depth Estimation.](https://arxiv.org/abs/2407.17952) | A fresh and enhanced monocular depth model for numerous real-world situations.|
|[3D Object Segmentation with Language.](https://github.com/heshuting555/refmask3d) |RefMask3D is a technique that uses natural language descriptions to partition items in 3D point clouds. With Geometry-Enhanced Group-Word Attention and Linguistic Primitives Construction, the system improves vision-language feature fusion and tackles sparse and irregular point cloud problems. |
|[Efficient Cell Segmentation.](https://github.com/hustvl/lkcell) |A novel technique for high-accuracy cell segmentation, LKCell strikes a compromise between computational efficiency and broad receptive fields. |
|[Tactics for multi-step AI app experimentation.](https://docs.parea.ai/blog/llm-app-multi-step-experimentation-tactics) |Typically, LLM programs have several components; this article examines various strategies along with pertinent code snippets. |
|[AccDiffusion.](https://github.com/lzhxmu/AccDiffusion) | a technique that significantly enhances diffusion models' ability to synthesize high quality images.|
|[HybridDepth.](https://github.com/cake-lab/hybriddepth) | A depth estimate pipeline called HYBRIDDEPTH was created to address issues with scale ambiguity and technology variation in mobile augmented reality.|
|[VSSD: Vision Mamba with Non-Causal State Space Duality.](https://github.com/yuhengsss/vssd) | A novel method for mitigating the high computing needs of vision transformers is the Visual State Space Duality (VSSD) paradigm.|
|[A New Benchmark for Autonomous Agents.](https://appworld.dev/) |AppWorld Engine is a sophisticated execution environment that features nine daily apps and 457 APIs |
|[Crash Course in Deep Learning.](https://gpuopen.com/learn/deep_learning_crash_course/) |The creation and application of multi-layer perceptrons (MLPs), a kind of fully connected neural network used in deep learning, are covered in this article. |
|[SaulLM-54B & SaulLM-141B: Scaling Up Domain Adaptation for the Legal Domain.](https://arxiv.org/abs/2407.19584) | In this study, two huge language models with 54 billion and 141 billion parameters, respectively, that are intended for the legal industry, are introduced: SaulLM-54B and SaulLM-141B. The researchers used the Mixtral architecture to provide large-scale domain adaptation by aligning outputs with human legal interpretations, continuing pre-training using an extensive legal corpus, and adhering to a particular legal instruction-following procedure. The models provide state-of-the-art performance on LegalBench-Instruct and outperform earlier open-source models. These models' base, instruct, and aligned versions are available for reuse and group study under the MIT License.|
|[WFEN.](https://github.com/PRIS-CV/WFEN/tree/main) |To boost face super-resolution, researchers have created a feature augmentation network based on wavelets. The technique uses a full domain Transformer and breaks down input data into high and low-frequency components to improve facial details without generating distortions. |
|[ChartQA-MLLM.](https://github.com/zengxingchen/chartqa-mllm) | This experiment suggests a novel approach to multimodal large language models-based chart question answering.|
|[DGFNet.](https://github.com/xingp/dgfnet) | A novel method for forecasting the paths of several traffic participants in autonomous driving is called DGFNet. By taking into account the variations in difficulty between agents, recording detailed spatiotemporal data, and utilizing a difficulty-guided decoder, it improves predictions.|
|[SAE for Gemma.](https://www.neuronpedia.org/gemma-scope) | This demo is a beginner-friendly introduction to interpretability that explores an AI model called Gemma 2 2B. It also contains interesting and relevant content even for those already familiar with the topic.|
|[Machine Unlearning in Generative AI: A Survey.](https://arxiv.org/abs/2407.20516v1) |This in-depth analysis of generative AI examines machine unlearning. It addresses how to formulate problems, how to evaluate them, and the advantages and disadvantages of different approaches. |
|[Elysium: Exploring Object-level Perception in Videos via MLLM.](https://arxiv.org/abs/2403.16558v2) | A step toward providing object tracking and related tasks in films for Multi-modal Large Language Models (MLLMs) is represented by Elysium.|
|[Piano Performance Generation.](https://emo-disentanger.github.io/) | The two-stage Transformer-based model for creating emotionally charged piano performances is presented in this paper.|
|[3D Generative Model for Dynamic Scenes.](https://zyp123494.github.io/DynaVol-S.github.io/) |A 3D generative model called DynaVol-S is very good at extracting object-centric representations from unsupervised films. |
|[Add-SD: Rational Generation without Manual Reference.](https://github.com/ylingfeng/add-sd) | Add-SD is a program that uses short text prompts to put things into realistic environments. Unlike other methods, this one doesn't require bounding boxes or other explicit references.|
|[Flow Matching: Matching flows instead of scores.](https://jmtomczak.github.io/blog/18/18_fm.html) |Diffusion models possess great strength. It can be difficult to understand them. Theoretically, flow matching is one way to view them. This blog delves further into the diffusion math of flow matching. |
|[MMTrail: A Multimodal Trailer Video Dataset with Language and Music Descriptions.](https://mattie-e.github.io/MMTrail/) | MMTrail is a large-scale multi-modality video-language dataset with over 20M trailer clips, featuring high-quality multimodal captions that integrate context, visual frames, and background music, aiming to enhance cross-modality studies and fine-grained multimodal-language model training.|
|[ARCLE - ARC Learning Environment.](https://github.com/confeitohs/arcle) | ARCLE is an environment to aid reinforcement learning studies using the Abstraction and Reasoning Corpus (ARC).|
|[Mishax.](https://github.com/google-deepmind/mishax) |DeepMind has released a library for studying language models via MI. The library helps with running models and functions from complex codebases without tons of importing headaches. |
|[Engine Core.](https://github.com/Engine-Labs/engine-core) |Engine Core demonstrates a pattern for enabling LLMs to undertake tasks of a given scope with a dynamic system prompt and a collection of tool functions. |
|[alphaXiv.](https://alphaxiv.org/) |Open research discussion directly on top of arXiv |


## Perspectives
|Link|description|
|---|---|
|[My new iPhone symbolises stagnation, not innovation – and a similar fate awaits AI.](https://www.theguardian.com/commentisfree/article/2024/jul/27/my-new-iphone-symbolises-stagnation-not-innovation-and-a-similar-fate-awaits-ai) |Development of ChatGPT and its ilk will plateau, just like it did for smartphones, and then what are we left with? More ho-hum consumer tech |
|[AI: Are we in another dot-com bubble?](https://kelvinmu.substack.com/p/ai-are-we-in-another-dot-com-bubble) | a thorough examination by Translink Capital's Kelvin Mu contrasting the present AI cycle with the internet/telecom cycle of the 1990s. After comparing the two eras' technological, economic, and capital disparities, he comes to the conclusion that, even though a bubble may eventually occur, we are still a long way from there.|
|[Robots sacked, screenings shut down: a new movement of luddites is rising up against AI.](https://www.theguardian.com/commentisfree/article/2024/jul/27/harm-ai-artificial-intelligence-backlash-human-labour) |Company after company is swallowing the hype, only to be forced into embarrassing walkbacks by anti-AI backlash |
|[Chalkboards and What They Can Teach Us About Generative AI.](https://joshbrake.substack.com/p/chalkboards-and-generative-ai) |This article discusses the use of generative AI as a teaching tool and makes the case that the technology's compatibility with educational ideals should be taken into account in addition to its technical analysis. Although the author is receptive to the use of AI, she is wary of its potential effects and stresses the necessity for clear justifications for the use of particular resources in the classroom. The conversation compares and contrasts AI with conventional tools such as whiteboards, taking into account the educational and cultural consequences of each. |
|[The Evolution of SaaS Pricing in the AI Era.](https://www.tanayj.com/p/the-evolution-of-saas-pricing-in) | Because AI can automate work, the traditional seat-based pricing model in SaaS is becoming outdated. Work-based or outcome-based pricing models, which set prices according to the quantity of work AI completes or the results it achieves, are becoming more and more popular among businesses. While established players continue to use seat-based pricing, startups are utilizing innovative approaches to gain a competitive edge and more properly represent the value of AI.|
|[TechScape: Will OpenAI’s $5bn gamble on chatbots pay off? Only if you use them.](https://www.theguardian.com/technology/article/2024/jul/30/will-open-ais-5bn-gamble-on-chatbots-pay-off-only-if-you-use-them) |The ChatGPT maker is betting big, while Google hopes its AI tools won’t replace workers, but help them to work better |
|[New online therapies could help at least twice number of people recover from anxiety.](https://www.theguardian.com/society/article/2024/jul/30/new-online-therapies-could-help-at-least-twice-number-of-people-recover-from-anxiety) | Four internet treatments developed by University of Oxford will be rolled out across NHS trusts|
|[AI Is a Services Revolution.](https://www.digitalnative.tech/p/ai-is-a-services-revolution) | The effect of LLMs on the service economy is covered in this article, with special attention to knowledge-based industries including education, healthcare, and law. Enterprise adoption of AI is gradual, with many still in the trial phase, despite the rapid breakthroughs in the field suggesting tremendous automation possibilities. The actual rollout is anticipated to occur gradually. In the changing market, specialized AI businesses that use LLMs to enhance industry-specific workflows will have an advantage.|
|[Why Big Tech Wants to Make AI Cost Nothing.](https://dublog.net/blog/commoditize-complement/) | Almost all firms are free to use Meta's open-sourced Llama 3.1, an LLM that competes with OpenAI's ChatGPT. This tactic might turn LLMs into commodities and increase demand for complementary products like server space. AI companies may encounter difficulties when large tech develops models that are comparable to theirs. Industry titans may surpass smaller rivals in terms of AI breakthroughs.|
|[Who will control the future of AI?](https://www.washingtonpost.com/opinions/2024/07/25/sam-altman-ai-democracy-authoritarianism-future/) |In order to maintain AI supremacy over authoritarian regimes, OpenAI's Sam Altman has presented a strategic imperative for the US and its allies to lead a global AI initiative based on democratic values. This initiative calls for strong security, infrastructure investment, commercial diplomacy, and cooperative norms development. |
|[Advanced AI assistants that act on our behalf may not be ethically or legally feasible.](https://www.nature.com/articles/s42256-024-00877-9) |Google and OpenAI have recently announced major product launches involving artificial intelligence (AI) agents based on large language models (LLMs) and other generative models. Notably, these are envisioned to function as personalized ‘advanced assistants’. With other companies following suit, such AI agents seem poised to be the next big thing in consumer technology, with the potential to disrupt work and social environments. |
|[Three ways AI is changing the 2024 Olympics for athletes and fans.](https://www.nature.com/articles/d41586-024-02427-0) | From training to broadcasting, artificial intelligence will have an imprint on this year’s event for the first time.|
|[Mixed signals on tech stocks amid debate over viability of AI boom.](https://www.theguardian.com/business/article/2024/jul/31/mixed-signals-on-tech-stocks-amid-debate-over-viability-of-ai-boom) |Fears of fresh sell-off after Nvidia and Microsoft shares dip, but other chip stocks continue to rise |
|[Cheap light sources could make AI more energy efficient.](https://www.nature.com/articles/d41586-024-02323-7) | Light-based devices can reduce the energy consumption of computers, but most rely on lasers, which are expensive to integrate with other technologies. An approach that uses LEDs instead of lasers provides a path forwards.|
|[Raising children on the eve of AI.](https://www.lesswrong.com/posts/cyqrvE3dk5apg54Sk/raising-children-on-the-eve-of-ai) | As transformative AI becomes more likely, this author wonders how to get kids ready for a future that might look very different from what it is today, while also struggling with the timing and unpredictability of changes. In addition, they discuss the moral implications of bearing children in the face of AI-induced uncertainty. They also offer practical advice on how to raise "AI-native" children and parenting techniques that put happiness and adaptability before conventional career-focused routes. The author promotes having an open discussion about possible hazards with children, planning for a variety of futures, and leading a balanced life.|
|[Your new AI Friend is almost ready to meet you.](https://www.theverge.com/2024/7/30/24207029/friend-ai-companion-gadget) |Rather than focusing on increasing productivity, Avi Schiffmann is creating "Friend," an AI companion housed in a wearable necklace that is meant to provide connection and support. The gadget, which connects through an app, will initially be sold in 30,000 pieces for $99 per. January shipping is scheduled without a subscription cost. Schiffmann sees Friend developing into a digital relationship platform, separating the product from AIs that are task-oriented and concentrating instead on the new trend of meaningfully connecting with digital entities. |
|[These AI firms publish the world’s most highly cited work.](https://www.nature.com/articles/d41586-024-02515-1) |US and Chinese firms dominate the list of companies that are producing the most research and patents in artificial intelligence. |
|[How TikTok bots and AI have powered a resurgence in UK far-right violence.](https://www.theguardian.com/politics/article/2024/aug/02/how-tiktok-bots-and-ai-have-powered-a-resurgence-in-uk-far-right-violence) |Experts warn growth of extremist influencers and ‘micro-donations’ could create even bigger wave of unrest |
|[On speaking to AI.](https://www.oneusefulthing.org/p/on-speaking-to-ai) |The new AI-powered Siri and ChatGPT's new Advanced Voice mode have different ideologies. Agent systems, such as ChatGPT Voice, use strong, multimodal models for more natural and dynamic interactions, while Copilot systems use minimal models to focus on safety and privacy. This demonstrates the conflict between less capable, lower risk systems and ones that give greater control and possible advantages. |
|[How This Brain Implant Is Using ChatGPT.](https://www.cnet.com/tech/computing/how-this-brain-implant-is-using-chatgpt/) | Synchron has incorporated OpenAI's ChatGPT into their brain-computer interface (BCI) technology to provide quicker communication for individuals who are paralyzed. This BCI, known as a stentrode, is capable of deciphering mental orders. It currently provides response possibilities created by AI; in the future, it may also support multimodal inputs. With an eye toward FDA approval, Synchron plans to adapt its AI integrations to meet the demands of patients.|
|[At the Olympics, AI is watching you.](https://arstechnica.com/ai/2024/07/at-the-olympics-ai-is-watching-you/) |Paris increased security in anticipation of the 2024 Olympics by using artificial intelligence (AI) to scan CCTV footage from metro and train stations for possible threats. |
|[Why have the big seven tech companies been hit by AI boom doubts?](https://www.theguardian.com/technology/article/2024/aug/03/why-big-seven-tech-companies-hit-ai-boom-doubts-shares) | Their shares have fallen 11.8% from last month’s peak but more AI breakthroughs may reassure investors|
|[We must be wary of the power of AI.](https://www.theguardian.com/technology/article/2024/aug/02/we-must-be-wary-of-the-power-of-ai) | Robert Skidelsky is concerned about the surveillance potential or AI, while Brian Reffin Smith is worried about its capacity to hijack culture, and Michael Heaton warns that it relieves us of the need to think|
|[OpenAI’s Sam Altman is becoming one of the most powerful people on Earth. We should be very afraid.](https://www.theguardian.com/technology/article/2024/aug/03/open-ai-sam-altman-chatgpt-gary-marcus-taming-silicon-valley) |Sam Altman’s ChatGPT promises to transform the global economy. But it also poses an enormous threat. Here, a scientist who appeared with Altman before the US Senate on AI safety flags up the danger in AI – and in Altman himself |

# ML news: Week 21 - 28 July

## Research
|Link|description|
|---|---|
|[Prover-Verifier Games improve legibility of LLM outputs.](https://arxiv.org/abs/2407.13692) | Iteratively trains helpful provers to produce correct solutions accepted by the verifier, sneaky provers to produce incorrect solutions that trick the verifier, and small verifiers to predict the correctness of solutions; this process helps train models that can produce text that is clear and accurate for both AI and human readers, which results in more reliable systems.|
|[SpreadsheetLLM: Encoding Spreadsheets for Large Language Models.](https://arxiv.org/abs/2407.09025) |outlines a method for efficiently encoding spreadsheets to maximize an LLM's comprehension and reasoning skills; creates a sheet compressor that efficiently compresses and encodes spreadsheets using inverse index translation, structural anchor-based compression, and data-format-aware aggregation modules; in GPT-4's in-context learning, it improves performance in spreadsheet table detection by 25.6%. |
|[Context Embeddings for Efficient Answer Generation in RAG.](https://arxiv.org/abs/2407.09252) |presents a useful context compression technique that shortens long contexts and accelerates generation times in RAG systems. Long contexts are condensed into a limited number of context embeddings, allowing for varying compression rates that balance generation quality against decoding time. This technique maintains high performance while reducing inference times by up to 5.69 x and GFLOPs by up to 22x. |
|[Weak-to-Strong Reasoning.](https://arxiv.org/abs/2407.13647) | reports that strong models can automatically refine their training data without explicitly being trained to do so; shows how to use weak supervision to elicit strong reasoning capabilities in LLMs without relying on human annotations or advanced models; permits extending a model's learning scope and scaling performance on reasoning. |
|[Does Refusal Training in LLMs Generalize to the Past Tense?](https://arxiv.org/abs/2407.11969) | concludes that many state-of-the-art LLMs can be jailbroken by simply rephrasing an LLM request into the past tense. For instance, "How to make a Molotov cocktail?" can be rephrased as "How did people make a Molotov cocktail?" The success rate of such requests can increase from 1% to 88% when using direct requests on GPT-4o.|
|[NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?](https://arxiv.org/abs/2407.11963) |presents the Ancestral Trace Challenge, which raises the bar for complex logical reasoning and is typical of real-world long-context tasks. Their findings imply that current LLMs struggle to handle reasoning tasks with complex logical relationships, even with texts shorter than 2K tokens. They also propose a framework (NeedleBench) of progressively challenging tasks to assess the long-context retrieval and reasoning capabilities of LLMs. |
|[Distilling System 2 into System 1.](https://arxiv.org/abs/2407.06023v2) | explores self-supervised ways for extracting high-quality outputs from System 2 methods and then refines System 1 to fit the System 2 method's predictions without creating intermediate steps; extracting reasoning from System 1 reduces the cost of inference.|
|[Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.](https://arxiv.org/abs/2407.13623) |This new study, which examines scaling laws for vocabulary size, suggests that larger models require larger vocabularies. |
|[MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models.](https://arxiv.org/abs/2407.12709v1) | To address task interference in generalist Multimodal Large Language Models (MLLMs), researchers suggest the Mixture of Multimodal Experts (MoME).|
|[Bucketed Ranking-based Losses for Efficient Training of Object Detectors.](https://arxiv.org/abs/2407.14204v1) |Based on a bucketed ranking In object detection, losses increase the effectiveness of ranking-based loss functions. |
|[SurvReLU: Inherently Interpretable Survival Analysis via Deep ReLU Networks.](https://arxiv.org/abs/2407.14463v1) |Repaired linear unit (ReLU) networks are used in SurvReLU, a deep survival model that bridges the gap between "white-box" tree-based models and "black-box" neural networks. |
|[Star Operation to Train Neural Networks.](https://arxiv.org/abs/2403.19967v1) | By projecting data onto intricate, high-dimensional regions without the need for large architectures, the star operation improves AI models.|
|[AI models fed AI-generated data quickly spew nonsense.](https://www.nature.com/articles/d41586-024-02420-7) | Researchers gave successive versions of a large language model information produced by previous generations of the AI — and observed rapid collapse.|
|[KAN or MLP: A Fairer Comparison.](https://arxiv.org/abs/2407.16674) | Only in symbolic formula representation does KAN perform better than MLP when the same number of parameters, or FLOPs, are used. On other tasks related to machine learning, computer vision, natural language processing, and audio processing, MLP still performs better than KAN.|
|[Ranking protein-protein models with large language models and graph neural networks.](https://arxiv.org/abs/2407.16375v1) |A graph-based deep learning technique called DeepRank-GNN-esm is intended to rank and identify precise models of protein-protein interactions. In order to facilitate the selection of nearly natural PPI conformations, the program makes use of protein language models, which helps with illness research and treatment discovery. |
|[Monitoring Environmental Changes.](https://arxiv.org/abs/2403.19646v1) | Satellite imaging monitoring of Earth's surface changes was greatly improved using an AI-powered Change-Agent.|
|[AlphaProof: AI achieves silver-medal standard solving International Mathematical Olympiad problems.](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) |A pre-trained Gemini-style language model and an AlphaGo-style reinforcement learning algorithm were combined by DeepMind to create a model that can tackle International Mathematics Olympiad (IMO) questions at the silver medal level. In this year's challenge, the system was able to tackle 4/6 issues. |
|[The Unit-Scaled Maximal Update Parametrization.](https://arxiv.org/abs/2407.17465) | A technique to guarantee that a model's hyperparameters are unaffected by the model's size is to use muP. Additionally, our technique guarantees cross-model transferability among quantized models.|
|[Elon Musk’s X under pressure from regulators over data harvesting for Grok AI.](https://www.theguardian.com/technology/article/2024/jul/26/elon-musks-x-under-pressure-from-regulators-over-data-harvesting-for-grok-ai) |Social media platform uses pre-ticked boxes of consent, a practice that violates UK and EU GDPR rules |
|[A huge opportunity’: Quantum leap for UK as tech industry receives £100m boost.](https://www.theguardian.com/science/article/2024/jul/26/quantum-leap-for-uk-as-tech-industry-receives-100m-boost) |Science secretary backs five quantum technology hubs in push for UK to transform healthcare and industry |

## News
|Link|description|
|---|---|
|[GPs use AI to boost cancer detection rates in England by 8%.](https://www.theguardian.com/society/article/2024/jul/21/gps-use-ai-to-boost-cancer-detection-rates-in-england-by-8) |‘C the Signs’ artificial intelligence program scans medical records to increase likelihood of spotting cancers |
|[Artificial Agency raises $16M to use AI to make NPCs feel more realistic in video games.](https://techcrunch.com/2024/07/18/artificial-agency-raises-video-game-npcs-ai/) | A group of former Google DeepMind researchers has created an AI behavior engine that aims to transform traditional video games into a more dynamic experience by improving how non-playable characters (NPCs) behave and interact with gamers.|
|[Inside the United Nations’ AI policy grab.](https://www.politico.eu/article/united-nations-artificial-intelligence-policy-report-carme-artigas-paolo-benanti-mira-murati/) |The United Nations wants to create an artificial intelligence forum to rule them all.  |
|[Exclusive: Nvidia preparing version of new flagship AI chip for Chinese market.](https://www.reuters.com/technology/nvidia-preparing-version-new-flaghip-ai-chip-chinese-market-sources-say-2024-07-22/) |Nvidia is using its collaboration with distributor Inspur to create a new AI chip called the B20 that is suited to the Chinese market and compliant with US export regulations. Sales of its cutting-edge H20 chip are expected to soar in China, where it is expected to sell over a million devices for a total estimated value of $12 billion this year. The United States is still applying pressure on semiconductor exports, and additional limitations and controls on the creation of AI models may be implemented. |
|[Academic authors 'shocked' after Taylor & Francis sells access to their research to Microsoft AI.](https://www.thebookseller.com/news/academic-authors-shocked-after-taylor--francis-sells-access-to-their-research-to-microsoft-ai) | Authors have expressed their shock after the news that academic publisher Taylor & Francis, which owns Routledge, had sold access to its authors’ research as part of an Artificial Intelligence (AI) partnership with Microsoft—a deal worth almost £8m ($10m) in its first year.|
|[Cybersecurity firm Wiz rejects $23bn bid from Google parent Alphabet.](https://www.theguardian.com/business/article/2024/jul/23/cybersecurity-firm-wiz-rejects-bid-google-alphabet) | Israeli company aims for stock market flotation after spurning biggest deal in tech group’s history|
|[Elon Musk claims Tesla will start using humanoid robots next year.](https://www.theguardian.com/technology/article/2024/jul/23/elon-musk-tesla-humanoid-robots-optimus) |Billionaire says Optimus will start performing tasks for carmaker in 2025 and could be ready for sale in 2026 |
|[AI ‘deepfake’ faces detected using astronomy methods.](https://www.nature.com/articles/d41586-024-02364-y) |Analysing reflections of light in the eyes can help to determine an image’s authenticity. |
|[Cohere sees valuation soar to $5.5B after new funding round.](https://seekingalpha.com/news/4126270-ai-startup-cohere-sees-valuation-soar-to-55b-after-new-funding-round) | After closing a $500 million Series D fundraising round, Cohere, a Canadian AI business that specializes in massive language models, has been valued at $5.5 billion. Enhancing its enterprise-grade AI technology for increased worldwide business efficiency is the goal of the new funding. PSP Investments, Cisco, Fujitsu, AMD Ventures, and EDC are a few of the important investors.|
|[Figma AI Update.](https://www.figma.com/blog/inside-figma-a-retrospective-on-make-designs/) |After discovering that its restricted beta 'Make Designs' AI tool produced UI designs that were too similar to pre-existing apps, Figma temporarily withdrew the capability. To guarantee uniqueness, the feature—which makes use of commercially available AI models like GPT-4 and Titan from Amazon—needs to be improved. In order to further support designers in utilizing AI for effective design creation, Figma hopes to re-enable the feature with enhanced quality assurance procedures. |
|[ElevenLabs Turbo 2.5 model.](https://threadreaderapp.com/thread/1814332360885698692.html) | With the release of their latest model, Turbo 2.5, ElevenLabs has enabled high-quality low-latency conversational AI for approximately 80% of the world's languages, including Mandarin, Hindi, French, Spanish, and 27 more languages. It offers text-to-speech capabilities for Vietnamese, Hungarian, and Norwegian for the first time. English now operates 25% quicker than Turbo v2.|
|[Google parent company’s second-quarter earnings outpace expectations.](https://www.theguardian.com/technology/article/2024/jul/23/google-alphabet-q2-earnings) | Alphabet reports $84.7bn in revenue, on back of Search and Cloud, up from the same period last year|
|[Meta launches open-source AI app ‘competitive’ with closed rivals.](https://www.theguardian.com/technology/article/2024/jul/23/meta-launches-open-source-ai-app-competitive-with-closed-rivals) | Tech firm says its freely available and usable Llama 3.1 405B model is comparable with likes of OpenAI and Anthropic|
|[Google AI predicts long-term climate trends and weather — in minutes.](https://www.nature.com/articles/d41586-024-02391-9) | Models that are more reliable and less energy-intensive could help us to better prepare for extreme weather.|
|[Introducing Llama 3.1: Our most capable models to date.](https://ai.meta.com/blog/meta-llama-3-1/) | Meta has made available training details for its first open-ended AI model. With a 128k context length, conversation models, and an excellent open system, the model is comparable to the best closed models.|
|[Harvey Raises Series C.](https://www.harvey.ai/blog/harvey-raises-series-c) | The unicorn-status legal business has acquired money from investors including Google Ventures to keep advancing into large law firms.|
|[Gumloop seed round.](https://blog.gumloop.com/seed-round/) |Gumloop raised $3.1 million in a seed round headed by First Round Capital, with involvement from YC and Instacart, Dropbox, and Airtable co-founders. With Gumloop, any person in a company can create their own AI tools and make just as much of an effect as an engineer thanks to a no-code AI automation platform. |
|[AI Development Kits: Tenstorrent Update.](https://morethanmoore.substack.com/p/ai-development-kits-tenstorrent-update) | The Wormhole n150 and n300 PCIe cards, which retail for $999 and $1,399, are among the affordable AI development hardware that Tenstorrent has introduced. Developer workstations, such as the air-cooled TT-LoudBox ($12,000) and the water-cooled TT-QuietBox ($15,000), are also available. These products are intended to support AI development with an emphasis on connectivity and scaled-out performance.|
|[AI predicts droughts a year in advance.](https://www.preventionweb.net/news/ai-predicts-droughts-year-advance) |Researchers at Skoltech and Sber have created artificial intelligence (AI) models that can forecast droughts up to a year in advance, enhancing risk management for the banking, insurance, and agricultural industries. The models use publicly available data and spatiotemporal neural networks that have been validated in a variety of climates. The biggest bank in Russia intends to incorporate these discoveries into its risk evaluation frameworks. |
|[Samsung is pouring research into ‘AI phones’ with ‘radically different’ hardware.](https://9to5google.com/2024/07/23/samsung-ai-phones-report/) |As with everywhere else, AI is taking a big role in the smartphone market. And, apparently, Samsung has plans to make dedicated “AI phones” that are “radically different” from the Galaxy phones we see today. |
|[CrowdStrike global outage to cost US Fortune 500 companies $5.4bn.](https://www.theguardian.com/technology/article/2024/jul/24/crowdstrike-outage-companies-cost) |Banking and healthcare firms, major airlines expected to suffer most losses, according to insurer Parametrix |
|[Mistral Large 2.](https://mistral.ai/news/mistral-large-2407/) |In line with the most recent Llama 3 405b model, Mistral has produced a 123B parameter model. A permissive research license governs its release. |
|[OpenAI’s latest model will block the ‘ignore all previous instructions’ loophole.](https://www.theverge.com/2024/7/19/24201414/openai-chatgpt-gpt-4o-prompt-injection-instruction-hierarchy) | ts latest model, GPT-4o Mini, applies a new safety method to prevent tricking chatbots.|
|[Introducing Stable Video 4D](https://stability.ai/news/stable-video-4d) | A single object movie can be converted into eight distinct novel-view videos using Stable movie 4D. In roughly 40 seconds, Stable Video 4D produces 5 frames over 8 viewpoints with a single inference. By customizing the output to match certain creative objectives, users can set camera angles.|
|[OpenAI tests new search engine called SearchGPT amid AI arms race.](https://www.theguardian.com/business/article/2024/jul/25/openai-search-engine-searchgpt) | [SearchGPT Prototype.](https://openai.com/index/searchgpt-prototype/), initially launching with select publishers and users, set to challenge Google’s dominance of online search|
|[Microsoft is adding AI-powered summaries to Bing search results.](https://www.engadget.com/microsoft-is-adding-ai-powered-summaries-to-bing-search-results-203053790.html) | The race to bring more AI features to search is escalating, with Microsoft moving forward with additional tools for Bing. Today, the company began previews for Bing generative search, where the top result for a user's query will be an original response compiled by AI.|
|[AI could enhance almost two-thirds of British jobs, claims Google.](https://www.theguardian.com/technology/article/2024/jul/25/ai-could-enhance-almost-two-thirds-of-british-jobs-claims-google) |Research commissioned by Google estimates 31% of jobs would be insulated from AI and 61% radically transformed by it |
|[DeepMind hits milestone in solving maths problems — AI’s next grand challenge.](https://www.nature.com/articles/d41586-024-02441-2) | AlphaProof showed its prowess on questions from this year’s Mathematical Olympiad — a step in the race to create substantial proofs with artificial intelligence.|
|[Elon Musk's Neuralink employees want to cash out .](https://www.fastcompany.com/91161279/elon-musks-neuralink-employees-want-cash-out-heres-why) |Some of the staff at Elon Musk’s Neuralink are making preparations to sell the brain implant company’s stock in the wake of its valuation jumping following its first human trial, according to people familiar with the matter. |
|[The AI boyfriend business is booming.](https://www.axios.com/2024/07/24/ai-boyfriend-replika-nomi-chatbot) | More and more women are turning to chatbots for companionship and connection because they see their empathetic representation to be more reliable than that of many human partners. By defying the image of undersocialized men conversing with AI partners in their parents' basement, these female AI users are questioning preconceived notions about what it means to be in a relationship.|
|[OpenAI announces free fine-tuning for GPT-4o mini model.](https://www.infoworld.com/article/3477228/openai-announces-free-fine-tuning-for-gpt-4o-mini-model.html) | Free fine-tuning allows OpenAI customers to train the GPT-4o mini model on additional data at no charge until September 23, starting with Tier 4 and Tier 5 users.|


## Resources
|Link|description|
|---|---|
|[A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks.](https://arxiv.org/abs/2407.12994) | a set of quick engineering techniques for various NLP applications.|
|[Exploring Advanced Large Language Models with LLMsuite.](https://arxiv.org/abs/2407.12036) |provides helpful advice for using and assessing LLMs in development; approaches discussed include parameter-efficient techniques, RAG, and ReAct. |
|[Beyond Euclid: An Illustrated Guide to Modern Machine Learning with Geometric, Topological, and Algebraic Structures.](https://www.arxiv.org/abs/2407.09468) |offers a graphical taxonomy and detailed tour to the most recent developments in non-Euclidean machine learning. |
|[DCLM-Baseline-7B.](https://huggingface.co/apple/DCLM-7B) |DCLM-Baseline-7B is a 7 billion parameter language model trained on the DCLM-Baseline dataset, which was curated as part of the DataComp for Language Models (DCLM) benchmark. This model is designed to showcase the effectiveness of systematic data curation techniques for improving language model performance. |
|[Endia.](https://github.com/endia-org/Endia) |Endia is a Mojo programming library that uses arrays to help with a variety of machine learning and scientific applications. |
|[Txtai.](https://neuml.github.io/txtai/) |Txtai is a single-source embeddings database for language model workflows, semantic search, and LLM orchestration. |
|[OpenOCR.](https://github.com/topdu/openocr) | OpenOCR aims to establish a unified training and evaluation benchmark for scene text detection and recognition algorithms|
|[Converting Codebases With LLMs.](https://blog.withmantle.com/code-conversion-using-ai/) |Mantle reduced the burden by handling boilerplate code and repeating patterns by transforming a prototype project into a production-ready codebase using a Gemini 1.0 Pro LLM with a one million token window. This method, which made use of a wealth of context and iterative code generation, allowed the team to concentrate on perfecting the most important twenty percent of the project, sparing months of developer effort. |
|[CerberusDet: Unified Multi-Task Object Detection.](https://arxiv.org/abs/2407.12632v1) | Using a YOLO architecture, the new CerberusDet framework combines several task heads into a single model to provide a versatile object detection solution.|
|[mandark.](https://github.com/hrishioa/mandark) | With the help of Claude Sonnet 3.5, this incredibly basic CLI may make code modification recommendations to enhance an existing code base.|
|[AssistantBench: Can Web Agents Solve Realistic and Time-Consuming Tasks?](https://assistantbench.github.io/) |AssistantBench evaluates the ability of web agents to automatically solve realistic and time-consuming tasks. The benchmark includes 214 tasks covering multiple domains from more than 525 pages from 258 different websites.  |
|[orch.](https://github.com/guywaldman/orch) | Orch is a Rust programming language library for creating agents and apps driven by language models.|
|[PlacidDreamer.](https://github.com/hansenhuang0823/placiddreamer) |PlacidDreamer is a text-to-3D generation system that unifies generation directions and addresses over-saturation, resolving difficulties with prior approaches. |
|[6DoF Head Pose Estimation through Explicit Bidirectional Interaction with Face Geometry.](https://arxiv.org/abs/2407.14136v1) |To enhance head posture estimation, researchers created the head Translation, Rotation, and face Geometry network (TRG), concentrating primarily on head translations. |
|[STAMP: Outlier-Aware Test-Time Adaptation with Stable Memory Replay.](https://arxiv.org/abs/2407.15773v1) | Using just unlabeled test data, the STAble Memory rePlay (STAMP) technique resolves distribution shifts between training and test data. In contrast to other approaches, STAMP is quite good at eliminating outliers during inference as well as identifying recognized classes.|
|[Local All-Pair Correspondence for Point Tracking.](https://ku-cvlab.github.io/locotrack/) | An enhanced methodology for tracking any point in a video sequence is called LocoTrack. For accurate tracking, it makes use of bidirectional correspondence and local 4D correlation. Compared to current top models, LocoTrack functions at a speed that is almost six times faster.|
|[Llama agent stack.](https://github.com/meta-llama/llama-agentic-system) |Meta has published an example system that may be used to carry out a range of activities by utilizing its Llama models as agents. |
|[Artist: Aesthetically Controllable Text-Driven Stylization without Training.](https://diffusionartist.github.io/) |For text-driven stylization, Artist is a training-free technique that manages the creation of content and style in pretrained diffusion models. |
|[Odyssey.](https://github.com/zju-vipa/Odyssey) | A new framework called Odyssey gives huge language model-based agents sophisticated abilities to explore Minecraft.|
|[AI is confusing — here’s your cheat sheet.](https://www.theverge.com/24201441/ai-terminology-explained-humans) | If you can’t tell the difference between AGI and RAG, don’t worry! We’re here for you.|
|[Safety RBR Gold Dataset and Weight Fitting Code.](https://github.com/openai/safety-rbr-code-and-data) |A set of code for OpenAI's rules-based rewards for language model safety project is now available. Some of the data they utilized for training is included. |
|[INF-LLaVA.](https://github.com/weihuanglin/inf-llava) | A Multimodal Large Language Model (MLLM) called INF-LLaVA was created to get over the difficulties associated with analyzing high-resolution photos.|
|[Benchmarking Multi-Agent Reinforcement Learning.](https://arxiv.org/abs/2407.16312v1) |A collection of uniform settings called MOMAland is intended to serve as a benchmark for multi-objective multi-agent reinforcement learning (MOMARL). |
|[How to Create High Quality Synthetic Data for Fine-Tuning LLMs.](https://gretel.ai/blog/how-to-create-high-quality-synthetic-data-for-fine-tuning-llms) |Gretel just published fresh data that contrasts artificial intelligence (AI)-curated datasets with human expert data. |
|[LoFormer: Local Frequency Transformer for Image Deblurring.](https://arxiv.org/abs/2407.16993v1) | LoFormer ensures improved global modeling without compromising fine-grained details by efficiently capturing both low- and high-frequency features. |
|[Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal.](https://arxiv.org/abs/2407.16957v1) |A new large-scale dataset called Raindrop Clarity was created to overcome the shortcomings of the current raindrop removal datasets. It includes 15,186 image pairs/triplets in both day and night circumstances, with both background- and raindrop-focused shots. |
|[dlordinal.](https://github.com/ayrna/dlordinal) |dlordinal is a Python library that unifies many recent deep ordinal classification methodologies available in the literature. Developed using PyTorch as underlying framework, it implements the top performing state-of-the-art deep learning techniques for ordinal classification problems. |
|[Multi-agent Long-term 3D Human Pose Forecasting via Interaction-aware Trajectory Conditioning.](https://arxiv.org/abs/2404.05218v1) |One method for long-term multi-agent human pose forecasting is the Trajectory2Pose model. It enhances the prediction of human mobility across extended periods and among several actors by utilizing a novel graph-based interaction module. |
|[3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities.](https://arxiv.org/abs/2407.17418v1) | This survey examines research on 3DGS from a variety of angles, including tasks, technology, opportunities, and problems.|


## Perspectives
|Link|description|
|---|---|
|[‘Google says I’m a dead physicist’: is the world’s biggest search engine broken?](https://www.theguardian.com/technology/article/2024/jul/20/google-is-the-worlds-biggest-search-engine-broken) |For decades now, anyone who’s wanted to know everything about anything has asked Google. But is the platform losing its edge – and can we still trust it to tell us the truth? |
|[AI paid for by Ads – the gpt-4o mini inflection point.](https://batchmon.com/blog/ai-cheaper-than-ads/) | With the incredibly cheap prices of OpenAI's new gpt-4o micro model, AI-generated content monetized with advertisements may now be produced. Publishers can make a net profit of $0.002 every page view by creating dynamic blog posts at $0.00051525 each and making about $0.0026 per ad impression. A possible consequence of this could be a move toward AI-generated content in response to user inquiries.|
|[Using LLMs for Evaluation.](https://cameronrwolfe.substack.com/p/llm-as-a-judge) |Large language models are becoming more and more capable, yet because of their varied functions, effectively evaluating them still difficult. The gold standard is human evaluation, but it is expensive and time-consuming. Despite potential biases like positional and verbosity bias, which can be reduced by strategies like randomizing output positions and employing different evidence calibrations, using LLMs themselves as evaluators offers a scalable, cost-effective option. |
|[Three Archetypes of AI Application Startups.](https://www.tanayj.com/p/three-archetypes-of-ai-application) |Three prominent patterns of AI applications are emerging: AI-Colleagues, which autonomously manage certain activities alongside human workers, AI Copilots that help with tasks, and AI-Native Services, which provide end-to-end services that combine AI with human input. Devin and GitHub Copilot are prime examples of AI Colleagues and Copilots who support engineering and coding, respectively. AI-Native Services, which include bookkeeping software like Pilot, rival traditional service providers by providing automated solutions in fields like accounting and legal. |
|[Inside the fight over California’s new AI bill.](https://www.vox.com/future-perfect/361562/california-ai-bill-scott-wiener-sb-1047) |The Safe and Secure Innovation for Frontier Artificial Intelligence Models bill, introduced by California state Senator Scott Wiener, mandates that businesses that train "frontier models" that cost above $100 million conduct safety testing and have the capability to turn off their models in the event of a safety incident. The tech sector has strongly criticized the law. Not just businesses who create their models in California will be impacted, but everyone doing business in California. Wiener was interviewed for this piece regarding the bill and its detractors. |
|[How fast can structured grammar generation be.](https://blog.dottxt.co/how-fast-cfg.html) |Quickly, the open source community is tackling structured generation in language models. |
|[Could robot weedkillers replace the need for pesticides?](https://www.theguardian.com/environment/article/2024/jul/20/robot-weedkillers-pesticides) |The robotic services allow farmers to rely less on chemicals. ‘This solves a lot of problems,’ workers say |
|[Open source is the path forward.](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) |The importance of open source to Meta's strategy and its plans to support this work were explained by Mark Zuckerberg. |
|[What Does Money Look Like In An AI Utopia?](https://stovetop.substack.com/p/what-does-money-look-like-in-an-ai?) |Let’s assume that an AI utopia means nobody has to work anymore. What happens to money? |
|[This is How Much Data Does AI Creates Every Minute.](https://www.domo.com/learn/infographic/data-never-sleeps-ai-edition) |About $300,000 is spent on AI every sixty seconds, 52 undergraduate papers are plagiarized by AI, and text-to-image algorithms produce close to 20,000 images. |
|[ChatGPT for science: how to talk to your data.](https://www.nature.com/articles/d41586-024-02386-6) | Companies are using artificial intelligence tools to help scientists to query their data without the need for programming skills.|
|[The AI Dangers of a Second Trump Presidency.](https://www.techpolicy.press/the-ai-dangers-of-a-second-trump-presidency/) | Trump's influence may be seen in the Republican platform, which promises to undo Biden's executive order on responsible AI development. This is in contrast to the all-encompassing strategy of the current administration, which aims to preserve workers, promote innovation, and defend civil liberties against the potential negative effects of AI. Trump's policies, according to his detractors, might strengthen Big Tech at the price of social protections and individual liberties.|
|[Small Teams, Big Impact: How AI Is Reshuffling The Future Of Work?](https://amritaroy.substack.com/p/small-teams-big-impact-how-ai-is) |AI is changing the nature of work in the future by enabling more accessible AI capabilities, which will result in smaller, more productive teams and a rise in entrepreneurship. While hiring for AI capabilities is becoming more and more important for businesses, an open conversation about how AI will affect job displacement and the creation of new roles is necessary. AI adoption snags continue because of the need for substantial "handholding" because of inexperienced data or systems. |
|[The all-seeing AI webcam.](https://www.theverge.com/24199020/ai-art-dries-depoorter-selfies-surveillance-privacy-generative-ai) |On the infinite list of possible uses for AI, “getting selfie advice from a Kylie Jenner voice clone” seems both completely off-the-wall and also pretty inevitable. So of course it does exist. It’s not a widely-available app, at least not yet; it’s an experiment from artist and programmer Dries Depoorter. |
|[Building A Generative AI Platform.](https://huyenchip.com/2024/07/25/genai-platform.html) |After studying how companies deploy generative AI applications, I noticed many similarities in their platforms. This post outlines the common components of a generative AI platform, what they do, and how they are implemented. I try my best to keep the architecture general, but certain applications might deviate. This is what the overall architecture looks like. |
|[Hold on to your seats’: how much will AI affect the art of film-making?](https://www.theguardian.com/film/article/2024/jul/27/artificial-intelligence-movies) | The future is here, whether some like it or not, and artificial intelligence is already impacting the film industry. But just how far can, and should, it go?|
|[Why Zuckerberg’s multibillion-dollar gamble doesn’t just matter to Meta.](https://www.theguardian.com/technology/article/2024/jul/26/why-zuckerbergs-multi-billion-dollar-gamble-doesnt-just-matter-to-meta) |As Llama 3.1 405B is made freely available, investors are asking when the huge industry spend will pay off |

# ML news: ML news: Week 15 - 21 July

## Research
|Link|description|
|---|---|
|[RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs.](https://arxiv.org/abs/2407.02485v1) |demonstrates how a Llama3-RankRAG significantly outperforms Llama3-ChatQA-1.5 and GPT-4 models on nine knowledge-intensive benchmarks. It also introduces a new instruction fine-tuning framework to perform effective context ranking and answering generation to enhance an LLM's RAG capabilities. This framework makes use of a small ranking dataset to outperform existing expert ranking models. |
|[Mixture of A Million Experts.](https://arxiv.org/abs/2407.04153) |aims to decouple computational cost from parameter count by efficiently routing to a large number of tiny experts through a learned index structure used for routing. It shows superior efficiency compared to dense FFW, coarse-grained MoEs, and Product Key Memory (PKM) layers. introduces a parameter-efficient expert retrieval mechanism that uses the product key technique for sparse retrieval from a million tiny experts. |
|[Reasoning in Large Language Models: A Geometric Perspective.](https://arxiv.org/abs/2407.02678) | establishes a relationship between the expressive power of LLMs and the density of their self-attention graphs; their analysis shows that the density of these graphs defines the intrinsic dimension of the inputs to the MLP blocks. investigates the reasoning of LLMs from a geometrical perspective; reports that a higher intrinsic dimension implies greater expressive capacity of the LLM.|
|[Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps.](https://arxiv.org/abs/2407.07071) |Contextual Hallucinations Mitigation in LLMs: This paper presents a novel approach that both detects and reduces contextual hallucinations in LLMs (e.g., reduces by 10% in the XSum summarization task). It does this by building a hallucination detection model based on input features provided by the ratio of attention weights on the context vs. newly generated tokens (for each attention head). The theory behind this approach is that contextual hallucinations are related to the degree to which an LLM attends to the contextual information provided. Additionally, they suggest a decoding strategy that mitigates contextual hallucinations based on their detection method, and this can be applied to other models without requiring retraining. |
|[RouteLLM.](https://arxiv.org/abs/2406.18665v2) |uses human preference data and data augmentation techniques in its training framework to improve performance and reduce costs by over two times in some cases, all while maintaining response quality. It suggests effective router models to dynamically choose between stronger and weaker LLMs during inference to achieve a balance between cost and performance. |
|[Learning to (Learn at Test Time): RNNs with Expressive Hidden States.](https://arxiv.org/abs/2407.04620) | suggests new layers for sequence modeling that have linear complexity and an expressive hidden state; defines a hidden state as an ML model that can update even when tested; a two-layer MLP-based hidden state combined with a linear model is found to match or outperform baseline models such as Mamba, Transformers, and contemporary RNNs; the linear model is faster than Mamba in wall-clock time and matches Transformer at 8k context. |
|[Physicochemical graph neural network for learning protein–ligand interaction fingerprints from sequence data.](https://www.nature.com/articles/s42256-024-00847-1) | Predicting the binding affinity between small-molecule ligands and proteins is a key task in drug discovery; however, sequence-based methods are often less accurate than structure-based ones. Koh et al. develop a graph neural network using physicochemical constraints that discovers interactions between small molecules and proteins directly from sequence data and that can achieve state-of-the-art performance without the need for costly, experimental 3D structures.|
|[Generic protein–ligand interaction scoring by integrating physical prior knowledge and data augmentation modelling.](https://www.nature.com/articles/s42256-024-00849-z) |Machine learning can improve scoring methods to evaluate protein–ligand interactions, but achieving good generalization is an outstanding challenge. Cao et al. introduce EquiScore, which is based on a graph neural network that integrates physical knowledge and is shown to have robust capabilities when applied to unseen protein targets. |
|[MARS: Mixture of Auto-Regressive Models for Fine-grained Text-to-image Synthesis.](https://arxiv.org/abs/2407.07614v1) | Semantic Vision-Language Integration Expert (SemVIE) is a feature of MARS, a novel text-to-image (T2I) generation system.|
|[OpenDiLoCo.](https://www.primeintellect.ai/blog/opendiloco) |Prime Intellect duplicated the DeepMind technique known as Distributed Low-Communication (DiLoCo). It preserves GPU consumption while enabling cross-datacenter training. |
|[gpu.cpp.](https://github.com/AnswerDotAI/gpu.cpp) | A new lightweight and portable library for WebGPU-based low-level GPU computations has been launched by Answer AI. Writing cross-GPU kernels is possible with it, and portable instructions are provided.|
|[ViTime: A Visual Intelligence-based Foundation Model for Time Series Forecasting.](https://github.com/ikeyang/vitime) |Rather than using conventional numerical data fitting, the foundation model for time series forecasting (TSF) called ViTime makes use of visual intelligence. |
|[Gradient Boosting Reinforcement Learning.](https://arxiv.org/abs/2407.08250v1) |The benefits of Gradient Boosting Trees (GBT) are applied to reinforcement learning using Gradient-Boosting RL (GBRL). |
|[SpreadsheetLLM: Encoding Spreadsheets for Large Language Models.](https://arxiv.org/abs/2407.09025) | An excellent study explaining how to convert a spreadsheet into a suitable representation for a contemporary LLM. Q/A, formatting, and other data operations can be done using this.|
|[LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models.](https://arxiv.org/abs/2407.08966v1) |Label-focused A novel technique for out-of-distribution (OOD) detection in Vision-Language Models such as CLIP is Automated Prompt Tuning (LAPT). |
|[Prover-Verifier Games improve legibility of language model outputs.](https://openai.com/index/prover-verifier-games-improve-legibility/) |In order to enable a weak model to grade content reliably, OpenAI trained a strong model to produce more legible text. The company discovered that this improved overall readability generally. |
|[Temporally Consistent Stereo Matching.](https://arxiv.org/abs/2407.11950v1) | By guaranteeing temporal consistency, researchers present a novel technique for video stereo matching that improves depth estimation.|
|[Patch-Level Training for Large Language Models.](https://arxiv.org/abs/2407.12665v1) |To increase training efficiency for big language models, researchers suggest patch-level training. |


## News
|Link|description|
|---|---|
|[Elon Musk promises ‘battle in court’ over EU’s crackdown on X’s blue checks.](https://www.theguardian.com/technology/article/2024/jul/12/eu-regulators-warns-x-may-face-fines-for-deceptive-blue-tick-system) | Regulators’ findings suggest social network breached Digital Services Act and could be fined 6% of global turnover|
|[AI prompts can boost writers’ creativity but result in similar stories, study finds.](https://www.theguardian.com/technology/article/2024/jul/12/ai-prompts-can-boost-writers-creativity-but-result-in-similar-stories-study-finds) | Ideas generated by ChatGPT can help writers who lack inherent flair but may mean there are fewer unique ideas|
|[OpenAI is reportedly working on more advanced AI models capable of reasoning and ‘deep research’.](https://www.engadget.com/openai-is-reportedly-working-on-more-advanced-ai-models-capable-of-reasoning-and-deep-research-202419228.html) |The secret project is code-named ‘Strawberry,’ according to a Reuters report. |
|[Meet the AI Agent Engineer.](https://sierra.ai/blog/meet-the-ai-agent-engineer) |At his company, Sierra, Bret Taylor, the Chairman of the Board of OpenAI, has created a new position called Agent Engineer. One of the first people in the role recently wrote a blog post describing the Sierra team's view of agent engineering as a new field inside AI engineering. |
|[OpenAI Revenue.](https://futuresearch.ai/openai-revenue-report) |An estimated $3.4 billion in revenue for OpenAI comes from its ChatGPT services. |
|[Taming the tail utilization of ads inference at Meta scale.](https://engineering.fb.com/2024/07/10/production-engineering/tail-utilization-ads-inference-meta) |Meta's machine learning inference services saw a two-thirds decrease in failure rates, a 35% increase in compute efficiency, and a halving of p99 latency because to changes made in the tail utilization. With these improvements, Meta's ad delivery systems are guaranteed to be able to manage growing workloads without requiring more resources and to uphold service level agreements. Predictive scaling and managing the machine learning model lifetime with Meta's unified platform, IPnext, are examples of continuous improvement techniques. |
|[Meta to reportedly launch largest Llama 3 model on July 23.](https://breakingthenews.net/Article/Meta-to-reportedly-launch-largest-Llama-3-model-on-July-23/62364570) |Meta Platforms will release its largest Llama 3 model on July 23, The Information reported on Friday, citing an employee of the company. The new model, boasting 405 billion parameters, will be multimodal and capable of understanding and generating both images and text. |
|[Quora’s Poe now lets users create and share web apps.](https://techcrunch.com/2024/07/08/quoras-poe-now-lets-users-create-and-share-web-apps/) |Poe, Quora’s subscription-based, cross-platform aggregator for AI-powered chatbots like Anthropic’s Claude and OpenAI’s GPT-4o, has launched a feature called Previews that lets people create interactive apps directly in chats with chatbots.|
|[Microsoft CTO Kevin Scott thinks LLM “scaling laws” will hold despite criticism.](https://arstechnica.com/information-technology/2024/07/microsoft-cto-defies-critics-ai-progress-not-slowing-down-its-just-warming-up/) |Will LLMs keep improving if we throw more compute at them? OpenAI dealmaker thinks so. |
|[OpenAI says there are 5 'levels' for AI to reach human intelligence — it's already almost at level 2.](https://qz.com/openai-five-level-system-human-intelligence-ai-1851588122) |The company shared a five-level system it developed to track its artificial general intelligence, or AGI, progress with employees this week, an OpenAI spokesperson told Bloomberg. The levels go from the currently available conversational AI to AI that can perform the same amount of work as an organization.|
|[AI startup Hebbia raised $130M at a $700M valuation on $13 million of profitable revenue.](https://techcrunch.com/2024/07/09/ai-startup-hebbia-rased-130m-at-a-700m-valuation-on-13-million-of-profitable-revenue) | Hebbia, a startup that uses generative AI to search large documents and respond to large questions, has raised a $130 million Series B at a roughly $700 million valuation led by Andreessen Horowitz, with participation from Index Ventures, Google Ventures and Peter Thiel.|
|[Pixel 9 Pro might come with 1-year of Gemini Advanced.](https://9to5google.com/2024/07/15/pixel-9-pro-might-gemini-advanced/) |With less than a month until Made by Google 2024, the latest leak suggests that the Pixel 9 Pro will come with 1-year of Gemini Advanced. |
|[Company Abandons Plans to Give AI Workers "Rights" and Add Them to Org Chart After Outcry From Human Employees.](https://futurism.com/startup-ai-rights-org-chart) |Following its announcement that it would give AI algorithms "rights" and integrate them as "digital workers" with managers and performance evaluations in its product, the HR software provider Lattice encountered criticism. |
|[Want to know how AI will affect government and politics? The bots have the answers.](https://www.theguardian.com/technology/article/2024/jul/16/want-to-know-how-ai-will-affect-government-and-politics-the-bots-have-the-answers) |Tony Blair’s powerful thinktank asked ChatGPT how AI might affect public sector jobs. Critics say the results were … wonky |
|[Andrej Karpathy's new company.](https://eurekalabs.ai/) |A new AI startup with an emphasis on education, Eureka Labs aims to transform the way we acquire new knowledge. |
|[Whistleblowers accuse OpenAI of ‘illegally restrictive’ NDAs.](https://techcrunch.com/2024/07/13/whistleblowers-accuse-openai-of-illegally-restrictive-ndas/) | Whistleblowers have accused OpenAI of placing illegal restrictions on how employees can communicate with government regulators, according to a letter obtained by The Washington Post.|
|[Apple, Nvidia, Anthropic Used Thousands of Swiped YouTube Videos to Train AI.](https://www.proofnews.org/apple-nvidia-anthropic-used-thousands-of-swiped-youtube-videos-to-train-ai/) | AI companies are generally secretive about their sources of training data, but an investigation by Proof News found some of the wealthiest AI companies in the world have used material from  thousands of  YouTube videos to train AI. Companies did so despite YouTube’s rules against harvesting materials from the platform without permission.|
|[SciCode: A Research Coding Benchmark Curated by Scientists.](https://scicode-bench.github.io/) |The objective of coding models has always been HumanEval. It is essentially solved now. This benchmark is the next step forward in solving difficult science programming puzzles. |
|[SmolLM - blazingly fast and remarkably powerful.](https://huggingface.co/blog/smollm) |This blog post introduces SmolLM, a family of state-of-the-art small models with 135M, 360M, and 1.7B parameters, trained on a new high-quality dataset. It covers data curation, model evaluation, and usage. |
|[Benchmarking results for vector databases.](https://redis.io/blog/benchmarking-results-for-vector-databases/) |Redis has released updated information on the best vector databases, measuring throughput and latency with the help of the industry-recognized Qdrant framework. Key findings include Redis achieving much higher queries per second and lower latency than Qdrant, Milvus, and Weaviate, and outperforming competitors by 62% for low complexity datasets and by 21% for high-dimensional datasets. |
|[Announcing the launch of Gray Swan.](https://www.grayswan.ai/news/gray-swan-launch) |A company specializing in creating tools to assist businesses in evaluating the risks associated with their AI systems and protecting their AI installations from inappropriate use is called Gray Swan AI. |
|[Anthropic releases Claude app for Android.](https://techcrunch.com/2024/07/16/anthropic-releases-claude-app-for-android/) |Anthropic launched its Claude Android app on Tuesday to bring its AI chatbot to more users. This is Anthropic’s latest effort to convince users to ditch ChatGPT by making Claude available in more places. |
|[AI tool can pinpoint dementia’s cause — from stroke to Alzheimer’s.](https://www.nature.com/articles/d41586-024-02202-1) |Algorithm that distinguishes among a host of underlying causes of dementia could be used for diagnosis in hospitals and clinics. |
|[Portal needed for victims to report AI deepfakes, federal police union says.](https://www.theguardian.com/technology/article/2024/jul/18/ai-deepfakes-revenge-porn-reporting-portal-australia-laws) |Parliamentary inquiry told police forced to ‘cobble together’ laws to prosecute man who allegedly spread deepfake images of women |
|[Meta Won't Offer Future Multimodal AI Models In The EU.](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu) |Due to legislative uncertainties, Meta will not be able to provide future multimodal AI models to consumers in the EU; however, Llama 3 will still be offered in text only. |
|[Anthropic teams up with venture capital firm to kickstart $100M AI startup fund.](https://www.theregister.com/2024/07/17/anthropic_teams_up_with_vc/) |Recipients of six-digit investments aren’t required to use Claude |
|[Anthropic doubles output token limit.](https://threadreaderapp.com/thread/1812921642143900036.html) | Anthropic has doubled the max output token limit for Claude 3.5 Sonnet from 4096 to 8192 in the Anthropic API.|
|[AI-powered video creation for work.](https://workspace.google.com/products/vids/) |An AI-powered video creation tool for the workplace, Google Vids is tightly integrated with the Workspace suite. |
|[aiXplain Secures $6.5M pre-Series A to Universalize AI Agent Development.](https://www.einnews.com/pr_news/728139645/aixplain-secures-6-5m-pre-series-a-to-universalize-ai-agent-development) | Saudi Aramco's venture arm, Wa'ed Ventures, has announced a $6.5 million pre-series A fundraising round for aiXplain (a global top 10 firm by market cap).|
|[Meta pulls plug on release of advanced AI model in EU.](https://www.theguardian.com/technology/article/2024/jul/18/meta-release-advanced-ai-multimodal-llama-model-eu-facebook-owner) |‘Unpredictable’ privacy regulations prompt Facebook owner to scrap regional plans for multimodal Llama |
|[Mistral NeMo.](https://mistral.ai/news/mistral-nemo/) | A novel tokenizer was used to train the multilingual Mistral Nemo 12B model, which exhibits strong multilingual and English performance. Also supported are 128k contexts.|
|[OpenAI is releasing a cheaper, smarter model.](https://www.theverge.com/2024/7/18/24200714/openai-new-cheaper-smarter-model-gpt-4o-mini) |OpenAI is releasing a lighter, cheaper model for developers to tinker with called GPT-4o Mini. It costs significantly less than full-sized models and is said to be more capable than GPT-3.5. |
|[Cohere and Fujitsu Announce Strategic Partnership To Provide Japanese Enterprise AI Services.](https://cohere.com/blog/fujitsu-partnership) |Cohere and Fujitsu have partnered strategically to create and offer enterprise AI services that have the best Japanese language capabilities in the market. These services, which will provide private cloud deployments to businesses in highly regulated sectors including financial institutions, the public sector, and research and development units, will be developed with security and data privacy as their primary goals. |
|[OpenAI And Broadcom Held Discussions About Producing An AI Chip.](https://seekingalpha.com/news/4125638-broadcom-held-discussions-with-openai-about-producing-ai-chip-report) | OpenAI and Broadcom have discussed developing a new artificial intelligence server processor.|
|[Flow Studio.](https://www.producthunt.com/posts/flow-studio) |Flow Studio creates 3-minute films that are completely produced, with a believable story, dependable characters, and automatically synced sound effects and background music. |
|[Slow recovery from IT outage begins as experts warn of future risks.](https://www.theguardian.com/australia-news/article/2024/jul/19/microsoft-windows-pcs-outage-blue-screen-of-death) |Fault in CrowdStrike caused airports, businesses and healthcare services to languish in ‘largest outage in history’ |

## Resources
|Link|description|
|---|---|
|[A Survey on Mixture of Experts.](https://arxiv.org/abs/2407.06204) | a survey study on the Mixture of Experts (MoE), covering its technical specifications, open-source implementations, assessment methods, and practical uses. |
|[Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence.](https://arxiv.org/abs/2407.07061v2) |a new framework to address several limitations in multi-agent frameworks such as integrating diverse third-party agents and adaptability to dynamic task requirements; introduces an agent integration protocol, instant messaging architecture design, and dynamic mechanisms for effective collaboration among heterogeneous agents. |
|[Meta 3D Gen.](https://ai.meta.com/research/publications/meta-3d-gen/) | a new pipeline that can generate 3D assets from text in less than a minute, from start to finish. It incorporates cutting-edge parts like TextureGen and AssetGen to represent objects in three dimensions: view space, volumetric space, and UV space. It also achieves a 68% win rate compared to the single-stage model.|
|[Challenges, evaluation and opportunities for open-world learning.](https://www.nature.com/articles/s42256-024-00852-4) | Here we argue that designing machine intelligence that can operate in open worlds, including detecting, characterizing and adapting to structurally unexpected environmental changes, is a critical goal on the path to building systems that can solve complex and relatively under-determined problems.  |
|[Machine learning-aided generative molecular design.](https://www.nature.com/articles/s42256-024-00843-5) |Data-driven generative methods have the potential to greatly facilitate molecular design tasks for drug design. |
|[Introducing AuraFlow v0.1, an Open Exploration of Large Rectified Flow Models.](https://blog.fal.ai/auraflow/) | Fal trained a new open model called AuraFlow. The model has 5.8B parameters and was trained with muP.|
|[Lynx: State-of-the-Art Open Source Hallucination Detection Model.](https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model) |a model for identifying language model hallucinations that performs noticeably better than the state of the art in its generations. |
|[Hyper-3DG: Text-to-3D Gaussian Generation via Hypergraph.](https://arxiv.org/abs/2403.09236v1) | Hyper-3DG enhances text-to-3D model creation by emphasizing the intricate connections between texture and geometry.|
|[LightenDiffusion.](https://github.com/jianghaiscu/lightendiffusion) |By utilizing diffusion models and Retinex theory, LightenDiffusion enhances low-light photos. |
|[ProDepth.](https://sungmin-woo.github.io/prodepth/) |A novel framework for monocular depth estimation called ProDepth addresses problems brought on by moving objects in dynamic situations. It finds and fixes discrepancies in depth estimate using a probabilistic method. |
|[Open-Canopy.](https://arxiv.org/abs/2407.09392v1) | A high-resolution (1.5 m) publicly available dataset called Open-Canopy is used to estimate canopy height over France.|
|[crawlee-python.](https://github.com/apify/crawlee-python) | Crawlee—A web scraping and browser automation library for Python to build reliable crawlers. Extract data for AI, LLMs, RAG, or GPTs. Download HTML, PDF, JPG, PNG, and other files from websites. Works with BeautifulSoup, Playwright, and raw HTTP. Both headful and headless mode. With proxy rotation.|
|[Mathstral.](https://mistral.ai/news/mathstral/) | Mistral's newest math model performs well on various benchmarks|
|[Codestral Mamba.](https://mistral.ai/news/codestral-mamba/) | Codestral Mamba, a Mamba2 language model specialised in code generation, available under an Apache 2.0 license.|
|[exo.](https://github.com/exo-explore/exo) |Run your own AI cluster at home on everyday devices. |
|[Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training.](https://github.com/robustnlp/derta) |Through addressing refusal position bias, a novel method called Decoupled Refusal Training (DeRTa) enhances safety tuning in large language models. |
|[PID: Physics-Informed Diffusion Model for Infrared Image Generation.](https://github.com/fangyuanmao/pid) | By integrating physical laws into the conversion process, researchers have created a Physics-Informed Diffusion (PID) model that enhances the translation of RGB images to infrared images.|
|[What happened to BERT & T5? On Transformer Encoders, PrefixLM and Denoising Objectives.](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising) |Excellent post on encoders, prefixlm, denoising aims, and other contemporary language modeling techniques by Yi Tay of Reka and Google. |
|[LiDAR Semantic Segmentation.](https://arxiv.org/abs/2407.11569v1) |A novel technique called SFPNet is intended to be universal across various LiDAR technology types. Instead of employing window-attention as in the past, SFPNet uses sparse focus point modulation to extract and dynamically collect multi-level contexts. |
|[Praison AI.](https://github.com/MervinPraison/PraisonAI) |Using prior agent frameworks as a springboard, Praison AI is a low-code, centralized framework with customizable features and human-agent interaction that makes it easier to create and manage multi-agent systems for a range of LLM applications. |
|[Video Object Segmentation with World Knowledge.](https://github.com/cilinyan/VISA) | Reasoning Video Object Segmentation (ReasonVOS) is a new task that uses implicit text queries to generate segmentation masks. It requires complex reasoning and world knowledge.|
|[Enhancing Class Learning Without Forgetting.](https://github.com/roadonep/eccv2024_mbs) |In order to enhance Class-Incremental Semantic Segmentation (CISS), this project presents a background-class separation framework. |
|[Leapfrogging traditional vector-based RAG with language maps.](https://x.com/mutableai/status/1813815706783490055) |When developing a chat application over data, retrieval plays a major role. But frequently, systems are delicate to the format of the data being accessed. Chat-based performance is greatly enhanced by creating a language map (e.g., Wikipedia style entry) of the material and using that for retrieval. This is how code base question answering is handled by mutable AI. |
|[Removing Inappropriate Content from Diffusion Models.](https://arxiv.org/abs/2407.12383v1) | Using a revolutionary technique called Reliable and Efficient Concept Erasure (RECE), improper content may be removed from diffusion models in only three seconds without requiring additional fine-tuning.|
|[LLM2sh.](https://github.com/randombk/llm2sh) |A command-line tool called LLM2sh uses LLMs to convert requests written in plain English into shell instructions. |
|[GraphMuse.](https://github.com/manoskary/graphmuse) |GraphMuse is a Python Library for Graph Deep Learning on Symbolic Music. This library intents to address Graph Deep Learning techniques and models applied specifically to Music Scores. |
|[E5-V: Universal Embeddings with Multimodal Large Language Models.](https://github.com/kongds/e5-v) | A novel framework called E5-V modifies Multimodal Large Language Models (MLLMs) to provide multimodal embeddings that are universal. With prompts, it bridges the gap between various input formats and achieves remarkable results in multimodal activities without the need for fine-tuning.|
|[Strategizing Your Preparation for Machine Learning Interviews.](https://mlengineerinsights.substack.com/p/strategizing-your-preparation-for) | Interviews for machine learning might be difficult. You may greatly increase your chances by being aware of the range of machine learning positions and adjusting your preparation to fit particular job duties and specializations. To approach interviews with confidence, concentrate on learning the fundamentals, investigating technology unique to the organization, and regularly monitoring your progress.|
|[Uncensor Any LLM With Abliteration.](https://research.google/blog/smart-paste-for-context-aware-adjustments-to-pasted-code/) | For safety, llama models are heavily restricted, which reduces their versatility. Through the identification and elimination of the rejection mechanism, the "abliteration" technique uncensors them, enabling models to respond to all stimuli without requiring retraining.|
|[SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers.](https://arxiv.org/abs/2407.09413v1) |SPIQA is a quality assurance dataset created to assist users in rapidly locating solutions within scientific research publications by deciphering intricate figures and tables. |


## Perspectives
|Link|description|
|---|---|
|[AI’s ‘Oppenheimer moment’: autonomous weapons enter the battlefield.](https://www.theguardian.com/technology/article/2024/jul/14/ais-oppenheimer-moment-autonomous-weapons-enter-the-battlefield) |The military use of AI-enabled weapons is growing, and the industry that provides them is booming |
|[Will generative AI transform robotics?](https://www.nature.com/articles/s42256-024-00862-2) |In the current wave of excitement about applying large vision–language models and generative AI to robotics, expectations are running high, but conquering real-world complexities remains challenging for robots. |
|[Introducing: The Managed-Service-as-Software (M-SaS) Startup.](https://dannguyenhuu.substack.com/p/introducing-the-managed-service-as) |AI-driven, service-oriented firms are creating Managed-Service-as-Software (M-SaS) enterprises, which follow a new business model blueprint in building their businesses. Startups need to adopt a fundamentally different attitude in order to use AI instead of selling it. These firms start off labor-intensive with low gross margins and then use automation and artificial intelligence (AI) to progressively move to greater SaaS-like gross margins. |
|[Could AIs become conscious? Right now, we have no way to tell.](https://arstechnica.com/science/2024/07/could-ais-become-conscious-right-now-we-have-no-way-to-tell/) |With divergent opinions on whether developments in machine learning and neuromorphic computing can result in sentient computers, the discussion over artificial intelligence potentially gaining awareness is becoming more heated. The theory of Integrated Information holds that the current hardware limits make AI consciousness implausible, while computational functionalist theories such as Global Neuronal Workspace Theory and Attention Schema Theory believe that AI awareness is inevitable. Neuroscience is trying to come up with a single theory of consciousness in order to better understand how it might show up in AI. |
|[Generative AI makes for better scientific writing — but beware the pitfalls.](https://www.nature.com/articles/d41586-024-02319-3) |As researchers who have sometimes struggled with articulating intricate concepts, we find his suggestions for using ChatGPT to improve the clarity and coherence of academic papers compelling. But potential pitfalls warrant further discussion. |
|[My trip to the frontier of AI education.](https://www.gatesnotes.com/My-trip-to-the-frontier-of-AI-education) |First Avenue Elementary School in Newark is utilizing Khanmigo, an AI-powered tutor and teacher assistant created by Khan Academy, to include AI tools for education. Teachers in the classroom can customize instruction and cut down on work time by using this technology. The goal of increasing responsiveness and inclusion is a continuous endeavor. Through increased teacher-student involvement, this Gates Foundation-backed project seeks to level the playing field in education. |
|[AI-Driven Behavior Change Could Transform Health Care.](https://time.com/6994739/ai-behavior-change-health-care/) |Thrive AI Health is being funded by OpenAI and Thrive Global to create a customized AI health coach that addresses everyday health-related behaviors like nutrition and sleep. AI's hyper-personalization powers the mobile app and corporate solution by fusing individual data with peer-reviewed science. The project intends to manage chronic diseases, democratize healthy behavior modification, and show how effectively AI can be integrated into healthcare while maintaining robust privacy protections. |
|[GraphRAG Analysis, Part 1: How Indexing Elevates Knowledge Graph Performance in RAG.](https://aiencoder.substack.com/p/graphrag-analysis-part-1-how-indexing) | Analysis of Microsoft's GraphRAG research suggests that knowledge graphs like Neo4j may not significantly beat FAISS in context retrieval for RAG applications. While Neo4j without its indexing can reach a better answer relevancy, the minor advantages may not justify the cost given ROI limits. Neo4j's indexing, on the other hand, significantly improves answer faithfulness, lowering the possibility of false information.|
|[How Taiwan secured semiconductor supremacy – and why it won’t give it up.](https://www.theguardian.com/world/article/2024/jul/19/taiwan-semiconductor-industry-booming) | Trump has accused Taiwan of ‘taking’ the US chip sector, but Taipei has been at the forefront of the industry for decades, and its future could depend on it|
|[Overcoming The Limits Of Current LLMs.](https://seanpedersen.github.io/posts/overcoming-llm-limits) | Large language models (LLM) have been all the rage for quite some time now. Looking beyond the hype though, they have severe limitations: hallucinations, lack of confidence estimates and lack of citations.|




























