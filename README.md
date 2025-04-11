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
|[Rope to Nope and Back Again: A New Hybrid Attention Strategy.](https://arxiv.org/abs/2501.18795) |Llama 4's breakthrough in handling over 10 million tokens in context comes from alternating between no positional embeddings and rotational positional embeddings. Although current benchmarks are limited to Needle in the Haystack, they strongly suggest the effectiveness of this alternating layer approach. |
|[Inference-Time Scaling for Generalist Reward Modeling.](https://arxiv.org/abs/2504.02495) |This DeepSeek paper explores using inference-time scaling to improve reward modeling as a way to develop stronger reasoners. It suggests a larger plan by the Chinese start-up to leverage its current reasoning models as a foundation for building the next wave of reward models to train future reasoners. |
|[CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation.](https://arxiv.org/abs/2503.22708) | Researchers at AI2 introduce CodeScientist, a system that autonomously generates and tests scientific hypotheses through code-based experimentation, making validated discoveries with minimal human input. CodeScientist reviews research papers and designs experiments using Python code blocks, following a five-step pipeline: Ideation, Planning, Code Execution, Reporting, and Meta-Analysis. From 50 AI research papers, it proposed 19 findings, with 6 deemed scientifically sound, including insights like the mismatch between LLM confidence and accuracy, the benefit of simpler states for better predictions, and the advantage of graph memory in simulations. While full automation is possible, human feedback enhances the quality of results. Despite successes, over half of experiments fail due to code errors, highlighting the need for peer review and more rigorous methodologies.|
|[One-Minute Video Generation with Test-Time Training.](https://test-time-training.github.io/video-dit) | This study presents Test-Time Training (TTT) layers with rich hidden states to address the shortcomings of traditional Transformers and models like Mamba in producing long, coherent videos. By adding TTT layers to a pre-trained model, it achieves one-minute video generation from text storyboards that significantly surpass baseline methods in conveying complex narratives, based on human evaluations. Tom and Jerry cartoons serve as the test environment.|
|[Scaling Analysis of Interleaved Speech-Text Language Models.](https://arxiv.org/abs/2504.02398v1) | This study shows that speech-language models initialized from text models using interleaved training scale more efficiently than models trained solely on speech.|
|[Retrieval-Augmented Reasoning Model.](https://arxiv.org/abs/2503.23513) |RARE introduces a new approach for training domain-specific LLMs focused on reasoning rather than memorization. Inspired by Bloom’s Taxonomy, it emphasizes applying and evaluating knowledge rather than merely recalling facts. RARE separates domain knowledge, retrieved externally, from domain thinking, learned during training, enabling better performance within limited parameter budgets. By using an open-book approach, it injects retrieved knowledge into training prompts, fostering reasoning patterns. This method outperforms standard SFT and RAG, especially in medicine, with small models like Llama-3.1-8B and Qwen-2.5-7B achieving up to 20% higher accuracy on medical QA benchmarks. RARE also uses distillation and adaptive retries to refine outputs and integrate retrieval during training to shape reasoning, replacing memorization with application. |
|[A New Batch Normalization.](https://arxiv.org/abs/2504.00660) |This paper proposes a new batch normalization method for SPD manifolds that uses a learnable Generalized Bures-Wasserstein metric. |
|[How Students Use Claude in Education.](https://www.anthropic.com/news/anthropic-education-report-how-university-students-use-claude) | Anthropic studied one million student conversations to explore AI use in education, finding that STEM students are the primary users, mainly using Claude for content creation, solving technical problems, and tackling advanced learning tasks.|
|[Why do LLMs Attend to First Token?](https://arxiv.org/abs/2504.02732) | This paper explains why LLMs tend to focus attention on the first token, a phenomenon called an attention sink. The theory suggests it prevents representational collapse in deep Transformers. Long contexts and deep layers can lead to over-mixing, causing similar embeddings for all tokens, but attention sinks act as no-ops to preserve representation diversity. Experiments on Gemma 7B and LLaMa 3.1 models show that attention heads fixate on the first token, with larger models requiring stronger sinks. Sinks form naturally due to the token's position, not its content, and removing the ⟨bos⟩ token after training leads to performance collapse. The paper connects this behavior to Jacobian norm bounds, demonstrating that sinks reduce sensitivity to token changes, and reveals that some attention heads use ⟨bos⟩ as a default unless triggered by specific patterns.|
|[MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions.](https://arxiv.org/abs/2503.22678) |MedAgentSim is an open-source, fully automated hospital simulation where LLM-powered agents simulate doctor-patient interactions in dynamic diagnostic settings. Unlike static QA benchmarks, it involves multi-turn consultations, lab and imaging requests, and iterative diagnosis refinement. The system improves through memory and reflection, using past cases and chain-of-thought reasoning to enhance performance over time. Users can choose to control the doctor or patient agents, and the simulation, built with a 2D game engine, allows for interaction with virtual medical tools. MedAgentSim outperforms baseline setups by 6-37% across several benchmarks, particularly in vision-language tasks, and its bias analysis highlights the importance of cognitive and implicit bias-aware evaluation. |
|[Z1: Efficient Test-time Scaling with Code.](https://arxiv.org/abs/2504.00810v1) |Z1 is a new method designed to make LLMs more compute-efficient during reasoning at test time. It involves training LLMs on both short and long code-based reasoning trajectories, then adjusting reasoning depth dynamically during inference. The Z1-Code-Reasoning-107K dataset pairs simple and complex coding problems to teach the model when to stop reasoning. A novel test-time strategy, the Shifted Thinking Window, adapts the reasoning token budget based on problem complexity, enabling shallow reasoning for simple tasks and deeper reasoning for complex ones. Z1-7B achieves efficiency gains, matching the performance of larger models like R1-Distill-Qwen-7B but using only 30% of the reasoning tokens. Despite being trained on code-based reasoning, Z1 generalizes well to other domains, outperforming other 7B models across multiple benchmarks. Ablation studies show that longer reasoning paths and larger training sample sizes improve inference quality and accuracy. |
|[Inside-Out: Hidden Factual Knowledge in LLMs.](https://arxiv.org/abs/2503.15299) | This study presents a framework to measure hidden knowledge in LLMs, revealing that models store significantly more factual information internally than they express in outputs, with a difference of up to 40%. It also finds that some answers, while internally known, are never generated, highlighting limitations in test-time sampling for QA tasks.|
|[Photonic chips provide a processing boost for AI.](https://www.nature.com/articles/d41586-025-00907-5) |Computer processors that exploit both electricity and light could improve the performance of artificial-intelligence systems while consuming less energy. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Llama 4.](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | Meta has introduced Llama 4 Scout and Maverick, two 17B-parameter multimodal models delivering top-tier results on key benchmarks, as well as Llama 4 Behemoth, a 288B model still in training that outperforms GPT-4.5 in STEM-related tasks.|
|[Meta Responds to Llama 4 Rumors.](https://x.com/Ahmad_Al_Dahle/status/1909302532306092107) |Meta's VP of Generative AI has refuted accusations that Llama 4 models were trained on benchmark test sets, rejecting claims that their performance results were artificially boosted. |
|[Amazon Nova Reel 2-Minute Videos.](https://aws.amazon.com/it/blogs/aws/amazon-nova-reel-1-1-featuring-up-to-2-minutes-multi-shot-videos/) | The upgraded Nova Reel model now handles multi-shot videos up to 2 minutes in length, providing greater creative control and improved efficiency for generating video content.|
|[Midjourney V7.](https://www.midjourney.com/updates/v7-alpha) | Midjourney has launched its V7 alpha image generation model, featuring improved text understanding, enhanced image consistency, and a new Draft Mode for quick, budget-friendly iterations, along with support for voice commands and personalization.| 
|[AI masters Minecraft: DeepMind program finds diamonds without being taught.](https://www.nature.com/articles/d41586-025-01019-w) |DeepMind's AI system, Dreamer, managed to learn how to collect diamonds in Minecraft without any human instruction, marking progress toward more general AI. Through reinforcement learning, Dreamer explores and models the game world on its own to forecast actions and results. This development points to possible real-world uses where trial-and-error learning would be expensive. |
|[OpenAI’s models ‘memorized’ copyrighted content, new study suggests.](https://techcrunch.com/2025/04/04/openais-models-memorized-copyrighted-content-new-study-suggests/) | A new study appears to lend credence to allegations that OpenAI trained at least some of its AI models on copyrighted content.|
|[UK Home Office loses attempt to keep legal battle with Apple secret.](https://www.theguardian.com/politics/2025/apr/07/uk-home-office-loses-attempt-to-keep-legal-battle-with-apple-secret) |Judges reject Home Office’s attempt to withhold from public details of case concerning access of Apple users’ data |
|[Investing in Krea.](https://a16z.com/announcement/investing-in-krea/) |Andreessen Horowitz has invested in Krea, a platform that blends AI models to assist creatives in generating and editing visual content. With over 20 million users, including teams at Pixar and Samsung, Krea is set to release an enterprise-grade product later this year. |
|[Google's AI Highlights in March.](https://blog.google/technology/ai/google-ai-updates-march-2025/) |Google recaps major March updates including Gemini 2.5 Pro, expanded AI Overviews, AI Mode, and other feature rollouts across its products. |
|[Genies unveils user-generated content tools that let anyone create custom AI avatars.](https://venturebeat.com/games/genies-unveils-user-generated-content-tools-that-let-anyone-create-custom-ai-avatars/) | Genies has released a no-code platform that lets users create intelligent AI avatars with distinct appearances, personalities, and behaviors for use in customizable gaming experiences called Parties. Powered by large language models, behavioral AI, and real-time animation, these avatars support dynamic interaction, gameplay, and emotional expression.|
|[OpenAI Plans O3 and O4-Mini Release Before GPT-5, Altman Say.](https://decrypt.co/313379/openai-o3-o4-mini-release-before-gpt5) | OpenAI plans to release intermediate models o3 and o4-mini ahead of GPT-5, citing technical challenges with GPT-5 and aiming to improve performance while managing demand. This move comes amid rising competition from models like Google’s Gemini 2.5 Pro, and follows OpenAI's recent $40 billion funding round.|
|[OpenAI’s o3 model might be costlier to run than originally estimated.](https://techcrunch.com/2025/04/02/openais-o3-model-might-be-costlier-to-run-than-originally-estimated/) |When OpenAI unveiled its o3 “reasoning” AI model in December, the company partnered with the creators of ARC-AGI, a benchmark designed to test highly capable AI, to showcase o3’s capabilities. Months later, the results have been revised, and they now look slightly less impressive than they did initially. |
|[Bringing multimodal search to AI Mode.](https://blog.google/products/search/ai-mode-multimodal-search) |Google is expanding its AI Mode feature to millions of U.S. Labs users, enhancing it with multimodal capabilities. |
|[Waymo may use interior camera data to train generative AI models, but riders will be able to opt out.](https://techcrunch.com/2025/04/08/waymo-may-use-interior-camera-data-to-train-generative-ai-models-sell-ads/) | Waymo is preparing to use data from its robotaxis, including video from interior cameras tied to rider identities, to train generative AI models, according to an unreleased version of its privacy policy found by researcher Jane Manchun Wong, raising fresh questions about how much of a rider’s behavior inside autonomous vehicles could be repurposed for AI training.|
|[Microsoft’s Copilot can now browse the web and perform actions for you.](https://techcrunch.com/2025/04/04/microsofts-copilot-can-now-browse-the-web-and-perform-actions-for-you/) |Microsoft's Copilot AI chatbot now performs tasks on popular websites, remembers user preferences, and analyzes real-time video. |
|[ElevenLabs releases official MCP server for AI-driven audio processing.](https://github.com/elevenlabs/elevenlabs-mcp) |ElevenLabs has launched an official Model Context Protocol server for Text-to-Speech and audio processing that is compatible with clients like Claude Desktop and OpenAI Agents. |
|[Big tech’s new datacentres will take water from the world’s driest areas.](https://www.theguardian.com/environment/2025/apr/09/big-tech-datacentres-water) | Amazon, Google and Microsoft are building datacentres in water-scarce parts of five continents|
|[EU to build AI gigafactories in €20bn push to catch up with US and China.](https://www.theguardian.com/technology/2025/apr/09/eu-to-build-ai-gigafactories-20bn-push-catch-up-us-china) |Up to five sites with power-hungry supercomputers and datacentres planned to drive AI ‘moonshots’ |
|[Online suicide forum investigated under new UK digital safety laws.](https://www.theguardian.com/technology/2025/apr/09/online-suicide-forum-ofcom-investigation-uk-digital-safety-laws) | Ofcom’s first investigation to look into whether site took adequate measures to shield users from illegal content|
|[White House insists iPhones will be US-made – but Apple calls it a non-starter.](https://www.theguardian.com/us-news/2025/apr/09/trump-apple-iphones-made-in-usa) |Experts doubt Trump line that tariffs and company’s $500bn investment will shift manufacturing from Asia |
|[Dr Oz tells federal health workers AI could replace frontline doctors.](https://www.theguardian.com/us-news/2025/apr/09/mehmet-oz-doctors-ai) |Former TV doctor who leads $1.5tn Medicare and Medicaid agency also says staff have ‘patriotic duty’ to stay healthy |
|[Bank of England says AI software could create market crisis for profit.](https://www.theguardian.com/business/2025/apr/09/bank-of-england-says-ai-software-could-create-market-crisis-profit) |Concern grows over programs deployed to act with autonomy that may ‘exploit weaknesses’ |
|[EU to build AI gigafactories in €20bn push to catch up with US and China.](https://www.theguardian.com/technology/2025/apr/09/eu-to-build-ai-gigafactories-20bn-push-catch-up-us-china) |Up to five sites with power-hungry supercomputers and datacentres planned to drive AI ‘moonshots’ |
|[EU will not rip up tech rules for trade deal with Trump, senior official says.](https://www.theguardian.com/world/2025/apr/11/eu-will-not-rip-up-tech-rules-for-trade-deal-with-trump-senior-official-says) |Bloc is ‘very committed’ to laws on big tech and is not targeting US companies, says European Commission’s Henna Virkkunen |
|[Amazon’s satellite launch designed to compete with Musk’s Starlink cancelled.](https://www.theguardian.com/us-news/2025/apr/10/amazon-satellite-launch-cancelled) |‘Liftoff not possible’ for rocket carrying Project Kuiper satellites, due to clouds that could trigger lightning strikes |
|[Apple said to be flying iPhones from India to US to avoid Trump tariffs.]() |Tech firm has reportedly flown 600 tonnes of handsets from Indian factories as Chinese goods face huge tariffs |
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
|[Unsupervised Panoptic Segmentation.](https://visinf.github.io/cups/) | CUPS is a novel approach to panoptic segmentation that requires no labeled data, using depth and motion cues to learn directly from scene-centric images.|
|[Generative Modeling for Crystals.](https://github.com/deepmodeling/crystalformer) | CrystalFormer is a transformer-based model that creates crystal structures by leveraging space group symmetry, enhancing efficiency and reducing data requirements in crystal generation.|
|[Nano Aha Moment.](https://github.com/McGill-NLP/nano-aha-moment) | A single file, single GPU, from scratch full parameter tuning library that replicates DeepSeek R1-Zero style training.|
|[Object Counting.](https://github.com/AhmedZgaren/Save) | A fully automated zero-shot object counting approach that uses feature maps and self-attention mechanisms, achieving state-of-the-art results on the FSC147 dataset.|
|[DeepSeek 1.58bit GGUF.](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S) |The Unsloth team identified which parts of the new R1 model can be effectively quantized, noting some tokenizer quirks that complicate the process. In short, only the MoE layers are quantized to 1.58 bits, while the rest stay at 4 or 6 bits using their dynamic quantization approach. |
|[Granite Speech 8B.](https://huggingface.co/ibm-granite/granite-speech-3.2-8b) |IBM silently launched a state-of-the-art speech recognition and understanding model based on its Granite series. |
|[Start building with Gemini 2.5 Pro.](https://blog.google/products/gemini/gemini-preview-model-billing-update/) | Google's Gemini 2.5 Pro is now in public preview via the Gemini API on Google AI Studio, with Vertex AI availability coming soon.|
|[Benchmarking Web Agent Capabilities.](https://arxiv.org/abs/2504.01382v1) | Online-Mind2Web is a practical evaluation benchmark for autonomous web agents, revealing that current models underperform compared to prior assumptions due to issues with earlier benchmarks.|
|[VarGPT.](https://github.com/VARGPT-family/VARGPT-v1.1) | A unified autoregressive model that handles both understanding and synthesis tasks, enabling it to generate images as well as produce captions.|
|[FlexTok: Resampling Images into 1D Token Sequences of Flexible Length.](https://github.com/apple/ml-flextok) | Apple's open-source release builds on its recent paper, introducing a method to tokenize images using a variable number of tokens, allowing more complex images to be represented with more tokens.|
|[ZClip: Adaptive Spike Mitigation for LLM Pre-Training.](https://github.com/bluorion-com/ZClip) | ZClip employs EMA-based gradient norm statistics to dynamically suppress outlier gradients, avoiding loss spikes and enhancing training stability without relying on fixed thresholds.|
|[Goku Video Model.](https://saiyan-world.github.io/goku/) |Goku from ByteDance is a flow based video generation model of 2B and 8B parameters with 160M image and 36M video pairs. |
|[AI Index 2025: State of AI in 10 Charts.](https://hai.stanford.edu/news/ai-index-2025-state-of-ai-in-10-charts) |A clear, high-level, and thorough overview in 10 charts capturing the current landscape of AI, covering models, funding, and associated costs. |
|[Benchmarking Open Source models for OCR.](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr) | OCR involves recognizing text within images—a task that's difficult in rare cases but highly valuable when accurate. While closed models like the Gemini series excel at it, the latest Llama 4 models significantly advance the performance of open-source alternatives.|
|[DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level.](https://www.together.ai/blog/deepcoder) | Together AI has developed a coding model that rivals closed-source reasoning models. They've released the data, code, and training recipes, highlighting the model's impressive long-context capabilities.|
|[Hi Dream Image Generation Model.](https://github.com/HiDream-ai/HiDream-I1) |A powerful 17B parameter image generation model that leverages four distinct text encoders for generation, delivering strong overall results and released under a permissive license. |
|[A Framework for Dynamic Multi-Product Pricing.](https://arxiv.org/abs/2504.02324) |This paper presents a new dynamic multi-product pricing framework based on a censored multinomial logit model, where buyers only evaluate products priced below their personal valuations. |
|[MotifBench for Protein Design.](https://github.com/blt2114/MotifBench) |MotifBench is a benchmark for computational protein design, centered on the motif-scaffolding challenge by finding protein structures that support and stabilize specific target motifs. |
|[Arabic AI Benchmarks.](https://huggingface.co/blog/leaderboard-3c3h-aragen-ifeval) | Inception and MBZUAI have introduced a unified Arabic AI evaluation platform, featuring refreshed AraGen benchmarks and a new instruction-following leaderboard based on the Arabic IFEval benchmark.|
|[17K reasoning traces from R1.](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | A great set of reasoning traces from R1 that can be used as training data to distill a smaller reasoner or kick start the RL process.|
|[How Google Built the Pixel's Add Me Feature.](https://blog.google/products/pixel/how-google-built-pixel-add-me/) |The "Add Me" feature on Pixel devices leverages advanced image segmentation and AI for personalized video experiences. |
|[PaperBench: Evaluating AI's Ability to Replicate AI Research.](https://arxiv.org/abs/2504.01848) | OpenAI introduces PaperBench, a benchmark to evaluate whether AI agents can replicate cutting-edge machine learning research papers from scratch. The challenge requires agents to understand papers, build codebases, and run experiments to match results, with each paper accompanied by a detailed rubric. Evaluation is done using an LLM-based judge that scores with high agreement to human experts. The highest score was 21.0% by Claude 3.5 Sonnet, with no model surpassing 26.0%. ML PhDs scored 41.4% on a subset in 48 hours, showing humans still outperform in long-term tasks. A simplified Code-Dev version showed better results for o1 (43.4%). Models often struggled with early failure, lack of planning, and iteration, highlighting the importance of proper prompting and scaffolding.|
|[Command A: An Enterprise-Ready Large Language Model.](https://arxiv.org/abs/2504.00698) |Cohere introduces Command A, a 111B parameter open-weights LLM designed for enterprise tasks like RAG, agents, code, and multilingual applications. Command A uses a decentralized training pipeline where expert models are fine-tuned for specific domains and then merged, maintaining most expert performance with a minimal drop. Its hybrid architecture improves long-context efficiency, supporting 256k contexts with lower memory usage, and it outperforms peers in long-context benchmarks. Command A excels in agentic capabilities, surpassing GPT-4o and Claude 3.5 in multiple tests. It leads in real-world generative tasks and RAG use cases, with top scores in multilingual tasks, including dialect alignment and language consistency. The model also undergoes alignment with SRPO and RLHF, showing significant improvements in human alignment. Despite its size, Command A is efficient, running on just 2×A100s or H100s and generating 156 tokens/sec. Model weights are openly available on Hugging Face. |
|[Open Deep Search: Democratizing Search with Open-source Reasoning Agents.](https://arxiv.org/abs/2503.20201) |Researchers from Sentient, UW, Princeton, and UC Berkeley introduce Open Deep Search (ODS), an open-source AI framework that competes with proprietary systems like GPT-4o Search Preview and Perplexity Sonar. ODS consists of two components: the Open Search Tool, which refines web results through query rephrasing and reranking, and the Open Reasoning Agent, which orchestrates tool usage to answer queries. ODS-v2, built on DeepSeek-R1, outperforms GPT-4o Search Preview by 9.7% on FRAMES and offers better cost-efficiency. It also surpasses Perplexity Sonar on complex reasoning tasks. The addition of CodeAct in ODS-v2 allows the system to run Python code for improved reasoning and precision, offering more flexibility than the CoT-based ReAct in ODS-v1. |
|[Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models.](https://arxiv.org/abs/2503.24377) | This survey examines reasoning economy in LLMs, exploring how to balance deep reasoning performance with computational cost. It reviews inefficiencies, behavioral patterns, and potential solutions during both post-training and inference stages.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Cyberattacks by AI agents are coming.](https://www.technologyreview.com/2025/04/04/1114228/cyberattacks-by-ai-agents-are-coming) | AI agents are becoming powerful assets in cybersecurity, capable of carrying out sophisticated attacks and scaling operations such as ransomware. The LLM Agent Honeypot project seeks to identify these agents by mimicking vulnerable servers, showing that agents are more adaptable and evasive than typical bots. Experts expect a rise in agent-led cyberattacks and emphasize the need to proactively build defenses as the technology advances.|
|[The artifact isn’t the art: Rethinking creativity in the age of AI.](https://www.freethink.com/opinion/studio-ghibli-chatgpt-creativity) | 
AI-generated Ghibli-style visuals have surged in popularity, straining OpenAI's servers and sparking debates about creativity in the AI age. While AI can rapidly produce artistic images, it lacks the human ability to experience and synthesize complex ideas and emotions. The future of creativity will focus on meaningful outputs shaped by human insight and purpose, with AI as a tool rather than a creator.|
|[How does the brain control consciousness? This deep-brain structure.](https://www.nature.com/articles/d41586-025-01021-2) | In a world of constant stimulation, the thalamus filters which thoughts we become aware of and which we don’t.|
|[AI for research: the ultimate guide to choosing the right tool.](https://www.nature.com/articles/d41586-025-01069-0) |Curious about using artificial intelligence to boost your research? Here are the programs you shouldn’t miss. |
|[AI race in 2025 is tighter than ever before.](https://www.nature.com/articles/d41586-025-01033-y) |State of the industry report also shows that 2024 was a breakthrough year for small, sleek models to rival the behemoths. |
|[Why more AI researchers should collaborate with governments.](https://www.nature.com/articles/d41586-025-01063-6) | Academics can drive policy innovation — but they must shift their focus from publishing papers to creating practical products.|
|[Why an overreliance on AI-driven modelling is bad for science.](https://www.nature.com/articles/d41586-025-01067-2) | Without clear protocols to catch errors, artificial intelligence’s growing role in science could do more harm than good.|
|[Beyond the binary: Navigating AI’s uncertain future in Africa.](https://www.science.org/doi/10.1126/science.adw9439) |The artificial intelligence (AI) debate is increasingly polarized in Africa, mirroring a trend across the globe. On one side, utopian headlines, such as “5 Ways To Harness AI And End Poverty Forever,” claim that AI will revolutionize development. On the other, warnings that “AI Is Bad News for the Global South” paint the technology as an inevitable amplifier of inequality and exploitation. |
|[The composer still making music four years after his death – thanks to an artificial brain.](https://www.theguardian.com/artanddesign/2025/apr/09/alvin-lucier-dead-composer-making-music-ai-artificial-intelligence-brain) | In Australia, a team of artists and scientists have resurrected the US composer Alvin Lucier. It raises a storm of questions about AI and authorship – and it’s also incredibly beautiful|
|[AlphaFold is running out of data — so drug firms are building their own version.](https://www.nature.com/articles/d41586-025-00868-9) | Thousands of 3D protein structures locked up in big-pharma vaults will be used to create a new AI tool that won’t be open to academics.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |




















































































































