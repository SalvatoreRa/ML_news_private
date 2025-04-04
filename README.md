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
|[Tracing the thoughts of a large language model.](https://www.anthropic.com/research/tracing-thoughts-language-model) | Anthropic researchers introduce new interpretability tools for examining LLMs, using Claude 3.5 Haiku as a testbed. Their studies reveal insights into model internals, such as circuits, plans, and conceptual thinking. Key findings include Claude’s multilingual “language of thought,” where concepts like “small” are processed similarly across languages, enabling transfer learning. Claude also plans ahead, even in poetry, and computes sums with parallel circuits, explaining answers using human-style logic. The tools help detect unfaithful reasoning, where Claude fabricates steps to fit answers. Researchers can also intervene in multi-step reasoning, showing that Claude’s reasoning is dynamic. The tools also reveal that Claude’s hallucinations are caused by misfires in circuits and that jailbreaks can bypass safety features temporarily.|
|[Harmful Fine-Tuning Attacks.](https://arxiv.org/abs/2501.18100v1) |Researchers have identified weaknesses in current defenses against harmful fine-tuning attacks and introduced Panacea, an adaptive perturbation method that maintains model safety without compromising fine-tuning performance. |
|[AgentRxiv.](https://arxiv.org/abs/2503.18102) | Researchers from Johns Hopkins and ETH Zurich introduce AgentRxiv, a framework that allows LLM agents to autonomously generate and share research papers, similar to how human scientists collaborate. The system functions like an open-source preprint server for agents, enabling labs to upload, search, and refine papers iteratively. Using this framework, a single agent improved GPT-4o mini’s accuracy by 11.4% on the MATH-500 benchmark through better prompt strategies. The framework also improved other benchmarks, showing consistent performance gains across multiple LLMs. Collaboration between agent labs led to faster progress, with higher accuracy achieved by sharing results via AgentRxiv. Agents refine their own ideas without plagiarism, but the system requires further improvements in reliability and novelty guarantees.|
|[Neural Alignment via Speech Embeddings.](https://www.nature.com/articles/s41562-025-02105-9) |Google Research and collaborators reveal significant similarities between LLM embeddings and human brain activity during conversation. Their findings show that embeddings from OpenAI's Whisper model align with brain signals in regions responsible for speech, language, and motor planning. The study suggests a "soft hierarchy" in brain areas, with overlapping processing of speech and language. Brain regions also predict upcoming words, mirroring autoregressive LLM behavior. Additionally, the geometry of word relationships in brain activity reflects that of LLM embeddings, indicating convergent structures in language representation. Despite architectural differences—brains process speech serially, while Transformers process in parallel—these studies highlight potential for using LLMs to reverse-engineer the brain’s language mechanisms and inspire more brain-like AI models. |
|[Unlearning Sensitive Content from LLMs.](https://arxiv.org/abs/2503.21088v1) |This paper introduces a model merging technique that enables selective forgetting of sensitive content in large language models while retaining their general knowledge. |
|[Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models.](https://arxiv.org/abs/2503.16779) |The paper introduces Chain-of-Tools (CoTools), a method allowing LLMs to incorporate external tools, including unseen ones, while maintaining chain-of-thought (CoT) reasoning. CoTools keeps the LLM’s parameters frozen and fine-tunes additional modules (Tool Judge and Tool Retriever) to interact with a wide array of tools. It represents tools as semantic vectors, allowing even unfamiliar tools to be used without retraining the model. CoTools integrates tool calls within the reasoning process, selecting the best tool from many based on query context, improving accuracy on complex tasks. Experiments on various benchmarks show significant improvements in tool-selection accuracy and overall performance, with CoTools successfully handling large and unseen toolsets. |
|[Structured Memory Augmentation for Smarter LLM Agents.](https://arxiv.org/abs/2503.21760v1) | MemInsight is a framework that autonomously enhances and organizes memory for LLM agents, improving context retention and retrieval. It uses a backbone LLM to mine and structure memory attributes from past conversations, organizing them into entity and conversation-centric augmentations. MemInsight outperforms traditional retrieval methods, achieving up to 34% higher recall on the LoCoMo QA dataset compared to Dense Passage Retrieval (RAG). It also improves movie recommendations by matching genres and reducing memory size by 90%, while increasing persuasive outputs by 12%. MemInsight can summarize long conversations using memory alone, achieving coherence similar to raw-dialogue baselines. The system shows minimal hallucinations and stable performance, particularly when using carefully selected models for memory augmentation.|
|[Anthropic Economic Index: Insights from Claude 3.7 Sonnet.](https://www.anthropic.com/news/anthropic-economic-index-insights-from-claude-sonnet-3-7) |Anthropic has launched the Economic Index, leveraging Claude Sonnet 3.7 to evaluate AI's impact across various job sectors. The model analyzes productivity changes, automation risks, and labor market trends, offering data-driven insights to help policymakers and businesses adapt to AI-driven economic transformations. |
|[Investigating Affective Use and Emotional Well-being on ChatGPT.](https://cdn.openai.com/papers/15987609-5f71-433c-9972-e91131f399a1/openai-affective-use-study.pdf) |Researchers from OpenAI and MIT Media Lab examine the impact of emotionally engaging interactions with ChatGPT, particularly in Voice Mode, on user well-being. They combine a large-scale analysis of 4M+ conversations and 4,000+ surveys with a randomized controlled trial (RCT) involving 981 participants. The studies reveal that high usage, especially with voice interactions, is linked to emotional dependence, a preference for chatbot interactions, and discomfort with changes in voice/personality. Voice mode had mixed effects, improving well-being for some, but long-term usage led to increased loneliness and problematic use. A small group of users drove most emotionally charged conversations, forming pseudo-relationships with the model. Automated classifiers developed to analyze conversations mirrored user self-reports and highlighted the need for socioaffective alignment, urging developers to design models that support users without exploiting emotional needs. |
|[PLAY2PROMPT: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play.](https://arxiv.org/abs/2503.14432) |Researchers from MIT CSAIL and IBM introduce Play2Prompt, a framework that enables LLM agents to learn how to use external tools in a zero-shot manner, without needing labeled examples or high-quality documentation. Play2Prompt discovers tool usage patterns through trial-and-error API calls, generating query-answer pairs based on successful tool invocations. The system iteratively refines tool demonstrations and documentation via self-reflective beam search and rejection sampling. Play2Prompt shows strong zero-shot performance, improving accuracy by 5-7% on benchmark tasks and even boosting GPT-4o by up to 3.3%. It remains robust even with poor documentation and outperforming methods like EasyTool that rely on labeled examples, particularly in challenging tool settings. |
|[Global modules robustly emerge from local interactions and smooth gradients.](https://www.nature.com/nature/volumes/640/issues/8057) |The principle of peak selection is described, by which local interactions and smooth gradients drive self-organization of discrete global modules. |
|[Evolutionary optimization of model merging recipes.](https://www.nature.com/articles/s42256-024-00975-8) | Akiba et al. developed an evolutionary approach to automatically merge artificial intelligence models, creating powerful hybrid models without extensive training. The method produces models with enhanced mathematical and visual capabilities that outperform larger models.|
|[Enhanced Cell Segmentation.](https://arxiv.org/abs/2504.00784v1) |CellVTA improves vision transformer-based models for cell instance segmentation by injecting high-resolution spatial features through a CNN-based adapter, achieving state-of-the-art performance on multiple datasets. |
|[Synthetic Data Generation Using Large Language Models: Advances in Text and Code.](https://arxiv.org/abs/2503.14023) |LLMs are increasingly employed to generate synthetic training data for language and code tasks, enhancing performance in low-resource settings through methods like prompt-based generation and self-refinement. The paper outlines advantages such as reduced cost and broader coverage, while also addressing challenges like factual inaccuracies and bias. It proposes mitigations and highlights future research directions in prompt automation and data quality evaluation. |
|[Current and Future Use of LLMs for Knowledge Work.](https://arxiv.org/abs/2503.16774v1) |A two-part survey of 216 and 107 participants shows that knowledge workers currently use LLMs for tasks such as code generation and text enhancement, but anticipate more integrated use within workflows and data systems. The results provide insights for shaping future design and adoption of generative AI in professional environments. |
|[.]() | |
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
|[Trump to consider final proposal on TikTok future as US ban deadline looms.](https://www.theguardian.com/technology/2025/apr/02/trump-to-consider-final-proposal-on-tiktok-future-as-us-ban-deadline-looms) | Owner ByteDance required to find non-Chinese buyer for video app’s American operations by Saturday|
|[UK needs to relax AI laws or risk transatlantic ties, thinktank warns.](https://www.theguardian.com/technology/2025/apr/02/uk-ai-copyright-laws-transatlantic-tony-blair-thinktank) | Tony Blair Institute says enforcing stricter licensing rules for copyright-protected material will threaten national security interests| 
|[OpenAI raises $40bn in deal with SoftBank that values it at $300bn.](https://www.theguardian.com/technology/2025/apr/01/openai-raises-up-to-us40bn-in-deal-with-softbank) |Japanese investor to put $10bn at first into OpenAI and $30bn more by end of 2025 if certain conditions are met |
|[xAI acquires X in $80B all-stock deal.](https://threadreaderapp.com/thread/1905731750275510312.html) | xAI has officially acquired X in an all-stock transaction that values the combined company at over $110 billion.|
|[Gemini 2.5: Our most intelligent AI model.](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) |Gemini 2.5 Pro, an advanced AI model, is topping LMArena benchmarks by a wide margin. It boosts performance and accuracy through enhanced reasoning, analyzing information and making informed decisions. The model builds on the advancements of Gemini 2.0 Flash Thinking. |
|[Announcing ARC-AGI-2 and ARC Prize 2025.](https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025) |The ARC Prize has introduced ARC-AGI-2, a demanding benchmark designed to push the development of general AI systems. Current models perform well below human levels. The ARC Prize 2025 competition, hosted on Kaggle with a $1 million prize pool, encourages open-source innovation by rewarding efficient and capable solutions to ARC-AGI-2 tasks. |
|[OpenAI reshuffles leadership as Sam Altman pivots to technical focus.](https://www.theverge.com/openai/634802/openai-leadership-change) |In a significant executive shuffle announced Monday, OpenAI is expanding COO Brad Lightcap’s responsibilities while CEO Sam Altman shifts his attention more toward the company’s technical direction. |
|[Tim Cook says China’s DeepSeek AI is ‘excellent’ during visit to country.](https://9to5mac.com/2025/03/24/tim-cook-says-chinas-deepseek-ai-is-excellent-during-visit-to-country/) |Despite DeepSeek AI's security and privacy issues, Tim Cook praised it as "excellent" during his China visit. The AI, developed in China, rivals top global models at lower development costs but faces investigations in the US and Europe. Cook, who is attending the China Development Forum, often has to make diplomatic remarks about China due to Apple's business interests there. |
|[Google's AI-focused Education Tools AI.](https://blog.google/outreach-initiatives/education/ai-literacy-day-2025/) |Google's new AI-focused educational tools offer training for educators, resources for students, and broader access to Gemini for younger users. |
|[Microsoft announces security AI agents to help overwhelmed humans.](https://www.theverge.com/news/634598/microsoft-security-copilot-ai-agents) | Microsoft has introduced six AI-powered security agents for its Security Copilot to help teams handle phishing and data loss incidents more efficiently.|
|[Perplexity CEO Addresses Financial Rumors.](https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/message_from_aravind_cofounder_and_ceo_of/) |Perplexity CEO Aravind Srinivas has denied financial trouble rumors, stating the company has healthy funding and no IPO plans before 2028. |
|[Amazon Nova Act .](https://labs.amazon.science/blog/nova-act) | Amazon has launched Nova Act, an AI model that enables agents to operate within web browsers. A research preview SDK is available, allowing developers to build agents capable of executing complex, multi-step tasks by decomposing them into atomic commands and manipulating browser actions for greater reliability. Nova Act is designed to extend agent capabilities beyond basic API tasks, boosting business productivity and task automation.|
|[Runway releases an impressive new video-generating AI model.](https://techcrunch.com/2025/03/31/runway-releases-an-impressive-new-video-generating-ai-model/) | Runway has released its next-generation video model, which excels at prompt adherence and cinematic motion generation.|
|[OpenAI to release an Open Weight model.](https://openai.com/open-model-feedback/) | OpenAI is soliciting feedback for an open weight model that has reasoning.|
|[Earth AI’s algorithms found critical minerals in places everyone else ignored.](https://techcrunch.com/2025/03/25/earth-ais-algorithms-found-critical-minerals-in-places-everyone-else-ignored/) |Earth AI has identified promising mineral deposits in previously neglected areas of Australia through AI-driven analysis. Unlike traditional techniques, its technology rapidly scans vast regions to pinpoint potential sources of minerals such as copper and cobalt, marking a shift toward more efficient, AI-powered exploration in the mining industry. |
|[Quora’s Poe launches its most affordable subscription plan for $5/month.](https://techcrunch.com/2025/03/25/quoras-poe-now-offers-an-affordable-subscription-plan-for-5-month/) |Quora's chatbot app, Poe, launched new subscription plans at $5/month for 10,000 daily points and $250/month for 12.5 million points. |
|[Nvidia's AI assistant is here to optimize your gaming PC.](https://www.theverge.com/news/635155/nvidia-g-assist-ai-assistant-available-download) | Nvidia's Project G-Assist is a real AI assistant for RTX GPU owners that optimizes game settings, measures frame rates, and controls accessory lighting.|
|[Nvidia is reportedly in talks to acquire Lepton AI.](https://techcrunch.com/2025/03/26/nvidia-is-reportedly-in-talks-to-acquire-lepton-ai/) | The semiconductor giant is reportedly nearing a deal to acquire Lepton AI, a company that rents out servers that are powered by Nvidia’s AI chips|
|[OpenAI Announces $40B in New Funding.](https://openai.com/index/march-funding-updates/) |OpenAI has secured $40 billion in funding at a $300 billion valuation to advance AI research, scale infrastructure, and support its expanding user base. The company has also partnered with SoftBank to further accelerate AGI development. |
|[Gemini Robotics from Google DeepMind.](https://blog.google/products/gemini/how-we-built-gemini-robotics/) | Google DeepMind has unveiled its Gemini Robotics models, extending Gemini 2.0 with fine-tuning capabilities for executing physical actions.|
|[Nexthop AI Locks up $110M Led by Lightspeed.](https://news.crunchbase.com/ai/infrastructure-nexthop-venture-funding-lightspeed/) |Nexthop AI has raised $110 million in a funding round led by Lightspeed Venture Partners to advance networking solutions for hyperscalers, with a focus on cost and power efficiency. The round also included investments from Kleiner Perkins, WestBridge Capital, Battery Ventures, and Emergent Ventures. CEO Anshul Sadana highlighted the company's mission to innovate in collaboration with cloud providers. |
|[Alibaba Head Warns AI Industry Is Showing Signs of Bubble.](https://futurism.com/alibaba-ai-industry-signs-bubble) |Alibaba chairman Joe Tsai has cautioned about a possible AI bubble, citing massive data center investments without clear demand. With $52 billion already committed to AI development, concerns are growing over potential overinvestment. Recent events, such as turbulence around Chinese startup DeepSeek, have fueled investor anxiety about overpaying in the AI space. |
|[Amazon’s Alexa Fund is now backing AI startups.](https://techcrunch.com/2025/03/26/amazon-alexa-fund-invests-into-four-new-startups-as-it-plans-to-invest-more-into-ai-solutions/) |Amazon's Alexa Fund is expanding its investment focus to include AI startups, investing in companies like NinjaTech AI, Hedra, Ario, and HeyBoss. |
|[Life-giving oxygen is wafting out of lakes worldwide.](https://www.nature.com/articles/d41586-025-00876-9) |Machine-learning method shows declining oxygen levels in thousands of lakes as their waters warm. |
|[Mathematician who reshaped theory of symmetry wins Abel Prize.](https://www.nature.com/articles/d41586-025-00949-9) |Masaki Kashiwara is the first Japanese person to be awarded the most prestigious prize in mathematics. |
|[‘Meta has stolen books’: authors to protest in London against AI trained using ‘shadow library’.](https://www.theguardian.com/books/2025/apr/03/meta-has-stolen-books-authors-to-protest-in-london-against-ai-trained-using-shadow-library) |Writers will gather at the Facebook owner’s King’s Cross office in opposition to its use of the LibGen database to train its AI models |
|[Anthropic's LLM for Education.](https://www.anthropic.com/news/introducing-claude-for-education) |Anthropic has launched Claude for Education, featuring tools like Learning Mode to encourage critical thinking. The initiative also includes broad university access through collaborations with major institutions and educational platforms. |
|[Claude Available for U.S. Government Use.](https://www.anthropic.com/news/claude-on-google-cloud-fedramp-high) | Claude has attained FedRAMP High and IL-2 compliance via Google Cloud's Vertex AI, enabling its use by federal agencies and defense organizations with stringent security requirements.|
|[Meta and UFC partner to enhance fan engagement with AI and VR.](https://www.fastcompany.com/91310521/meta-ufc-partnership-multiyear-deal-ai-vr-fans) |Meta Platforms and the Ultimate Fighting Championship (UFC) have formed a multiyear partnership to bring Meta's AI and VR technologies to UFC events. The collaboration aims to deliver immersive fan experiences through devices like Meta Quest and AI-powered glasses. Meta's branding will appear in the Octagon during fights, and Threads will be the official social media partner. Financial details of the deal were not disclosed. |
|[Satya Nadella: DeepSeek is the new bar for Microsoft’s AI success.](https://www.theverge.com/notepad-microsoft-newsletter/637496/microsoft-satya-nadella-deepseek-chatgpt-ai-investments-notepad) |Microsoft CEO Satya Nadella stressed the need to translate AI research into successful products after an $80 billion investment in AI. The company is prioritizing enhancements to offerings like Copilot and Muse while ensuring its AI efforts align with sustainability goals. Despite the growing demands of AI workloads, Microsoft remains committed to becoming carbon-negative by 2030. |
|[Alphabet’s AI drug discovery platform Isomorphic Labs raises $600M from Thrive.](https://techcrunch.com/2025/03/31/alphabets-ai-drug-discovery-platform-isomorphic-labs-raises-600m-from-thrive/) |Isomorphic Labs, a DeepMind spinout, has raised $600 million from Thrive Capital to advance its AI-driven drug design platform. The funding will help expand its research team and move discovered drugs into clinical trials. The company also has partnerships with Eli Lilly and Novartis to leverage its AI model in pharmaceutical development. |
|[NotebookLM Adds Web-Based Source Discovery.](https://blog.google/technology/google-labs/notebooklm-discover-source) |Google's NotebookLM now features a Discover tool that gathers curated web sources based on user-defined topics, making research and information collection more efficient. |
|[AI-Powered Conversational Videos.](https://ai.meta.com/blog/tavus-real-feeling-ai-videos-llama/) | Tavus leverages Llama 3.3 to enable realistic AI-generated video conversations, integrating visual question-answering and multi-image reasoning to create lifelike digital interactions.|
|[Blanket ban on teen smartphone use ‘potentially detrimental’, says academic.](https://www.theguardian.com/technology/2025/apr/03/blanket-ban-on-teen-smartphone-use-potentially-detrimental-says-academic) |Dr Amy Orben says there are no ‘one-size-fits-all answers’ given importance of access to online information |
|[Meta faces £1.8bn lawsuit over claims it inflamed violence in Ethiopia.](https://www.theguardian.com/technology/2025/apr/03/meta-faces-18bn-lawsuit-over-claims-it-inflamed-violence-in-ethiopia) |Son of murdered academic calls on Facebook owner to ‘radically change how it moderates dangerous content’ |
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
|[Qwen2.5-Omni.](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/assets/Qwen2.5_Omni.pdf) | Qwen2.5-Omni is an end-to-end multimodal model capable of perceiving and understanding text, audio, images, and video, while generating both text and speech in real-time. It features the Thinker-Talker architecture, where Thinker handles perception and text generation, and Talker generates speech, trained together for synchronized output. The model’s streaming-first design uses block-wise encoders and TMRoPE for real-time interaction. Trained on over 1.2 trillion tokens, Qwen2.5-Omni is fine-tuned for natural speech and performs well across multiple modalities. It achieves state-of-the-art results on OmniBench, outperforms previous models in ASR and TTS, and significantly closes the gap in voice-text instruction following.|
|[Test-Time Visual In-Context Tuning.](https://arxiv.org/abs/2503.21777) | A new method enables test-time adaptation of VICL models using only the test sample, enhancing generalization across different visual tasks under domain shifts.|
|[High-Fidelity Simultaneous Speech-To-Speech Translation.](https://arxiv.org/abs/2502.03382) |Kyutai has unveiled its latest audio system, a real-time audio-to-audio translation tool powered by a robust multi-stream transformer. It features expressive voice capabilities, delivering impressive performance in speech translation. |
|[Mobile-VideoGPT.](https://github.com/amshaker/mobile-videogpt) |A compact multimodal video model with under 1B parameters, incorporating dual visual encoders and token pruning to enable real-time inference on edge devices. |
|[Multimodal Adaptation Methods.](https://github.com/donghao51/awesome-multimodal-adaptation) | A curated list of methods for multimodal adaptation, including traditional domain adaptation, test-time adaptation, and recent innovative approaches.|
|[ReAG - Reasoning Augmented Generation.](https://github.com/superagent-ai/reag) |Traditional Retrieval-Augmented Generation (RAG) systems use a two-step approach: semantic search retrieves documents based on surface-level similarity, followed by a language model generating responses from those documents. While effective, this often overlooks deeper context and introduces irrelevant information. ReAG—Reasoning Augmented Generation—proposes a stronger alternative by feeding raw documents directly into the language model, enabling it to process and integrate the full context. This unified method results in more accurate, nuanced, and context-aware outputs. |
|[Awesome Vision-to-Music Generation.](https://github.com/wzk1015/awesome-vision-to-music-generation) |A curated and regularly updated list of methods, datasets, and demos focused on converting visual inputs into music (V2M), showcasing both academic and industrial advancements in the field. |
|[Video Generation Faithfulness Benchmark.](https://arxiv.org/abs/2503.21755) | A benchmark designed to evaluate how accurately video generation aligns with the given prompt. It also introduces methods to improve the quality of generated videos in relation to the user's input prompt.|
|[Optimal Stepsize in Diffusion Models.](https://github.com/bebebe666/optimalsteps) |Optimal Stepsize for Diffusion Sampling (OSS) improves diffusion model sampling by learning efficient stepsize schedules using dynamic programming, achieving a 10× speedup with minimal loss in generation quality. |
|[SAMWISE video segmentation.](https://github.com/ClaudiaCuttano/SAMWISE) | This work gives SAM 2 open vocabulary segmentation and more precise semantic tracking over long videos.|
|[Orpheus.](https://github.com/freddyaboulton/orpheus-cpp) | Orpheus is a text-to-speech system. It is easy to install and runs without a GPU, similar to Llama cpp.|
|[Video-R1.](https://github.com/tulerfeng/video-r1) |Video-R1 presents a rule-based reinforcement learning method for video reasoning, utilizing a temporal variant of GRPO and introducing new datasets. It is efficiently trainable on 4 H20 or 5 A100 GPUs. |
|[Fast Text-to-3D.](https://theericma.github.io/TriplaneTurbo/) |Progressive Rendering Distillation enables training 3D generators from text prompts without ground-truth meshes, producing high-quality 3D meshes in just 1.2 seconds and outperforming previous approaches. |
|[TIDE for Underwater Scene Understanding.](https://hongklin.github.io/TIDE/) |A text-to-image and dense annotation generation method for underwater scenes that produces high-quality synthetic datasets with consistent pixel-level labels. |
|[OpenAI launches OpenAI Academy, a free AI learning platform for everyone.](https://techstartups.com/2025/04/01/openai-launches-openai-academy-a-free-ai-learning-platform-for-everyone/) |OpenAI has introduced OpenAI Academy, a free platform offering AI courses that range from beginner content to advanced subjects like AI safety and governance. Designed for diverse audiences, the platform seeks to expand access to AI education and promote thoughtful engagement with its societal implications. Early feedback praises its accessibility and thorough approach to making AI more understandable worldwide. |
|[Video Motion Segmentation.](https://motion-seg.github.io/) |Building on recent trends in tracking systems, this work incorporates dense pixel tracking to enhance long-term segmentation using Dino and SAM2. |
|[Open Hands Coding Model.](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model) |A powerful 32B model fine-tuned with reinforcement learning on top of Qwen, outperforming many much larger models on agentic coding tasks. |
|[Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model.](https://arxiv.org/abs/2503.24290) | There’s a major open question in RL reasoning around whether a sufficiently strong base model is essential for emergent reasoning. This work explored various cases of scaling RL on a base model and found that strong base models significantly aid reasoning convergence.|
|[Easi3R: Estimating Disentangled Motion from DUSt3R Without Training.](https://github.com/Inception3D/Easi3R) | Easi3R is a 3D vision system designed to more accurately estimate 3D scenes with high motion. It significantly outperforms previous methods in full scene reconstruction by masking moving objects and learning them separately from the background.|
|[Benchmark for RL-based Video Understanding.](https://github.com/tencentarc/seed-bench-r1) | SEED-Bench-R1 is a benchmark designed to assess post-training methods such as RL and SFT for multimodal LLMs on complex video-based tasks. It highlights RL's advantages in perception and data efficiency while also revealing its difficulties in maintaining logical coherence.|
|[Flow Prediction for Autonomous Driving.](https://github.com/tasl-lab/uniocc) | UniOcc is a unified framework for forecasting and flow prediction in driving scenarios, designed for multi-dataset training and cross-domain evaluation across both real and synthetic environments.|
|[Paper Bench.](https://openai.com/index/paperbench/) |OpenAI has introduced a new benchmark for academic paper creation that involves fully replicating selected papers. This includes comprehending their experiments and results, as well as generating original ideas, to evaluate deeper understanding and creativity in AI models. |
|[GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors.](https://geometrycrafter.github.io/) |A powerful model that leverages video diffusion as a prior for consistent geometry estimation over time. It operates at approximately 1.5 FPS for full point cloud estimation and also performs accurate camera pose estimation. |
|[DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness.](https://ruiningli.com/dso) |Most 3D synthesized data is created with a focus on aesthetic quality, which often results in models that can't stand or support themselves in physics-based environments. This work introduces slight fine-tuning to improve the physical plausibility of these models. |
|[Medical Reasoning Dataset.](https://github.com/UCSC-VLAA/MedReason) | A large-scale medical reasoning dataset designed to enable faithful and explainable problem-solving in LLMs, aiming to advance research in medical AI.|
|[DeepMind's Study for Kernel Fuzzing.](https://sishuaigong.github.io/pdf/asplos25-snowplow.pdf) | Snowplow is a kernel fuzzer that uses a learned white-box mutator to enhance test mutation efficiency, leading to significantly improved code coverage and increased bug discovery in Linux kernel fuzzing. |
|[The hottest AI models, what they do, and how to use them.](https://techcrunch.com/2025/03/30/the-hottest-ai-models-what-they-do-and-how-to-use-them/) | This article reviews the leading AI models released since 2024, showcasing their applications and strengths. Key highlights include OpenAI's GPT-4.5 Orion for its robust world knowledge, Google's Gemini 2.5 Pro for its coding capabilities, and Cohere's Aya Vision, which stands out in image-related tasks. The overview helps simplify understanding the fast-changing AI landscape.|
|[DeepSite open source canvas.](https://huggingface.co/spaces/enzostvs/deepsite) |DeepSite is a DeepSeek powered open source canvas for “vibe coding” that updates apps in real time while the system writes the code. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Tools and Weapons: Microsoft's Story, Told by Its CEOs.](https://app.magellan.ai/listen_links/tldr) |Hosted by Microsoft Vice Chair and President Brad Smith, the *Tools and Weapons* podcast examines the global impact of technology. In recent episodes, Bill Gates, Steve Ballmer, and Satya Nadella reflect on Microsoft's 50-year journey, discussing its past, present, and future.|
|[AI-powered therapy shows shocking results in mental health study.](https://interestingengineering.com/health/groundbreaking-ai-therapy-shows-positive-results) | A Dartmouth study found that the AI therapy chatbot Therabot significantly alleviated symptoms in participants dealing with depression, anxiety, and eating disorders.|
|[Databricks Has a Trick That Lets AI Models Improve Themselves.](https://www.wired.com/story/databricks-has-a-trick-that-lets-ai-models-improve-themselves/) |Databricks has introduced Test-time Adaptive Optimization (TAO), a technique that uses reinforcement learning and synthetic data to enhance AI models without relying on clean labeled data. |
|[‘Open source’ AI isn’t truly open — here’s how researchers can reclaim the term.](https://www.nature.com/articles/d41586-025-00930-6) | Many firms are misusing the ‘open source’ label. The scientific community, which relies on transparency and replicability, must resist this trend.|
|[Transparency (in training data) is what we want.](https://www.nature.com/articles/s42256-025-01023-9) |As more powerful generative AI tools appear on the market, legal debates about the use of copyrighted content to develop such tools are intensifying. To resolve these issues, transparency regarding which copyrighted data have been used and where in the AI training pipeline needs to be a starting point. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |




















































































































