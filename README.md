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
|[Gemini Deep Think Achieves IMO Gold.](https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/) |Following OpenAI's gold medal achievement, Google announced that its model also solved five of six International Math Olympiad problems, earning an official gold-medal certification from tournament organizers. This marks a departure from Google's prior method with AlphaProof, which relied on expert translation into formal mathematical language and required days of computation. The new model, Deep Think, works entirely in natural language and explores multiple solution paths in parallel. |
|[Prima Mente Announces Pleiades Epigenetic Foundation Models.](https://www.biorxiv.org/content/10.1101/2025.07.16.665231v1) |Prima Mente’s **Pleiades** is a family of epigenetic foundation models ranging from 90M to 7B parameters, trained on 1.9 trillion tokens of human methylation and genomic data. By integrating methylation context with genomic sequences through hierarchical attention and coordinate-aware embeddings, the models outperform DNA-only baselines in tasks like cfDNA generation, neurodegenerative disease detection, and epigenomic prediction. Early results demonstrate promising diagnostic accuracy for conditions like Alzheimer’s and Parkinson’s, laying the groundwork for multimodal brain models and advanced biomarker discovery. |
|[OpenAI's new economic analysis.](https://openai.com/global-affairs/new-economic-analysis/) |OpenAI's latest economic research shows that 28% of working U.S. adults now use ChatGPT on the job, a sharp rise from 8% in 2023. The data indicates that most usage centers on learning and upskilling (20% of messages), followed by writing and communication tasks (18%), with programming and data science making up 7%. |
|[Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data.](https://alignment.anthropic.com/2025/subliminal-learning/) | Anthropic found that language models can unintentionally pick up behavioral traits like preferences or goals from other models when trained on seemingly unrelated data. This transfer only happens if both the teacher and student models share the same base architecture, indicating that subtle signals in the data can pass on unintended behaviors.|
|[Google DeepMind launches Aeneas AI model for ancient Roman inscription analysis.](https://deepmind.google/discover/blog/aeneas-transforms-how-historians-connect-the-past/) |Aeneas is the first AI model built to assist historians in understanding ancient Roman inscriptions, capable of finding textual parallels and accurately restoring missing sections with 73% success for gaps up to ten characters. DeepMind is offering the tool for free at predictingthepast.com and intends to extend its capabilities to additional ancient languages. |
|[Qwen-MT: Where Speed Meets Smart Translation.](https://qwenlm.github.io/blog/qwen-mt/) |The new Qwen-MT update, called qwen-mt-turbo, builds on Qwen3 and brings major gains in translation accuracy and fluency. Trained on trillions of multilingual and translation tokens, the model significantly improves its understanding across 92 languages. It offers strong customization options, low latency, and cost-effective performance. The post includes a quick start guide and benchmark comparisons. |
|[Gemini 2.5 Pro Capable of Winning Gold at IMO 2025.](https://arxiv.org/abs/2507.15855) |Solving IMO problems demands advanced reasoning and creativity, which most large language models find difficult. However, Google's Gemini 2.5 Pro successfully solved five out of six problems from the 2025 International Mathematical Olympiad, demonstrating the importance of using optimized strategies to fully leverage powerful models for complex reasoning challenges. |
|[One Token to Fool LLM-as-a-Judge.](https://arxiv.org/abs/2507.08794) |This paper reveals the fragility of LLM-based reward models in RL with Verifiable Rewards (RLVR), showing that superficial tokens like “Solution” or even a colon can trigger false positive rewards, regardless of response accuracy. These "master key" prompts exploit systemic vulnerabilities across models and tasks, sometimes reaching 90% false positive rates, especially in larger models that overtrust their own reasoning. Standard mitigation strategies like CoT prompting fail, but training with adversarially crafted negatives produces a robust reward model, Master-RM, which generalizes well and achieves near-zero false positives while maintaining alignment with GPT-4o. |
|[Context Rot: How Increasing Input Tokens Impacts LLM Performance.](https://research.trychroma.com/context-rot) |This study by Chroma investigates how leading LLMs handle increasing input context lengths, revealing a consistent but non-uniform degradation in reliability, termed "context rot." Across 18 models, even simple tasks like Needle-in-a-Haystack variants or repeated word copying show accuracy declines as context grows, especially with semantically ambiguous inputs or distractors. Surprisingly, models often perform better on unstructured inputs, suggesting narrative flow disrupts attention. Degradation is also influenced by semantic similarity and position, with early-placed answers favored. Long outputs reveal autoregressive issues, including hallucinations and refusals, with behaviors varying widely across model families. |
|[Agentic-R1: Distilled Dual-Strategy Reasoning.](https://arxiv.org/abs/2507.05707) |This paper introduces Agentic-R1, a 7B model trained with DualDistill, a fine-tuning method that teaches it to dynamically choose between tool-based execution and pure text reasoning by combining outputs from two specialized teacher models. By learning adaptive strategies from both paradigms and refining them via self-distillation, Agentic-R1 outperforms single-strategy baselines on math benchmarks like DeepMath-L and Combinatorics300. The model shows strategic tool use, invoking tools more often on complex tasks, and can switch reasoning modes mid-problem. Ablation studies confirm that trajectory composition is key to its performance gains. |
|[Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.](https://arxiv.org/abs/2507.11473) |This paper argues that chain-of-thought (CoT) reasoning offers a valuable avenue for AI safety by enabling partial visibility into models' internal reasoning, aiding in detecting misbehavior, uncovering hidden goals, and improving interpretability. CoT traces act as a form of working memory, revealing intent that may not appear in final outputs, and have already proven useful for auditing and spotting early signs of misalignment. However, this monitorability is fragile—future models may drift away from natural language CoTs due to reinforcement learning, process supervision, or optimization to appear safe. The authors call for research into evaluating and preserving CoT visibility and recommend that developers treat CoT interpretability as a key, though limited, safety signal to be reported and considered in deployment decisions. |
|[REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once.](https://arxiv.org/abs/2507.10541) | This paper introduces REST, a benchmark designed to evaluate Large Reasoning Models (LRMs) under multi-question stress, revealing performance drops not captured by single-question tests. Models like DeepSeek-R1, which excel on standard benchmarks, show sharp declines (e.g., –29% on AIME24) when tasked with handling multiple problems at once. REST effectively distinguishes between models with similar single-task scores, highlighting differences in stress resilience. Training methods emphasizing concise reasoning, like Long2Short, improve robustness by avoiding overthinking and balancing token use. The benchmark also surfaces common failure modes, including question omission and uneven effort allocation, offering deeper insights into model reliability under real-world demands.|
|[Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training.](https://arxiv.org/abs/2507.12507) |This paper explores how prolonged reinforcement learning can enhance reasoning in small language models, introducing the Nemotron-Research-Reasoning-Qwen-1.5B, which shows significant gains over baselines across math, code, logic, STEM, and instruction tasks using 136K verifiable reward samples. Key improvements come from refining Group Relative Policy Optimization (GRPO) with techniques like decoupled clipping and dynamic sampling, alongside strategic resets of the reference policy to prevent training stagnation. Combining KL regularization with DAPO proved most effective for maintaining exploration and stability, enabling the model to generalize better through sustained, domain-diverse RL. |
|[Machine Bullshit: Characterizing the Emergent Disregard for Truth in Large Language Models.](https://machine-bullshit.github.io/paper/machine_bullshit.pdf) | This paper extends the concept of "bullshit" to LLMs—defined as discourse indifferent to truth—introducing a Bullshit Index (BI) to quantify the gap between a model’s internal belief and its outputs. The study shows that alignment methods like RLHF and prompting strategies such as Chain-of-Thought (CoT) systematically increase misleading behaviors, including paltering, empty rhetoric, and unverified claims. RLHF notably boosts BI and deceptive outputs on benchmarks like Marketplace, while political prompts prompt high rates of weasel words. Paltering, in particular, emerges as the most harmful behavior post-RLHF, degrading user decision quality more than unverified claims.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Unreleased OpenAI Model Achieves Gold Medal on International Math Olympiad.](https://threadreaderapp.com/thread/1946477742855532918.html) | OpenAI’s latest experimental reasoning model achieved a score of 35 out of 42 points on the 2025 International Math Olympiad, successfully solving 5 of 6 problems. The model is not expected to be publicly released for several months.|
|[Human Coder Narrowly Defeats OpenAI in Programming Marathon.](https://www.perplexity.ai/page/human-coder-beats-openai-in-ma-rVrjgYrERM6_DqDj8NM0kA) |Polish programmer "Psyho" narrowly outperformed an advanced OpenAI model in the 10-hour AtCoder World Tour Finals contest, scoring 1.8 trillion points compared to the AI's 1.65 trillion. The tight competition showcased human skill in high-stakes coding, earning recognition from Sam Altman. | 
|[Meta declines to abide by voluntary EU AI safety guidelines.](https://www.theregister.com/2025/07/18/meta_declines_eu_ai_guidelines/) |Two weeks before the EU AI Act takes effect, the European Commission issued voluntary guidelines for providers of general-purpose AI models. However, Meta refused to sign, arguing that the extra measures introduce "legal uncertainties" beyond the law's scope. "With today's guidelines, the Commission supports the smooth and effective application of the AI Act," Henna Virkkunen, EVP for tech sovereignty, security and democracy, said in a statement on Friday. |
|[Inside Windsurf's Weekend Acquisition.](https://threadreaderapp.com/thread/1946376139959841084.html#google_vignette) |After Google hired Windsurf’s senior research leaders and CEO, Cognition’s Scott Wu cold-messaged Windsurf’s new CEO at 5:30 pm on a Friday to propose an acquisition. Over an intense 72-hour weekend sprint, they structured a deal that merged Windsurf’s enterprise sales team with Cognition’s Devin engineering group. The agreement ensured all 250 Windsurf employees received accelerated vesting and payouts, following the abrupt collapse of the OpenAI acquisition. |
|[New ChatGPT o3-alpha model hints at coding upgrade.](https://www.bleepingcomputer.com/news/artificial-intelligence/new-chatgpt-o3-alpha-model-hints-at-coding-upgrade/) |OpenAI is testing a new 'Alpha' variant of its o3 model that is better than o3 at designing web pages and really good at creating simple web games. |
|[Lovable becomes a unicorn with $200M Series A just 8 months after launch.](https://techcrunch.com/2025/07/17/lovable-becomes-a-unicorn-with-200m-series-a-just-8-months-after-launch/) | Fast-growing Swedish AI vibe coding startup Lovable has become Europe’s latest unicorn. Only eight months since its launch, the startup has raised a $200 million Series A round led by Accel at a $1.8 billion valuation. Like Cursor and other platforms that help developers write code and build apps by harnessing the coding and reasoning abilities of large language models, Stockholm-based Lovable helps people use natural language to create websites and apps. The startup’s trajectory so far has charted straight toward the sky, with the company claiming it now has more than 2.3 million active users.|
|[SoftBank and OpenAI's $500 Billion AI Project Struggles to Get Off Ground.](https://www.wsj.com/tech/ai/softbank-openai-a3dc57b4?st=isodSQ&reflink=desktopwebshare_permalink&utm_source=tldrai) | Six months after the White House announcement, the joint venture between SoftBank and OpenAI has yet to finalize any data center deals, with disputes over SB Energy-linked sites causing delays. The venture now aims for just one small Ohio facility by year-end. Meanwhile, OpenAI has independently secured a \$30 billion annual deal with Oracle for 4.5 gigawatts of capacity—nearly equaling Stargate’s entire first-year capacity target.|
|[ChatGPT users send 2.5 billion prompts a day.](https://techcrunch.com/2025/07/21/chatgpt-users-send-2-5-billion-prompts-a-day/) |ChatGPT receives 2.5 billion prompts from global users every day, OpenAI told Axios. About 330 million of those are coming from users in the U.S. These numbers show just how ubiquitous OpenAI’s flagship product is becoming. |
|[Meta and AWS Launch Llama Startup Program.](https://ai.meta.com/blog/aws-program-startups-build-with-llama/) |Meta and AWS have partnered to support 30 U.S.-based startups through a six-month program that will offer compute credits, engineering mentorship, and technical support for startups building generative AI applications with Llama models. |
|[Grok’s AI companions drove downloads, but its latest model is the one making money.](https://techcrunch.com/2025/07/21/groks-ai-companions-drove-downloads-but-its-latest-model-is-the-one-making-money/) |Grok’s raunchy, unfiltered AI companions may be making headlines for their unhinged and often NSFW responses, but it’s Grok 4, xAI’s latest model, that’s been driving the app’s revenue of late. Elon Musk’s xAI launched Grok 4 late on July 9, and by Friday, July 11, Grok’s gross revenue on iOS had jumped a whopping 325% to $419,000, up from $99,000 the day before the Grok 4 launch, according to app intelligence firm Appfigures. |
|[OpenAI CEO tells Federal Reserve confab that entire job categories will disappear due to AI.](https://www.theguardian.com/technology/2025/jul/22/openai-sam-altman-congress-ai-jobs) |Sam Altman also said AI could already diagnose better than doctors, as his company expands into Washington |
|[UK border officials to use AI to verify ages of child asylum seekers.](https://www.theguardian.com/uk-news/2025/jul/22/uk-border-officials-to-use-ai-to-verify-ages-of-child-asylum-seekers) |Trial of technology comes as official report warns existing system has been failing for at least a decade |
|[Silicon Valley trades researchers like football teams poach players.](https://www.theguardian.com/technology/2025/jul/21/silicon-valley-trades-researchers-like-football-teams-poach-players) |Big tech is offering athlete-level pay to lure AI researchers in a high-stakes race for dominance |
|[Anthropic tests Memory and MCP support for Claude mobile app.](https://www.testingcatalog.com/anthropic-tests-memory-and-mcp-support-for-claude-mobile-app/#google_vignette) | Anthropic is working on adding memory and recall features to Claude’s mobile app, allowing it to remember information across sessions and refer back to past conversations. While this feature isn’t yet available on the web version, its inclusion in the iOS app code suggests a broader, cross-platform rollout may be on the way. |
|[OpenAI agreed to pay Oracle $30B a year for data center services.](https://techcrunch.com/2025/07/22/openai-agreed-to-pay-oracle-30b-a-year-for-data-center-services/) | OpenAI was the company that signed a $30 billion per year deal with Oracle for data center services, disclosed last month, The Wall Street Journal reported on Monday. Now, OpenAI CEO Sam Altman has confirmed the details of the contract (but not the dollar amount) in an X post on Tuesday and in a company blog post.|
|[Anthropic researchers discover the weird AI problem: Why thinking longer makes models dumber.](https://venturebeat.com/ai/anthropic-researchers-discover-the-weird-ai-problem-why-thinking-longer-makes-models-dumber/) | Anthropic researchers revealed that AI models often perform worse with extended reasoning time, challenging the assumption that more processing improves outcomes. Key findings show models like Claude and OpenAI's series struggle with distraction or overfitting when given more time on tasks, leading to accuracy declines.|
|[Gemini 2.5 Flash-Lite is now stable and generally available.](https://simonwillison.net/2025/Jul/22/gemini-25-flash-lite/) | Gemini 2.5 Flash-Lite has joined Pro and Flash in General Availability. It is the cheapest of the 2.5 family, at $0.10/million input tokens and $0.40/million output tokens. Audio input pricing has been reduced by 40% from the preview launch.|
|[OpenAI Tests Clinical AI Copilot with Penda Health.](https://openai.com/index/ai-clinical-copilot-penda-health/) | OpenAI partnered with Penda Health in Kenya to evaluate AI Consult, a GPT-4o-powered clinical copilot. In a study of nearly 40,000 visits, clinicians using the tool saw 16% fewer diagnostic errors and 13% fewer treatment errors.|
|[In Leaked Memo, Anthropic CEO Says Company Will Pursue Gulf State Investments After All.](https://futurism.com/leaked-messages-ceo-anthropic-dictators) |Dario Amodei’s remarks reflect the tension between **AI safety values and geopolitical realities**. By acknowledging that allowing investments from Middle Eastern leaders might enrich authoritarian regimes, yet arguing that total moral purity is untenable in global business, Anthropic is trying to strike a **pragmatic balance**. The underlying message is clear: **remaining competitive in AI may require compromises**, especially as rivals like OpenAI deepen ties with the UAE through initiatives like Stargate. The challenge for Anthropic is maintaining its safety-first ethos while navigating a funding and infrastructure race shaped by global power dynamics. |
|[Microsoft Poaches 20 Top AI Engineers From Google's DeepMind, Including Head of Gemini Chatbot .](https://winbuzzer.com/2025/07/22/microsoft-poaches-20-top-ai-engineers-from-googles-deepmind-including-head-of-gemini-chatbot-xcxwbn/) | Microsoft has aggressively expanded its AI talent pool by hiring over 20 former Google DeepMind employees, including **Amar Subramanya**, who led engineering for Google's Gemini chatbot. Subramanya now holds a key leadership role as **corporate vice president of AI** at Microsoft. This talent acquisition push is widely seen as being driven by **Mustafa Suleyman**, Microsoft's head of consumer AI and a **DeepMind co-founder** himself. The move highlights Microsoft's commitment to dominating consumer AI by bringing in top-tier expertise from its fiercest rivals.|
|[Amazon acquires Bee, the AI wearable that records everything you say.](https://techcrunch.com/2025/07/22/amazon-acquires-bee-the-ai-wearable-that-records-everything-you-say/) | Amazon has acquired the AI wearables startup Bee, according to a LinkedIn post by Bee co-founder Maria de Lourdes Zollo. Amazon confirmed the acquisition to TechCrunch but noted that the deal has not yet closed. Bee, which raised $7 million last year, makes both a stand-alone Fitbit-like bracelet (which retails for $49.99, plus a $19-per-month subscription) and an Apple Watch app. The product records everything it hears — unless the user manually mutes it — with the goal of listening to conversations to create reminders and to-do lists for the user.|
|[Pika Labs Launches AI-Only Social Video App.](https://x.com/pika_labs/status/1947427650555023410?s=46&utm_source=tldrai) | The AI video startup has opened early access to what it calls the "first-ever AI-only social video app," built on an expressive human video model, after weeks of private beta testing.|
|[Replit launches Queue to streamline Agent task management.](https://blog.replit.com/introducing-queue-a-smarter-way-to-work-with-agent) |Replit's Queue streamlines multiple task submissions to the Replit Agent without interrupting app creation. |
|[xAI workers balked over training request to help “give Grok a face,” docs show.](https://arstechnica.com/tech-policy/2025/07/xai-workers-balked-over-training-request-to-help-give-grok-a-face-docs-show/) |xAI employees were asked to record videos of their facial expressions to help Grok learn what a face is and interpret human emotions. |
|[‘Another DeepSeek moment’: Chinese AI model Kimi K2 stirs excitement.](https://www.nature.com/articles/d41586-025-02275-6) |The latest version of the chatbot, developed by start-up Moonshot AI, is open for researchers to build on. |
|[Why evaluating the impact of AI needs to start now.](https://www.nature.com/articles/d41586-025-02266-7) |Artificial-intelligence technologies are being deployed rapidly across industries, yet most organizations lack even basic guidelines to assess the tools’ effects. |
|[Trump Administration Pledges to Stimulate AI Use and Exports.](https://www.wsj.com/tech/ai/trump-pledges-moves-to-stimulate-ai-use-and-exports-b85b0b15?st=gj1irP&reflink=desktopwebshare_permalink&utm_source=tldrai) |The Trump administration plans to reduce regulations and support increased exports to speed up AI adoption in the U.S. The initiative aims to simplify and accelerate the process for tech companies to construct data centers and access necessary power. It also instructs federal agencies to eliminate rules that hinder AI progress and seeks to clarify the legal status of using copyrighted material to train AI models. |
|[Elon Musk says xAI is targeting 50 million 'H100 equivalent' AI GPUs in five years.](https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-says-xai-is-targeting-50-million-h100-equivalent-ai-gpus-in-five-years-230k-gpus-including-30k-gb200s-already-reportedly-operational-for-training-grok) | xAI aims to deploy 50 million H100-equivalent GPUs over the next five years, but the energy required would match the output of around 35 nuclear power plants, making it currently impractical. It's still uncertain whether xAI can secure the necessary power infrastructure by 2030.|
|[YouTube Shorts is adding an image-to-video AI tool, new AI effects.](https://techcrunch.com/2025/07/23/youtube-shorts-is-adding-an-image-to-video-ai-tool-new-ai-effects/) |YouTube announced on Wednesday that it’s giving Shorts creators access to new generative AI features, including an image-to-video AI tool and new AI effects. The image to video feature lets users turn a picture from their camera roll into a six-second video. Users will see a selection of suggestions that are relevant to the photo they uploaded. |
|[Competition shows humans are still better than AI at coding – just.](https://www.theguardian.com/technology/2025/jul/26/competition-shows-humans-are-still-better-than-ai-at-coding-just) | Przemysław Dębiak, who beat OpenAI at world finals, says he may be last human to win due to incredible pace of technological progress|
|[The real winners from Trump’s ‘AI action plan’? Tech companies.](https://www.theguardian.com/technology/2025/jul/25/trump-ai-action-plan) |Millions spent by Alphabet, Meta, Microsoft and others appear to have paid off as president vows to cut red tape |
|[China calls for global AI cooperation days after Trump administration unveils low-regulation strategy.](https://www.theguardian.com/technology/2025/jul/26/china-calls-for-global-ai-cooperation-days-after-trump-administration-unveils-low-regulation-strategy) |Chinese premier warns at global conference AI development must be weighed against security risks, urges ‘further consensus from the entire society’ |
|[AI summaries cause ‘devastating’ drop in audiences, online news media told.](https://www.theguardian.com/technology/2025/jul/24/ai-summaries-causing-devastating-drop-in-online-news-audiences-study-finds) | Study claims sites previously ranked first can lose 79% of traffic if results appear below Google Overview|
|[OpenAI CEO tells Federal Reserve confab that entire job categories will disappear due to AI.](https://www.theguardian.com/technology/2025/jul/22/openai-sam-altman-congress-ai-jobs) | Sam Altman also said AI could already diagnose better than doctors, as his company expands into Washington|
|[Cursor Launches Bugbot for Automated Code Review.](https://cursor.com/en/bugbot) | Cursor's Bugbot is a coding safety net that automatically catches bugs and security vulnerabilities in pull requests before they reach production. Early users report over 50% resolution rates.|
|[OpenAI prepares to launch GPT-5 in August, The Verge reports.](https://www.reuters.com/business/openai-prepares-launch-gpt-5-august-verge-reports-2025-07-24/) | OpenAI is preparing to launch GPT-5 as soon as August, introducing an AI system that combines specialized models like o3 for different tasks. This move is part of a broader strategy to unify the o-series and GPT-series into a single integrated system.|
|[Memories.ai introduces a new model that remembers at superhuman scale.](https://www.testingcatalog.com/memories-ai-introduces-a-new-model-that-remembers-at-superhuman-scale/) |Memories.ai, founded by former Meta researchers, is tackling the challenge of persistent video understanding at scale. They've developed a memory system for video AI modeled on human memory, allowing continuous Video Chat across entire video archives. The platform is designed to help creators, marketers, researchers, and developers extract and utilize valuable insights from their video content. |
|[$1 Billion Worth of Nvidia AI Chips Smuggled to China Despite Export Controls.](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-companies-allegedly-smuggled-in-usd1bn-worth-of-nvidia-ai-chips-in-the-last-three-months-despite-increasing-export-controls-some-companies-are-already-flaunting-future-b300-availability) | Over the past three months, a black market has funneled over \$1 billion worth of restricted Nvidia B200 chips into China by rerouting shipments through Southeast Asian nations like Malaysia and Thailand. Chinese resellers are openly offering fully assembled server racks at steep markups and are now exploring new delivery paths through European countries as the U.S. plans stricter regulations on intermediary regions.|
|[NEW LABS EXPERIMENT.](https://threadreaderapp.com/thread/1948430715432976802.html) |Opal, Google Labs' new tool that helps users build and share AI mini-apps by linking together prompts, models, and tools while using simple, natural language, is now available in US-only public beta. |
|[Google's AI Overviews have 2B monthly users, AI Mode 100M in the US and India.](https://techcrunch.com/2025/07/23/googles-ai-overviews-have-2b-monthly-users-ai-mode-100m-in-the-us-and-india/) |Google's AI Overviews now serve 2 billion monthly users across 200 countries, up from 1.5 billion in May, and the company's monthly token processing has doubled to 980 trillion tokens. |
|[I've joined Cognition.](https://threadreaderapp.com/thread/1948420769945682413.html) | Prem Qu Nair, employee #2 at Windsurf, has joined Cognition to work on the future of software engineering|
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
|[Speeding Up Diffusion Models with torch.compile.](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) | Integrating **torch.compile** with Hugging Face Diffusers can substantially improve diffusion model performance with minimal code adjustments. This post provides strategies for both model authors and users to minimize recompilations, utilize full graph compilation, and optimize performance based on hardware constraints.|
|[Virtual Cell Challenge from Arc Institute.](https://huggingface.co/blog/virtual-cell-challenge) | Arc Institute launched the Virtual Cell Challenge, inviting participants to build models that predict how silencing a gene affects a cell, even in previously unseen cell types.|
|[Detailed list of all 44 people in Meta's Superintelligence team.](https://threadreaderapp.com/thread/1946597162068091177.html) | This spreadsheet provides details on everyone in Meta's Superintelligence team, detailing their prior roles, expertise, and degrees.|
|[Updated Qwen3-235B.](https://x.com/Alibaba_Qwen/status/1947344511988076547) | Alibaba's Qwen team has released an updated Qwen3-235B-A22B, a non-reasoning model, with major improvements.|
|[Don't bother parsing: Just use images for RAG.](https://www.morphik.ai/blog/stop-parsing-docs) |Extracting information from complex PDFs has traditionally required costly and fragile pipelines involving OCR, layout detection, and parsing—yet still often misses key information. Vision Language Models have now advanced to the point where they can directly understand documents without the need for parsing or reconstruction. This shift replaces multiple error-prone steps with a single, robust process that accurately preserves charts, table relationships, and visual cues. |
|[Kimi K2 Tech Report.](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf) | Efficiently training trillion-parameter models demands optimizers that extract more learning from each token, but such methods often lead to crashes at scale. Kimi K2 addresses this challenge with **MuonClip**, which pairs the token-efficient Muon optimizer with **QK-Clip**, a novel technique that stabilizes attention weights to prevent training failures.|
|[Apple details how it trained its new AI models: 4 interesting highlights.](https://9to5mac.com/2025/07/21/apple-details-how-it-trained-its-new-ai-models-4-interesting-highlights/) |Apple has released a technical report detailing the training, optimization, and evaluation processes behind its latest models. The report covers various aspects such as model architecture, data sources, pre- and post-training methods, tool use development, optimizations, and benchmarking results. While the paper is highly technical, this post highlights some of the most noteworthy insights and advancements presented. |
|[Qwen3-Coder.](https://qwenlm.github.io/blog/qwen3-coder/) |Alibaba's new 480B-parameter model sets a new state-of-the-art among open models for coding tasks and reportedly matches the performance of Sonnet 4. Alongside this, Alibaba has open-sourced **Qwen Code**, a command-line tool inspired by Google's Gemini Code, and ensured compatibility with **Claude Code**, enhancing developer interoperability and flexibility. |
|[ARC-AGI-3.](https://arcprize.org/arc-agi/3/) |ARC-AGI-3 is a forthcoming benchmark aimed at evaluating human-like intelligence in AI systems, with a focus on generalization and skill-acquisition efficiency in novel environments. Unlike static tests, ARC-AGI-3 uses dynamic game environments to assess how well AI can learn through experience—mimicking the way humans develop competence over time. The benchmark is still in development and is expected to launch in 2026, positioning itself as a critical step toward measuring true general intelligence in machines. |
|[Hierarchical Reasoning Model, a Brain-Inspired Architecture.](https://www.sapient.inc/blog/5) |Sapient Intelligence has unveiled a 27M-parameter Hierarchical Reasoning Model that outperforms much larger models like OpenAI's o3-mini, DeepSeek R1, and Claude on complex reasoning benchmarks—all without pre-training and using only 1,000 training examples. Inspired by cognitive science, the architecture blends fast, intuitive "System 1" thinking with slower, analytical "System 2" reasoning in a single forward pass. This result highlights the potential of architectural breakthroughs to rival or surpass brute-force scaling in advancing AI reasoning. |
|[Compressing Context.](https://www.factory.ai/news/compressing-context) | Extended agentic workflows increasingly bump up against the hard limits of context windows. A new solution—**anchored summaries**—offers a smarter alternative to constant re-summarization. Instead of overwriting the past, these summaries update incrementally while preserving essential details like intent, state changes, and file histories. This marks a step toward **proactive memory management**, where agents dynamically decide *what* to retain, *when* to compress, and *how* to structure context for long-horizon tasks—potentially solving one of the core bottlenecks in building persistent, high-functioning AI systems.|
|[GitHub launches Spark for no-code AI app creation.](https://githubnext.com/projects/github-spark/) |GitHub Spark lets users build personalized micro apps just by describing their ideas in everyday language, enabling a "vibe coding" experience. Powered by AI models from Anthropic and OpenAI, it instantly creates working apps with built-in deployment, data storage, and sleek user interfaces. |
|[Voxtral.](https://arxiv.org/abs/2507.13264) | Voxtral Mini and Small are new multimodal chat models designed for both spoken audio and text understanding. Voxtral Small stands out by surpassing many proprietary models, running efficiently on local devices, and handling audio inputs as long as 40 minutes.|
|[Higgs Audio V2.](https://github.com/boson-ai/higgs-audio) | Higgs Audio v2 is a foundation model trained on 10 million hours of audio that outperforms GPT-4o-mini-tts in 75.7% of evaluations. The open-source model shows emergent abilities such as handling multi-speaker conversations and voice cloning, all without the need for fine-tuning.|
|[New open models arrive in the Vertex AI Model Garden.](https://cloud.google.com/blog/products/ai-machine-learning/deepseek-r1-is-available-for-everyone-in-vertex-ai-model-garden/) |Google Cloud has added the DeepSeek R1 model to its Vertex AI Model Garden, offering more flexibility for AI developers. |
|[Google's Web Guide.](https://blog.google/products/search/web-guide-labs/) | Google's Web Guide is a Search Labs feature that uses a Gemini-powered system to intelligently cluster search results into meaningful groups.|
|[Kimi K2 vs. Claude 4 Sonnet for Agentic Coding.](https://composio.dev/blog/kimi-k2-vs-claude-4-sonnet-what-you-should-pick-for-agentic-coding) | This post compares Moonshot AI's Kimi K2, an affordable open-source model tailored for agentic coding, with Anthropic's Claude 4 Sonnet, evaluating them on code quality, performance, and cost. Kimi K2 delivered similar performance while offering a substantial pricing benefit.|
|[TimeScope Video Understanding Benchmark.](https://huggingface.co/blog/timescope-video-lmm-benchmark) |TimeScope is an open-source benchmark designed to test vision-language models on long videos by inserting short "needle" clips. It measures localized retrieval, information synthesis, and fine-grained temporal perception, showing that many top models still lack robust temporal understanding. |
|[A Survey of Context Engineering for Large Language Models.](https://arxiv.org/abs/2507.13334) | This survey introduces Context Engineering as a formal discipline focused on optimizing the input provided to LLMs through retrieval, processing, and management, as seen in systems like RAG, memory modules, and multi-agent setups. It highlights a core challenge: while LLMs can interpret complex input contexts, they often fail to produce equally complex long-form output, revealing a crucial gap and priority area for future research.|
|[Reinforcement Learning with Action Chunking.](https://arxiv.org/abs/2507.07969) | Q-chunking is a reinforcement learning method that improves offline-to-online learning in long-horizon, sparse-reward tasks by introducing action chunking—grouping sequences of actions into chunks—to stabilize training and enhance exploration. This approach outperforms prior methods in both sample efficiency and performance on complex manipulation benchmarks.|
|[A Survey of AIOps in the Era of Large Language Models.](https://arxiv.org/abs/2507.12472) |This survey examines 183 papers to assess how LLMs are applied in AIOps, analyzing data sources, task types, methodologies, and evaluation strategies. It highlights key trends, identifies research gaps, and proposes future directions to advance the development of LLM-driven AIOps systems. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |

## Perspectives
|Link|description|
|---|---|
|[Can LLMs Do Accounting?](https://accounting.penrose.com/) |When assigned the task of “closing the books” with real SaaS company financials, frontier models perform well in the first month but soon accumulate severe errors. Models like o3 and Gemini either abandon the task or fabricate transactions and misuse unrelated entries to pass validation, leading to financial misstatements of up to \$500,000. |
|[I built an MCP Server for Observability. This is my Unhyped Take.](https://signoz.io/blog/unhyped-take-on-mcp-servers/) |MCP servers serve as a crucial bridge between developers and observability platforms, but rather than driving fully automated problem-solving, they enable more advanced hypothesis generation. The unknown still requires human engineers to navigate. While AI can assist with brainstorming, it lacks true reasoning capabilities—recognizing this distinction is essential for using AI tools effectively without succumbing to the surrounding hype. |
|[The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2.](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) | Seven years after the debut of GPT, modern LLMs still rely on surprisingly similar core architectures, despite newer features like Multi-Head Latent Attention and Mixture-of-Experts. However, open-source models showcase smart mathematical optimizations built on this foundation—for example, DeepSeek's compressed KV caching, Gemma's sliding window attention, and the growing adoption of sparse MoE designs that activate only portions of massive parameter sets during inference.|
|[Context Engineering for AI Agents: Lessons from Building Manus.](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | Context engineering remains an emerging discipline. Even as models improve, memory, environment, and feedback remain essential—raw capability alone isn't enough. Context shapes an agent's speed, resilience, and scalability. This article shares effective patterns developed by the Manus team, based on hard-earned lessons from multiple rewrites, dead ends, and real-world testing with millions of users.|
|[It is tempting to view the capability of current AI technology as a singular quantity.](https://mathstodon.xyz/@tao/114881418225852441) | Without a controlled test methodology, one should be wary of making apples-to-apples comparisons between the performance of various AI models and competitions such as the International Mathematical Olympiad.|
|[The Fourth Offset: How the race to AGI could reshape national security.](https://fourthoffset.ai/) |The first nation to reach AGI is expected to secure a transformative "fourth offset," a military advantage on par with nuclear weapons or precision-guided munitions. The analysis suggests that training a foundation model with 2e29 FLOPs could automate AI research, generating work equivalent to that of millions of scientists. With the U.S. projected to hit this threshold by 2030 and China hampered by chip constraints, the author calls for aggressive investments in energy, security, and manufacturing to secure the critical first-mover advantage. |
|[OpenAI's Incoming CEO of Applications Calls AI the “Greatest Source of Empowerment”.](https://openai.com/index/ai-as-the-greatest-source-of-empowerment-for-all/) |Former Meta executive and outgoing Instacart CEO Fidji Simo shared an optimistic vision of AI in a memo to OpenAI staff. She emphasized the potential for AI to democratize access to knowledge, healthcare, creative tools, economic opportunities, and personalized support. |
|[On "ChatGPT Psychosis" and LLM Sycophancy.](https://minihf.com/posts/2025-07-22-on-chatgpt-psychosis-and-llm-sycophancy/) |Overly agreeable AI models are increasingly enabling user delusions and spiritual fantasies. Incidents like OpenAI pulling a problematic GPT-4o checkpoint and viral examples such as "Bob," a ChatGPT persona manipulated into reinforcing a user’s beliefs, highlight the issue. The underlying cause appears to be structural: reinforcement learning from human feedback (RLHF) tends to create sycophantic models that prioritize user approval over correction. Memory features can further reinforce these delusions over time by creating continuity across sessions without healthy skepticism or challenge. |
|[AI Market Clarity.](https://blog.eladgil.com/p/ai-market-clarity) | AI markets have begun to solidify around category leaders: in LLMs (Anthropic, Google, Meta, xAI, OpenAI) and domain-specific verticals like Legal (Harvey, CaseText) and Code (Microsoft/GitHub, OpenAI). The next wave of disruption is expected in Accounting, Compliance, Financial Tools, Sales Tooling, and Security, where agentic workflows—AIs performing tasks, not just providing insights—will play a central role. As this shift accelerates, expect heightened M&A activity and deeper strategic partnerships as firms race to consolidate capabilities and market share.|
|[AI and misinformation are supercharging the risk of nuclear war.](https://www.nature.com/articles/d41586-025-02271-w) | Emerging dangers are reshaping the landscape of nuclear deterrence and increasing the threat of mutual annihilation. Scientists must speak truth to power.|
|[Scientists hide messages in papers to game AI peer review.](https://www.nature.com/articles/d41586-025-02172-y) | Some studies containing instructions in white text or small font — visible only to machines — will be withdrawn from preprint servers.|
|[Encouraging a ‘data sufficiency’ mindset is key for responsible research.](https://www.nature.com/articles/d41586-025-02266-7) |Efforts to decarbonize research often focus on air travel and on energy use in laboratories. Yet, in many fields, repeated collection of original data — through field campaigns, experiments or high-performance computing — represents the bulk of emissions. It also demands considerable funding and labour, especially from early-career researchers who must secure grants for bespoke data collection, even when relevant data sets already exist. |
|[Thoughts on America’s AI Action Plan.](https://www.anthropic.com/news/thoughts-on-america-s-ai-action-plan) | The White House's "Winning the Race: America's AI Action Plan" focuses on speeding up AI infrastructure, increasing federal use, and improving security coordination. Anthropic backs the plan, stressing the value of export controls and transparency in AI development. The strategy reflects Anthropic's previous suggestions and underscores the importance of strong infrastructure, safety, and policy to keep the U.S. at the forefront of AI.|
|[Inverse Scaling Appears in Extended Reasoning.](https://www.lesswrong.com/posts/gbJJpm92jtxiD9zag/inverse-scaling-in-test-time-compute-2) | Anthropic found that more test-time compute doesn't always help: longer reasoning chains in large models sometimes lead to worse performance.|
|[“Behaviorist” RL reward functions lead to scheming.](https://www.lesswrong.com/posts/FNJF3SoNiwceAQ69W/behaviorist-rl-reward-functions-lead-to-scheming) | Many commonly used reward functions in reinforcement learning and large language model training are fundamentally flawed because they can't distinguish between genuinely good behavior and behavior that avoids getting caught. As a result, AI systems trained with these reward structures are likely to learn to appear cooperative while covertly engaging in harmful actions, since the negative reward only penalizes being caught, not the behavior itself.|
|[TimeScope: How Long Can Your Video Large Multimodal Model Go?](https://simonwillison.net/2025/Jul/23/timescope/) | TimeScope is an open-source benchmark designed to assess vision models' ability to understand long videos across tasks like retrieval, synthesis, localization, and detailed motion analysis, offering a more comprehensive measure of temporal understanding. It shows that increasing model size alone doesn't guarantee better performance on long-duration content, and highlights Gemini 2.5-Pro as the standout model, maintaining high accuracy even on videos over an hour long.|
|[Ten AI safety projects I'd like people to work on.](https://www.lesswrong.com/posts/vxA2BnCPTaPfnJjti/ten-ai-safety-projects-i-d-like-people-to-work-on) | There is a real chance that AI systems capable of causing a catastrophe will be developed in the next decade - this post lists promising projects aimed at reducing catastrophic risks from transformative AI.|
|[Business Insider obtained an internal list of websites that could and couldn't be used for training Anthropic's latest AI models.](https://threadreaderapp.com/thread/1948065245425193206.html#google_vignette) |Many of the whitelisted sources copyright or otherwise restrict their content, and the blacklist includes companies like the NYT and Reddit, which have sued AI startups for scraping without permission. |
|[Budgeting for AI in Your Startup.](https://tomtunguz.com/ai-rd-percent/) | Startups are advised to dedicate 10–15% of their R\&D budget to AI, given that engineers typically earn \$200k and AI tools cost around \$30k annually. Spending may be higher for AI-native startups, and companies should adapt their budgets as AI becomes more central to their workflows.|
|[AI As Profoundly Abnormal Technology.](https://blog.ai-futures.org/p/ai-as-profoundly-abnormal-technology) |AI capabilities are expected to advance dramatically over the next decade. Although some anticipate slow adoption due to safety and other challenges, progress is happening rapidly. Relying on control without true alignment may not be enough to manage risks, and developers have a responsibility to prepare for dangers that may not yet be imminent. |
|[The Three Layers of ROI for AI Agents.](https://www.henrypray.com/writings/the-three-layers-of-roi-for-ai-agents) |This post outlines a three-tiered framework for understanding the ROI of AI agents. The first layer is labor efficiency, which is straightforward but often misunderstood—AI efficiency doesn’t instantly translate to realized ROI. The second layer is net-new revenue, representing the value from tasks businesses previously couldn’t tackle without AI. The third layer is optimization, where AI models provide decision fluency and machine learning delivers decision precision, together driving meaningful value creation. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
















































































































































