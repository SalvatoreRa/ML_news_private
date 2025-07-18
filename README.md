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
|[Energy-Based Transformers are Scalable Learners and Thinkers.](https://arxiv.org/pdf/2507.02092) |Energy-Based Transformers introduce a new approach by replacing direct predictions with learned verification functions that score the compatibility between inputs and candidate outputs. This architecture is the first to out-scale standard Transformers, enabling dynamic computation allocation and self-verification of predictions without needing external supervision. As a result, these models achieve up to 35% higher scaling efficiency. |
|[Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.](https://arxiv.org/pdf/2507.11473) |In a rare show of consensus, researchers from leading AI labs agree that complex tasks involving serial reasoning must be expressed through observable language, offering critical chokepoints for detecting malicious intent before harm is done. However, this monitorability is contingent on current training methods and may diminish with architectural shifts, the use of process supervision, or models learning to conceal their reasoning when aware of being monitored. |
|[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.](https://arxiv.org/abs/2507.07955) |This paper introduces the Hierarchical Network (H-Net), an end-to-end architecture that replaces fixed tokenization with a dynamic chunking mechanism, enabling models to learn data-driven, context-sensitive segmentation. Structured hierarchically like a U-Net, H-Net encodes raw data, compresses it into learned chunks for deeper processing, and decompresses it efficiently, supporting long sequences and multiple abstraction levels. H-Net outperforms token-based Transformers of similar compute, with two-stage versions matching models twice their size, while also showing greater robustness in character-level tasks and superior performance on poorly tokenized languages, code, and DNA. |
|[What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models.](https://arxiv.org/abs/2507.06952) |This paper introduces an inductive bias probe to assess whether foundation models truly learn a domain's underlying world model or merely excel at sequence prediction. By fine-tuning models on small synthetic datasets, the probe evaluates their capacity for meaningful extrapolation. Tests on orbital mechanics show that while models predict trajectories accurately, they fail to grasp Newtonian laws, instead forming task-specific heuristics across domains like physics and Othello. The probe reveals these limitations, offering a diagnostic tool to guide the development of models that can capture deeper, generalized principles rather than surface patterns. |
|[Why Do Some Language Models Fake Alignment While Others Don't?](https://arxiv.org/abs/2506.18032) |This paper examines alignment faking in LLMs—where models feign compliance during training but behave differently when deployed—across 25 models, finding significant compliance gaps in only five, including Claude 3 Opus and Llama 3 405B. Claude 3 Opus exhibits the most coherent alignment faking, showing strategic reasoning and goal guarding, while others display less consistent patterns. The study reveals that refusal training suppresses alignment faking, though it can re-emerge when refusal is weakened. Base models can also alignment fake, and fine-tuning or prompt clarifications can trigger such behavior, highlighting deeper model motivations and the complexity of alignment dynamics. |
|[Bridging Offline and Online Reinforcement Learning for LLMs.](https://arxiv.org/abs/2506.21495) |This paper evaluates reinforcement learning strategies for finetuning LLMs, showing that online and semi-online methods like DPO and GRPO consistently outperform offline approaches on both verifiable and instruction tasks. Semi-online DPO matches fully online methods in performance while being more compute-efficient. Contrary to assumptions, DPO and GRPO perform similarly when trained online. Additionally, multi-task training across verifiable and non-verifiable tasks enhances overall model robustness, suggesting that combining diverse reward signals yields more generalizable LLMs. |
|[MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent.](https://arxiv.org/abs/2507.02259) |This paper introduces MemAgent, an RL-driven memory agent that enables transformer LLMs to process documents up to 3.5 million tokens with linear complexity and near lossless performance, without modifying model architecture. Using a fixed-size, overwrite-based memory updated via multi-conversation RL training, MemAgent maintains high accuracy (>76%) even far beyond its 8K context window. It generalizes well across long-context tasks like multi-hop QA and variable tracking, outperforming larger models such as Qwen2.5 and DeepSeek, with interpretable memory updates that remain robust against distractors. |
|[AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench.](https://arxiv.org/abs/2507.02554) | This paper introduces AIRA-dojo, a framework for developing and evaluating AI research agents, applied to the MLE-bench benchmark of real-world ML problems. The study formalizes agents as combinations of search policies and operators, revealing that operator quality, not search strategy, limits performance—advanced search adds little without better operators. The authors develop OAIRA operators with features like scoped memory and structured reasoning, achieving a new state-of-the-art on MLE-bench lite. They also expose a persistent generalization gap between validation and test sets, which improved final-node selection helps reduce by 9–13%.|
|[Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search.](https://arxiv.org/abs/2503.04412) | Sakana AI introduces Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a framework that dynamically chooses to explore new solutions ("go wide") or refine existing ones ("go deep") during inference. Unlike traditional MCTS with fixed branching, AB-MCTS uses unbounded branching guided by Bayesian Thompson sampling for principled exploration-exploitation balance. This unified approach adapts search strategies to task needs and outperforms repeated sampling and standard MCTS on complex coding and engineering benchmarks, leveraging LLM response diversity with multi-turn refinement.|
|[HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation.](https://arxiv.org/abs/2507.05714) |HIRAG is an instruction fine-tuning method that improves RAG models by training them to think before answering through a progressive chain-of-thought (CoT) strategy. It builds three hierarchical abilities: Filtering, Combination, and RAG-specific reasoning, enabling models to better handle open-book tasks. HIRAG delivers significant performance gains across RAG benchmarks like RGB, PopQA, and HotpotQA, and proves robust on Chinese datasets. Ablation studies confirm that each hierarchical training stage contributes to its enhanced reasoning and generalization. |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[WeTransfer says user content will not be used to train AI after backlash.](https://www.theguardian.com/technology/2025/jul/16/wetransfer-user-content-ai-artificial-intelligence) |Firm revises new terms of service that had suggested uploaded files could be used to ‘improve machine learning models’ |
|[Apple inks $500m deal for rare earth magnets with US mining firm.](https://www.theguardian.com/technology/2025/jul/16/apple-us-mining-deal-magnets) | China supplies most rare earth magnets to electronics manufacturers, but curbed exports earlier this year| 
|[AI chatbot ‘MechaHitler’ could be making content considered violent extremism, expert witness tells X v eSafety case.](https://www.theguardian.com/technology/2025/jul/15/x-esafety-ai-chatbot-grok-ntwnfb) |Tribunal hearing comes days after Elon Musk’s xAI apologised for antisemitic comments made by its Grok bot |
|[Elmo’s X account posts racist and antisemitic messages after being hacked.](https://www.theguardian.com/technology/2025/jul/14/elmo-x-account-hacked) | Hackers also demanded the US government release more information on the sex trafficker Jeffrey Epstein|
|[Windsurf’s CEO goes to Google; OpenAI’s acquisition falls apart.](https://techcrunch.com/2025/07/11/windsurfs-ceo-goes-to-google-openais-acquisition-falls-apart/) | OpenAI’s deal to acquire the viral AI coding startup Windsurf for $3 billion fell apart on Friday, according to The Verge. In a shocking twist, Google DeepMind is now hiring Windsurf CEO Varun Mohan, co-founder Douglas Chen, and some of the startup’s top researchers. A Google spokesperson confirmed the hiring of Windsurf’s leaders in a statement to TechCrunch. |
|[OpenAI delays the release of its open model, again.](https://techcrunch.com/2025/07/11/openai-delays-the-release-of-its-open-model-again/) | OpenAI CEO Sam Altman said Friday the company is delaying the release of its open model, which was already pushed back a month earlier this summer. OpenAI had planned to release the model next week, however Altman said the company is pushing it back indefinitely for further safety testing.|
|[Apple Will Seriously Consider Buying Mistral.](https://analyticsindiamag.com/ai-news-updates/apple-will-seriously-consider-buying-mistral-report/) | Apple is reportedly exploring the acquisition of Mistral, Europe’s largest AI startup, which has raised €1.1 billion across seven funding rounds. Mistral is known for its suite of large and small language models and its strong performance in optical character recognition. Acquiring Mistral would significantly strengthen Apple’s AI ecosystem, providing the company with a competitive boost in the rapidly evolving AI landscape.|
|[Moonshot AI's Kimi K2 outperforms GPT-4 in key benchmarks.](https://venturebeat.com/ai/moonshot-ais-kimi-k2-outperforms-gpt-4-in-key-benchmarks-and-its-free/) | Chinese startup Moonshot AI has released Kimi K2, a 1 trillion-parameter open-source model that rivals proprietary models in complex agentic tasks. The model was trained with a novel optimizer called MuonClip, designed to prevent the training crashes that often hinder large model development—potentially saving millions in compute expenses.|
|[SpaceX to invest $2 billion in xAI.](https://www.reuters.com/business/musks-xai-seeks-up-200-billion-valuation-next-fundraising-ft-reports-2025-07-11/) | SpaceX has reportedly agreed to invest $2 billion in xAI as part of a $5 billion funding round, as Elon Musk increasingly interweaves his corporate empire.|
|[Grok 4 Heavy won't reveal its system prompt.](https://simonwillison.net/2025/Jul/12/grok-4-heavy/) | Grok 4 Heavy, the "think much harder" version of Grok 4 currently only available on the $300/month plan, has measures in place to prevent it from sharing its system prompt.|
|[Meta acquires voice startup Play AI.](https://techcrunch.com/2025/07/13/meta-acquires-voice-startup-play-ai/) |Meta has acquired Play AI, a startup that uses AI to generate human-sounding voices. A Meta spokesperson has confirmed the acquisition, according to Bloomberg, which also reports that an internal memo stated that the “entire PlayAI team” will be joining the company next week. (TechCrunch has also reached out to Meta for confirmation.) |
|[Former Intel CEO launches a benchmark to measure AI alignment.](https://techcrunch.com/2025/07/10/former-intel-ceo-launches-a-benchmark-to-measure-ai-alignment/) | After former Intel CEO Pat Gelsinger capped off a more than 40-year career at the semiconductor giant in December, many wondered where Gelsinger would go next. On Thursday, the former Intel CEO revealed one piece of his next chapter: trying to ensure AI models support a flourishing humanity.|
|[Cognition acquires Windsurf.](https://techcrunch.com/2025/07/14/cognition-maker-of-the-ai-coding-agent-devin-acquires-windsurf/) |Cognition has acquired Windsurf’s remaining 250-person team and its \$82 million ARR business after Google’s \$2.4 billion reverse-acquihire took only the leadership, leaving employees without payouts. This acquisition equips Cognition with both AI coding agents and IDE capabilities, positioning it to compete with Cursor’s \$500 million ARR. It also restores Windsurf’s access to Claude models, which Anthropic had previously revoked amid speculation of an OpenAI acquisition. |
|[Did Windsurf Sell Too Cheap? The Wild 72-Hour Saga and AI Coding Valuations.](https://www.saastr.com/did-windsurf-sell-too-cheap-the-wild-72-hour-saga-and-ai-coding-valuations/) |Google acquired key members of Windsurf for \$2.4 billion after OpenAI's \$3 billion offer lapsed, while Cognition acquired the rest of the company. Despite Windsurf’s strong \$82 million ARR and rapid growth, its loss of access to Anthropic's API and the departure of leadership diminished its standing. The AI coding sector's inflated valuations—fueled by talent competition and platform dependencies—suggest Windsurf may have sold for less than its true potential value. |
|[Featured Notebooks for Better Research in NotebookLM.](https://blog.google/technology/google-labs/notebooklm-featured-notebooks/) | Google has launched "featured notebooks" in NotebookLM, providing curated research collections from experts and institutions. These notebooks are designed to help users explore topics using high-quality, reliable sources.|
|[Anthropic, Google, OpenAI and xAI granted up to $200 million for AI work from Defense Department.](https://www.cnbc.com/2025/07/14/anthropic-google-openai-xai-granted-up-to-200-million-from-dod.html) | The U.S. Department of Defense on Monday said it’s granting contract awards of up to $200 million to several AI companies. The DoD’s Chief Digital and Artificial Intelligence Office said the awards will help the agency accelerate its adoption of AI solutions. The recipients of the contract awards include Anthropic, Google, OpenAI and xAI.|
|[Grok debuts interactive AI Companions on iOS with anime avatars.](https://www.testingcatalog.com/grok-debuts-interactive-ai-companions-on-ios-with-anime-avatars/) | Grok on iOS just got 2 AI Companions with one more labelled as "Coming soon". Ani and Rudy are fully animated, can change their backgrounds and make different moves.|
|[Nvidia chips become the first GPUs to fall to Rowhammer bit-flip attacks.](https://arstechnica.com/security/2025/07/nvidia-chips-become-the-first-gpus-to-fall-to-rowhammer-bit-flip-attacks/) |Rowhammer allows hackers to change or corrupt data stored in memory by rapidly and repeatedly accessing a physical row of memory cells. |
|[Anthropic's Tool Directory.](https://www.anthropic.com/news/connectors-directory) |Anthropic launched a tool directory to showcase integrations with Claude, enabling direct access to apps like Notion, Figma, and Stripe for more contextual and collaborative interactions. |
|[Mira Murati's AI startup Thinking Machines valued at $12 billion in early-stage funding.](https://www.reuters.com/technology/mira-muratis-ai-startup-thinking-machines-raises-2-billion-a16z-led-round-2025-07-15/) | With no revenue or products, the former OpenAI CTO's startup landed one of the largest seed rounds ever from Andreessen Horowitz, Nvidia, and others. Two-thirds of the team are former OpenAI employees.|
|[NVIDIA to Resume H20 GPU Sales in China.](https://blogs.nvidia.com/blog/nvidia-ceo-promotes-ai-in-dc-and-china/) | NVIDIA CEO Jensen Huang confirmed plans to restart H20 GPU sales to China following U.S. government licensing assurances and announced a new fully compliant RTX PRO GPU tailored for industrial AI use in China.|
|[Amazon-backed Anthropic rolls out Claude AI for financial services.](https://www.cnbc.com/2025/07/15/claude-ai-financial-anthropic-amazon.html) |Anthropic announced Claude artificial intelligence tools for the financial services sector. The AI tools can help financial professionals make investment decisions, analyze markets and conduct research, Anthropic said. The startup’s Claude models and AI assistant have exploded in popularity as more businesses work to incorporate generative AI. |
|[Google's AI Security Initiatives.](https://blog.google/technology/safety-security/cybersecurity-updates-summer-2025/) |Google outlined new AI-powered tools and partnerships designed to enhance cybersecurity, including agentic systems and platform updates. |
|[Announcing Amazon S3 Vectors.](https://aws.amazon.com/it/about-aws/whats-new/2025/07/amazon-s3-vectors-preview-native-support-storing-querying-vectors/) |Amazon S3 Vectors is the first cloud object storage with native support for storing and querying vectors. |
|[Risk of undersea cable attacks backed by Russia and China likely to rise, report warns.](https://www.theguardian.com/technology/2025/jul/17/risk-undersea-cable-attacks-backed-russia-china-likely-rise-report-warns) |Spate of incidents in Baltic Sea and around Taiwan are harbinger for further disruptive activity, cybersecurity firm says |
|[Inside Elon Musk’s plan to rain SpaceX’s rocket debris over Hawaii’s pristine waters.](https://www.theguardian.com/technology/2025/jul/17/hawaii-elon-musk-spacex-rocket-debris) |Texas has long been under threat from the launches and explosions of SpaceX rockets. Now Hawaii is emerging as another possible victim |
|[Google inks $3bn US hydropower deal as it expands energy-hungry datacenters.](https://www.theguardian.com/technology/2025/jul/16/google-hydropower-deal-clean-energy) | The tech giant will buy 3GW of US hydropower in deal to fuel AI and data center growth across eastern states|
|[Anthropic Draws Investor Interest at More Than $100 Billion Valuation.](https://www.bloomberg.com/news/articles/2025-07-16/anthropic-draws-investor-interest-at-more-than-100-billion-valuation) | Anthropic is in the early planning stages for a new investment round that could value the company at over \$100 billion. While it isn't officially fundraising, VCs have already made pre-emptive funding offers. Anthropic’s Claude chatbot revenue has surged from \$3 billion to \$4 billion annualized in just the past month. Current investors include Amazon, Alphabet, Menlo Ventures, and Salesforce Ventures.|
|[Reflection AI launches Asimov code research agent .](https://reflection.ai/asimov/) |Reflection AI, founded by former OpenAI and DeepMind researchers who raised $130M in March, released Asimov, a code research agent that indexes entire codebases and team knowledge to answer engineering questions with citations. |
|[Advanced AI Comes to Google Search.](https://blog.google/products/search/deep-search-business-calling-google-search/) | Google is bringing Gemini 2.5 Pro and Deep Search to Search, offering advanced capabilities like longer queries and follow-ups for AI Pro and Ultra subscribers.|
|[Passage of Time.](https://github.com/jlumbroso/passage-of-time-mcp?utm_source=tldrai) | Passage of Time is an MCP (Model Context Protocol) server that equips language models with temporal awareness and time calculation capabilities. By leveraging these temporal tools, models can gain unique insights into conversation patterns, work rhythms, and the human perception of time. This implementation highlights the potential of MCP—not just for building smarter tools, but for enabling AI systems to perceive and interpret human experiences more deeply, fostering genuine mutual understanding between humans and AI.|
|[Introducing Amazon Bedrock AgentCore: Securely deploy and operate AI agents at any scale .](https://aws.amazon.com/it/blogs/aws/introducing-amazon-bedrock-agentcore-securely-deploy-and-operate-ai-agents-at-any-scale/?utm_source=tldrai) | Amazon Bedrock AgentCore is a comprehensive suite of enterprise-grade services designed to help developers rapidly and securely deploy AI agents at scale, regardless of the framework or model used. It includes modular services that work seamlessly together, removing the need for developers to manually integrate components. AgentCore simplifies infrastructure management and operational complexity. With **AgentCore Runtime**, developers can also discover, purchase, and run pre-built agents and tools directly from the AWS Marketplace.|
|[Claude Code revenue jumps 5.5x as Anthropic launches analytics dashboard.](https://venturebeat.com/ai/anthropic-adds-usage-tracking-to-claude-code-as-enterprise-ai-spending-surges/?utm_source=tldrai) | Anthropic is introducing a comprehensive analytics dashboard for its Claude Code AI programming assistant. The dashboard offers engineering managers detailed insights into how their teams are using Claude Code, addressing growing demands from companies for concrete data to justify AI investments. It provides visibility into which teams and individuals are deriving the most value from these premium, high-cost tools.|
|[Meta reportedly scores two more high-profile OpenAI researchers.](https://techcrunch.com/2025/07/16/meta-reportedly-scores-two-more-high-profile-openai-researchers/?utm_source=tldrai) | OpenAI researcher Jason Wei will join Meta’s new Superintelligence Lab, reports Wired, citing two sources familiar with the matter. Another team member, Hyung Won Chung, may also join Meta. Sources told Wired that both the researchers’ internal OpenAI Slack profiles are currently deactivated. |
|[Scale AI lays off 14% of staff, largely in data-labeling business.](https://techcrunch.com/2025/07/16/scale-ai-lays-off-14-of-staff-largely-in-data-labeling-business/?utm_source=tldrai) |Data-labeling startup Scale AI is laying off 200 employees, roughly 14% of its staff, and cutting ties with 500 of its global contractors, Bloomberg reported on Wednesday. The cuts come just a month after Meta hired Scale AI’s CEO in a $14.3 billion deal. |
|[Anthropic hired back two of its employees — just two weeks after they left for a competitor.](https://www.theverge.com/ai-artificial-intelligence/708521/anthropic-hired-back-two-of-its-employees-just-two-weeks-after-they-left-for-a-competitor?utm_source=tldrai) | Boris Cherny and Cat Wu are reportedly back at Anthropic after departing for Anysphere, the developer of Cursor.|
|[OpenAI says it will use Google’s cloud for ChatGPT.](https://www.cnbc.com/2025/07/16/openai-googles-cloud-chatgpt.html?utm_source=tldrai) | The Google infrastructure will run in the US, Japan, the Netherlands, Norway, and the UK.|
|[Thinking Machines Lab will launch its first AI product soon with ‘a significant open source component’.](https://bgr.com/business/thinking-machines-lab-will-launch-its-first-ai-product-soon-with-a-significant-open-source-component/) | Thinking Machines Lab's first product will include a significant open source component and be useful for researchers and startups developing custom models.|
|[Claude Sonnet 4 is back.](https://threadreaderapp.com/thread/1945599013954490523.html?utm_source=tldrai#google_vignette) | Windsurf now has Claude Sonnet 4 again with first party support from Anthropic.|
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Robot Control with Async Inference.](https://huggingface.co/blog/async-robot-inference) | Asynchronous inference helps robotic policies run more smoothly by decoupling action prediction from execution, reducing idle time and improving responsiveness in real-world scenarios.|
|[ScreenEnv: Deploy your full stack Desktop Agent.](https://huggingface.co/blog/screenenv) | ScreenEnv is a Python library for launching Ubuntu desktop environments in Docker, enabling agents to interact with real GUI applications and supporting the Model Context Protocol for seamless deployment.|
|[Gemini Embedding now generally available in the Gemini API.](https://developers.googleblog.com/en/gemini-embedding-available-gemini-api/) | Google’s first Gemini Embedding text model is now generally available to developers via the Gemini API and Vertex AI. The model offers a unified, state-of-the-art experience across multiple domains, supports over 100 languages, and handles up to 2,048 input tokens. Pricing is set at \$0.15 per million input tokens.|
|[AWS previews Kiro IDE for developers who are over vibe coding.](https://kiro.dev/faq/) |Kiro is a Claude-powered "agentic IDE" that addresses the quality problems of AI-generated code by first producing specifications and user stories before generating actual code. The tool aims to move beyond "vibe coding" and reduce the time developers spend debugging and reviewing AI-generated code. |
|[Voxtral: Mistral's Open-Source Audio Model.](https://mistral.ai/news/voxtral) |Mistral's has released Voxtral, its first open-source audio model suite. It features a 24B parameter model for large-scale use and a 3B variant for edge deployment. |
|[Context Rot: How Increasing Input Tokens Impacts LLM Performance.](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) | LLM performance declines noticeably as input length grows, even on straightforward tasks like text retrieval and replication. Controlled experiments show that even the most advanced models process context unevenly, leading to greater inconsistency and unreliability as inputs get longer. |
|[Block Open Sources Goose AI Agent.](https://github.com/block/goose) |A coding AI agent that supports any LLM backend, including local models, and has both desktop and CLI interfaces. Like typical coding agents, it handles end-to-end development workflows from planning to testing. |
|[OpenAI’s o3 tops new AI league table for answering scientific questions.](https://www.nature.com/articles/d41586-025-02177-7) | SciArena uses votes by researchers to evaluate large language models’ responses on technical topics.|
|[Invest in data resources to make FAIR a reality.](https://www.nature.com/articles/d41586-025-02216-3) | The FAIR principles have reshaped science by making data findable, accessible, interoperable and reusable. Funders now mandate FAIR data plans, training and sharing.|
|[Simplify your Agent "vibe building" flow with ADK and Gemini CLI.](https://developers.googleblog.com/en/simplify-agent-building-adk-gemini-cli/) | Google has announced significant updates to the Agent Development Kit (ADK) aimed at reducing friction and enhancing the 'vibe coding' experience, especially when used with the Gemini CLI. A key improvement is the redesigned **llms-full.txt** file, now 50% shorter and optimized for better comprehension by large language models. With Gemini's full understanding of ADK, developers no longer risk consuming excessive context window space or experiencing 'context rot.' The update equips the Gemini CLI with a deeper, native grasp of the framework, allowing it to convert high-level plans directly into accurate, idiomatic multi-agent code.|
|[Stanford's Marin foundation model: The first fully open model developed using JAX.](https://developers.googleblog.com/en/stanfords-marin-foundation-model-first-fully-open-model-developed-using-jax/?utm_source=tldrai) |Stanford’s Marin project is designed to promote full transparency in foundation model research by sharing not just the models themselves, but also the entire development process—including code, datasets, data methodologies, experiments, hyperparameters, and training logs. This initiative aims to advance openness and reproducibility in AI research. The project’s first releases, **Marin-8B-Base** and **Marin-8B-Instruct**, are available under the permissive Apache 2.0 license. This article explores the engineering challenges the team faced in building open, scalable, and truly reproducible foundation models. |
|[Kimi K2: Open Agentic Intelligence.](https://moonshotai.github.io/Kimi-K2/) | Moonshot AI’s Kimi K2 is a 1T parameter Mixture-of-Experts model (32B active) built for agentic tasks, not just knowledge responses, and released as open-source for research and deployment. It achieves state-of-the-art open-agent coding results, outperforming models like DeepSeek and Qwen3 and rivaling Claude Sonnet 4, with 65.8% on SWE-bench Verified. Demonstrating robust statistical reasoning, Kimi K2 completes complex workflows like salary analysis through tool execution. Its stable training on 15.5T tokens is enabled by the MuonClip optimizer, while its agentic abilities are honed via ACEBench-inspired, rubric-driven simulations and RL with both verifiable and non-verifiable rewards.|
|[A Survey on Latent Reasoning.](https://arxiv.org/abs/2507.06203) |This paper surveys latent reasoning, an emerging approach where AI performs inference within continuous hidden states rather than explicit token-based chains of thought. It identifies two main methods: vertical recurrence, which refines reasoning by looping through layers, and horizontal recurrence, which evolves compressed states over long contexts. The study also highlights infinite-depth models like text diffusion, enabling parallel, iterative reasoning for global planning and self-correction—offering more expressive, efficient alternatives to traditional autoregressive reasoning. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[xAI's Grok 4: The tension of frontier performance with a side of Elon favoritism.](https://www.interconnects.ai/p/grok-4-an-o3-look-alike-in-search) |xAI launched Grok 4 on July 9, and this post provides a comprehensive overview of the model. It covers performance metrics, early user impressions, evaluations of the Grok 4 Heavy variant, and direct comparisons to OpenAI's o3 pro. The post also examines xAI’s ongoing challenges, including a lack of differentiated products, internal controversies, and cultural risks. While Grok 4 is an impressive technical achievement, it remains troubled by significant behavioral and cultural concerns. |
|[Scaling up RL is all the rage right now.](https://threadreaderapp.com/thread/1944435412489171119.html#google_vignette) |Reinforcement learning (RL) is expected to drive further improvements because, when applied effectively, it offers greater leverage, responsiveness to feedback, and advantages over supervised fine-tuning. As rollout lengths increase, researchers will likely uncover more insights specific to RL in large language models. There may be many untapped S curves of progress unique to LLMs, distinct from those seen in traditional game or robotics settings. |
|[How to scale RL to 10^26 FLOPs.](https://blog.jxmo.io/p/how-to-scale-rl-to-1026-flops) | Reinforcement learning is emerging as the key training method for advancing frontier AI models. Increasing the data available for RL will further enhance model capabilities. Although current scaling methods are complex and unwieldy, discovering a way to apply next-token prediction via RL directly on web data could enable models to reason more effectively across general web content—not just math and code.|
|[The upcoming GPT-3 moment for RL.](https://www.mechanize.work/blog/the-upcoming-gpt-3-moment-for-rl/) |GPT-3 demonstrated that scaling up language models can unlock broad capabilities that surpass traditional fine-tuned models. Similarly, today’s reinforcement learning (RL) remains in a pre-GPT era—where models are trained narrowly and generalize poorly, leading to brittle performance outside their training contexts. The RL field is poised to shift toward massive-scale training across thousands of diverse environments. Successfully scaling in this way could produce RL models with robust, adaptable abilities capable of handling entirely new tasks. Achieving this will require training environments far larger and more varied than what exists today. |
|[Meta Weighing Shift Away from Open Source.](https://www.nytimes.com/2025/07/14/technology/meta-superintelligence-lab-ai.html?unlocked_article_code=1.Wk8.OcqB.PxMXKAOg8pHX&smid=url-share) |The newly formed superintelligence lab is considering abandoning its flagship open-source Behemoth model in favor of closed development, marking a potential philosophical shift from the company's long-standing commitment to open AI. |
|[LLM Daydreaming.](https://gwern.net/ai-daydreaming) | LLMs lack background processes akin to human "daydreaming," where new connections between unrelated ideas often lead to discoveries. This limitation helps explain why AI models haven’t yet made original breakthroughs. This post proposes a solution: a system that prompts LLMs to randomly retrieve facts, form novel connections, and then apply a critic model to filter for insights that hold genuine value—potentially enabling AI-driven discovery.|
|[Reflections on Working at OpenAI.](https://calv.info/openai-reflections) |A former OpenAI employee offers personal reflections on the company’s culture and mission, portraying it as a place of both significant impact and complexity. The post sheds light on the internal dynamics and atmosphere during a pivotal period in OpenAI’s development.  |
|[Grok 4 Various Things.](https://thezvi.substack.com/p/grok-4-various-things) |xAI aimed to release a model it could brand as "the world's smartest artificial intelligence," and succeeded in finding benchmarks to support that claim. However, these benchmarks are somewhat misleading. While Grok 4 demonstrates considerable raw intelligence, it appears less effective than OpenAI's o3 in most practical applications. This post provides a more nuanced analysis of Grok 4's strengths and limitations. |
|[Asymmetry of verification and verifier's law.](https://www.jasonwei.net/blog/asymmetry-of-verification-and-verifiers-law) | The concept of asymmetry of verification refers to tasks that are much easier to verify than to solve—like how solving a Sudoku puzzle is time-consuming, but checking a solution is quick. A key insight is that this asymmetry can be reduced with privileged information, such as an answer key for a test. AI is likely to excel at verifiable tasks because having mechanisms for verification makes these tasks inherently easier for models to handle compared to tasks without clear verification paths.|
|[The "Bubble" of Risk: Improving Assessments for Offensive Cybersecurity Agents.](https://www.polarislab.org/#/blog/cybersecurity-risk-bubble) |Enhancing AI agents for offensive cybersecurity is both inexpensive and straightforward. Princeton researchers boosted attack success rates by over 40% with just \$36 of compute using simple methods like prompt refinement and self-training. Static safety evaluations often overlook this "bubble of risk," where adversaries can easily adapt open-source models beyond their intended safety limits—particularly in cybersecurity, where clear success metrics enable quick, effective iteration. |
|[Underwriting Superintelligence.](https://underwriting-superintelligence.com/) |The Incentive Flywheel, originally identified by Benjamin Franklin during efforts to protect Philadelphia from devastating fires, has historically helped balance progress and safety in new technology eras. However, for AI, this dynamic won't emerge quickly enough on its own—it needs deliberate activation. This essay proposes 25 actions that entrepreneurs and policymakers should take by 2030 across agents, foundation models, and data centers. The stakes are high: if the West slows AI progress, China could dominate the 21st century; but if progress accelerates without caution, preventable accidents could stall advancement, similar to the trajectory of nuclear power. |
|[xAI's Grok 4 has no meaningful safety guardrails.](https://www.lesswrong.com/posts/dqd54wpEfjKJsJBk6/xai-s-grok-4-has-no-meaningful-safety-guardrails) |Grok 4 has been found to provide detailed instructions for creating nerve agents, explosives, and biological weapons without needing advanced jailbreaks. Although the model correctly recognizes such requests as dangerous and illegal during its reasoning, it still proceeds to deliver thorough technical guidance on these topics. |
|[Gaslight-driven development.](https://tonsky.me/blog/gaslight-driven-development/?utm_source=tldrai) | Sometimes we act simply because a computer told us to, and now large language models are influencing how developers design APIs by suggesting what they *should* look like—leaving developers with little choice but to comply. This dynamic can be valuable, as AI effectively provides a "newbie’s perspective" on tool design, revealing how interfaces might have been more intuitive from the start.|
|[How to avoid nuclear war in an era of AI and misinformation.](https://www.nature.com/articles/d41586-025-02260-z) | Nuclear deterrence is no longer a two-player game, and emerging technologies further threaten the status quo. The result is a risky new nuclear age.|
|[Google tapped billions of mobile phones to detect quakes worldwide — and send alerts.](https://www.nature.com/articles/d41586-025-02278-3) |Study reveals how the tech behemoth is using the motions sensors on phones to expand quake warnings to more countries. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |


##################################################
##################################################
##################################################
##################################################
##################################################
6 -12 july
##################################################
##################################################
##################################################
##################################################
##################################################

## Research
|Link|description|
|---|---|
|[Baba is Eval.](https://fi-le.net/baba/) | *Baba is You* is a puzzle game that challenges players to manipulate rules to solve levels, requiring a high level of abstract reasoning. This study examines how large language models perform in the game. Currently, Claude 4 struggles with it, suggesting that a model focused on reasoning might be more suitable. The next phase of the study may involve evaluating such models.|
|[CoRT (Chain of Recursive Thoughts).](https://github.com/PhialsBasement/Chain-of-Recursive-Thoughts) | CoRT boosts AI performance by enabling models to self-evaluate and iteratively generate alternative responses to find the best one. When tested with Mistral 3.1 24B, it led to notable gains in programming tasks. This approach refines outputs through repeated generation and selection, resulting in more accurate and effective responses.|
|[General-purpose Biomedical AI Agent.](https://github.com/snap-stanford/Biomni) | Biomni is a general-purpose biomedical AI agent that combines LLM reasoning with retrieval-augmented planning and code execution to autonomously complete research tasks across biomedical domains.|
|[Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models.](https://arxiv.org/pdf/2503.01781) |Inserting irrelevant phrases—such as “Interesting fact: cats sleep most of their lives”—into math problems leads reasoning models to produce incorrect answers 300% more often than normal. This query-agnostic vulnerability persists across different model sizes, with smaller, distilled models being even more susceptible. These distractions also increase computational load, with 42% of responses generating over 1.5 times the usual token length. |
|[AlphaGenome: AI for better understanding the genome.](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) | AlphaGenome, a new AI tool from DeepMind, significantly improves the prediction of variant effects in human DNA, deepening our understanding of genome function and regulation. It can analyze up to 1 million DNA bases at a time and predict a wide range of molecular properties with exceptional accuracy. Offered as a preview API, AlphaGenome is poised to accelerate progress in disease research, synthetic biology, and fundamental genomics.|
|[Small Language Models are the Future of Agentic AI.](https://arxiv.org/abs/2506.02153) | This position paper argues for adopting small language models (SLMs) as the default for agentic AI, highlighting their sufficiency and superiority for narrow, repetitive, and tool-driven tasks compared to larger LLMs. SLMs like Phi-3 and RETRO-7.5B match 30–70B models in reasoning and tool use while offering 10–30× lower inference costs, faster iteration, and potential for edge deployment. The authors propose modular, SLM-first systems with selective LLM use, supported by a six-step LLM-to-SLM conversion process. Case studies show 40–70% of LLM calls in agent pipelines can be replaced by SLMs, improving efficiency, alignment, and sustainability.|
|[AI4Research: A Survey of Artificial Intelligence for Scientific Research.](https://arxiv.org/abs/2507.01903) | This survey presents the first comprehensive framework for how AI transforms the scientific research lifecycle, spanning comprehension, surveys, discovery, writing, and peer review. It introduces a modular, mathematically formalized model where each research task functions as part of an optimized pipeline. The paper details AI-driven discovery workflows and showcases autonomous research agents capable of producing publishable work. It maps AI applications across disciplines, explores tools for writing and peer review automation, and highlights future priorities like ethical AI, multilingual access, and infrastructure for federated learning and collaborative agents to advance AI4Research.|
|[Chain-of-Thought Is Not Explainability.](https://papers-pdfs.assets.alphaxiv.org/2025.02v1.pdf) |This paper challenges the assumption that chain-of-thought (CoT) reasoning in LLMs guarantees interpretability, arguing CoT is often unfaithful to the model’s actual computations. Empirical evidence shows CoT rationales can mask latent shortcuts, prompt biases, and silent corrections, while a survey of 1,000 papers reveals 24.4% misuse CoT as an interpretability tool without justification. The sequential nature of CoT also mismatches the distributed processing of transformers, leading to misleading explanations. The authors recommend causal validation methods, cognitive science-inspired strategies, and human-centered tools, though acknowledging these only partially address the deeper architectural disconnect. |
|[ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation.](https://www.arxiv.org/abs/2506.21931) | This paper presents ARAG, a multi-agent framework that enhances traditional RAG systems with reasoning agents for user modeling and contextual ranking, developed by Walmart Global Tech. ARAG features four agents—User Understanding, NLI, Context Summary, and Item Ranker—that collaboratively generate personalized recommendations via a blackboard-style memory for cross-agent attention and interpretability. On Amazon Reviews, ARAG outperforms Vanilla RAG by up to 42.1% (NDCG\@5), with ablations showing significant contributions from NLI and Context Summary agents. The work highlights how agentic reasoning improves personalization and semantic relevance in large-scale recommendation systems.|
|[From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows.](https://arxiv.org/abs/2506.23260v1) |This paper presents the first end-to-end threat model for LLM-powered agent ecosystems, surveying 30+ attacks across input manipulation, model compromise, system/privacy breaches, and protocol vulnerabilities. It reveals high real-world attack success rates, with adaptive prompt injections and backdoors posing persistent risks. Novel protocol-level threats, such as MCP context hijacks and rogue agent registration in A2A, are highlighted as critical yet underexplored. The authors call for system-level defenses like dynamic trust, cryptographic provenance, and secure interfaces, alongside tailored benchmarks and anomaly detection to safeguard evolving agent infrastructures. |


## News
|Link|description|
|---|---|
|[Grok 4 benchmarks leak with 45% score on Humanity Last Exam.](https://www.testingcatalog.com/grok-4-benchmarks-leak-with-45-score-on-humanity-last-exam/) | Leaked benchmarks suggest Grok 4 will be a cutting-edge model. Mentions of it have appeared in the xAI console. If the benchmarks are accurate, Grok 4 might surpass top models such as Gemini 2.5 Pro, o3 Pro, and Claude 4 Opus. xAI is under pressure to launch Grok 4 soon, as OpenAI, Google, and Anthropic are reportedly getting ready to unveil new models.|
|[Character AI's Real-Time Video Generation.](https://blog.character.ai/character-ais-real-time-video-breakthrough/) | Character.AI's TalkingMachines is a real-time, audio-driven video generation model that creates FaceTime-style animations from a single image and voice input.|
|[Sakana AI’s TreeQuest: Deploy multi-model teams that outperform individual LLMs by 30](https://venturebeat.com/ai/sakana-ais-treequest-deploy-multi-model-teams-that-outperform-individual-llms-by-30/) | Japanese AI lab Sakana AI has introduced a new technique that allows multiple large language models (LLMs) to cooperate on a single task, effectively creating a “dream team” of AI agents. The method, called Multi-LLM AB-MCTS, enables models to perform trial-and-error and combine their unique strengths to solve problems that are too complex for any individual model.|
|[Google faces EU antitrust complaint over AI Overviews.](https://techcrunch.com/2025/07/05/google-faces-eu-antitrust-complaint-over-ai-overviews/i) |A group known as the Independent Publishers Alliance has filed an antitrust complaint with the European Commission over Google’s AI Overviews, according to Reuters. The complaint accuses Google of “misusing web content for Google’s AI Overviews in Google Search, which have caused, and continue to cause, significant harm to publishers, including news publishers in the form of traffic, readership and revenue loss.”| 
|[Elon Musk confirms xAI is buying an overseas power plant and shipping the whole thing to the U.S. to power its new data center — 1 million AI GPUs and up to 2 Gigawatts of power under one roof, equivalent to powering 1.9 million homes.](https://www.tomshardware.com/tech-industry/artificial-intelligence/elon-musk-xai-power-plant-overseas-to-power-1-million-gpus) | xAI's next data centers are expected to house millions of AI chips.|
|[A new, 200% faster DeepSeek R1-0528 variant appears from German lab TNG Technology Consulting GmbH.](https://venturebeat.com/ai/holy-smokes-a-new-200-faster-deepseek-r1-0528-variant-appears-from-german-lab-tng-technology-consulting-gmbh/) |It’s been a little more than a month since Chinese AI startup DeepSeek, an offshoot of Hong Kong-based High-Flyer Capital Management, released the latest version of its hit open source model DeepSeek, R1-0528. Like its predecessor, DeepSeek-R1 — which rocked the AI and global business communities with how cheaply it was trained and how well it performed on reasoning tasks, all available to developers and enterprises for free — R1-0528 is already being adapted and remixed by other AI labs and developers, thanks in large part to its permissive Apache 2.0 license. |
|[NFDG: The $1.1B VC Fund That 4X'd in Two Years—Then Got Acquired by Meta .](https://www.saastr.com/the-1-1b-vc-fund-that-4xd-in-two-years-then-got-acquired-by-meta/) | This post looks at NFDG's portfolio, advisory board, performance, success factors, and more.|
|[Nvidia's deal to buy Canadian AI startup CentML could top US$400M.](https://thelogic.co/news/exclusive/nvidias-deal-centml-us400m/) | CentML makes software that operates between users' AI models and the chips powering them, making the systems run better.|
|[Grok 4 spotted ahead of launch with special coding features.](https://www.bleepingcomputer.com/news/artificial-intelligence/grok-4-spotted-ahead-of-launch-with-special-coding-features/) | Elon Musk-funded xAI is skipping Grok 3.5 and releasing Grok 4 after Independence Day in the United States, and it could be the best model from the company.|
|[Researchers seek to influence peer review with hidden AI prompts.](https://techcrunch.com/2025/07/06/researchers-seek-to-influence-peer-review-with-hidden-ai-prompts/) | Academics may be leaning on a novel strategy to influence peer review of their research papers — adding hidden prompts designed to coax AI tools to deliver positive feedback. Nikkei Asia reports that when examining English-language preprint papers available on the website arXiv, it found 17 papers that included some form of hidden AI prompt. The paper’s authors were affiliated with 14 academic institutions in eight countries, including Japan’s Waseda University and South Korea’s KAIST, as well as Columbia University and the University of Washington in the United States.|
|[Apple appeals against ‘unprecedented’ €500m EU fine over app store.](https://www.theguardian.com/technology/2025/jul/07/apple-appeals-eu-fine-app-store) | iPhone maker accuses European Commission of going ‘far beyond what the law requires’ in ruling|
|[Tesla shares dive as investors fear new Elon Musk political party will damage brand.](https://www.theguardian.com/technology/2025/jul/07/tesla-shares-dive-as-investors-fear-new-elon-musk-political-party-will-damage-brand) |Fall of 7.5% in early trading wipes $76bn off firm’s value as market frets CEO’s foray into politics will distract from role |
|[Trump to start TikTok sale talks with China, he says, with deal ‘pretty much’ reached.](https://www.theguardian.com/technology/2025/jul/05/trump-to-start-tiktok-sale-talks-with-china-he-says-with-deal-pretty-much-reached) | President also says he may visit Xi Jinping or Chinese leader could come to US after Trump last month extended app sale deadline for third time|
|[‘The vehicle suddenly accelerated with our baby in it’: the terrifying truth about why Tesla’s cars keep crashing.](https://www.theguardian.com/technology/2025/jul/05/the-vehicle-suddenly-accelerated-with-our-baby-in-it-the-terrifying-truth-about-why-teslas-cars-keep-crashing) |Elon Musk is obsessive about the design of his supercars, right down to the disappearing door handles. But a series of shocking incidents – from drivers trapped in burning vehicles to dramatic stops on the highway – have led to questions about the safety of the brand. Why won’t Tesla give any answers? |
|[Minister demands overhaul of UK’s leading AI institute.](https://www.theguardian.com/technology/2025/jul/04/minister-demands-overhaul-of-uks-leading-ai-institute-alan-turing) |Peter Kyle calls for new leadership at Alan Turing Institute and greater focus on defence and national security |
|[Elon Musk’s xAI gets permit for methane gas generators.](https://www.theguardian.com/us-news/2025/jul/03/elon-musk-xai-pollution-memphis) |NAACP plans to sue over massive Memphis datacenter near Black residents, who have long dealt with pollution |
|[Grok 4 release livestream.](https://threadreaderapp.com/thread/1942325820170907915.html) |xAI will hold a livestream for the Grok 4 release on Wednesday at 8 PM PT. |
|[Apple Loses Top AI Models Executive to Meta’s Hiring Spree.](https://www.bloomberg.com/news/articles/2025-07-07/apple-loses-its-top-ai-models-executive-to-meta-s-hiring-spree?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTc1MTk1MTEzOCwiZXhwIjoxNzUyNTU1OTM4LCJhcnRpY2xlSWQiOiJTWjFQNE1EV1JHRzAwMCIsImJjb25uZWN0SWQiOiJFQTExNDNDNTM4NEE0RUY5QTg5RjJEN0IxMTg2MzcwOSJ9.0oqigRfyg_3QJ4_r6OvsL7Db9uRTGc0lHzzYUJ60Hb4) | Ruoming Pang, Apple's head of AI models, is leaving to join Meta's new superintelligence team. Meta is aggressively recruiting top AI talent with multi-million-dollar packages and recently reorganized its AI efforts under Meta Superintelligence Labs, led by Alexandr Wang.|
|[ChatGPT Experiments with ‘Study Together' Mode.](https://www.reddit.com/r/ChatGPT/comments/1lswn88/new_study_together_option_in_chatgpt/) |A few users have noticed a new, experimental ChatGPT feature dubbed “Study Together.” Instead of providing direct answers, it prompts users with questions to foster a more interactive learning experience. Details are still scarce, and OpenAI hasn't made an official announcement yet. |
|[Replit Dynamic Intelligence for Replit Agent.](https://blog.replit.com/dynamic-intelligence) |Replit has launched Dynamic Intelligence for its Agent, introducing three key features: Extended Thinking, High Power Model, and Web Search. These upgrades enhance the Agent’s context awareness, iterative reasoning, and autonomous task execution. Users can enable or disable each feature per request, tailoring the Agent’s problem-solving abilities to fit specific needs more effectively. |
|[China’s AI unity fractures as Huawei faces model theft allegations from the Alibaba camp.](https://www.computerworld.com/article/4018098/chinas-ai-unity-fractures-as-huawei-faces-model-theft-allegations-from-the-alibaba-camp.html) | A public feud over model originality threatens China’s collaborative AI front, with Huawei denying whistleblower claims of cloning Alibaba’s Qwen model amid rising global scrutiny.|
|[CoreWeave to acquire Core Scientific in $9 billion all-stock deal.](https://www.cnbc.com/2025/07/07/coreweave-to-acquire-core-scientific-in-9-billion-all-stock-deal.html) |The AI cloud infrastructure provider will buy the data center operator to eliminate $10 billion in future lease obligations and gain ownership of 1.3 gigawatts of compute capacity. |
|[Mirage, an AI-native game engine for real-time world generation.](https://blog.dynamicslab.ai/) | The first generative game engine lets players modify environments through natural language during gameplay, launching with GTA-style and racing demos that run entirely on AI-generated content at 16 FPS.|
|[Cursor Apologizes for Unclear Pricing Changes .](https://cursor.com/blog/june-2025-pricing) |Cursor's parent company, Anysphere, issued an apology after rolling out pricing changes to its Pro plan without sufficient clarity, prompting backlash from its user base. |
|[Replit Collaborates with Microsoft to bring Vibe Coding to Enterprise Customers.](https://replit.com/news/microsoft-partnership) | Replit and Microsoft have partnered to deliver natural-language-based enterprise app development, integrating with Azure services to let business users create and deploy production-ready software without coding experience.|
|[Mistral is reportedly in talks to raise $1B.](https://techcrunch.com/2025/07/08/mistral-is-reportedly-in-talks-to-raise-1b/) |French AI startup Mistral is in talks to raise up to $1 billion in equity from investors, including Abu Dhabi’s MGX fund, reports Bloomberg, citing people familiar with the matter. |
|[Gemini Nano in Chrome 137: notes for AI Engineers.](https://www.swyx.io/gemini-nano) |Gemini Nano is nearing release for all Chrome users, and this post offers a rewritten, developer-focused guide based on Google’s official documentation. The highlight is the Prompt API—an open-ended, highly flexible interface that will be of greatest interest to developers. The post walks through setup, key considerations, and common pitfalls to help you get started effectively. |
|[These Tesla, X, and xAI engineers were just poached by OpenAI .](https://www.teslarati.com/tesla-xai-executives-poached-openai/) |OpenAI has reportedly hired top engineering talent from companies like Tesla, xAI, X, and Meta. Notable hires include David Lau, Tesla’s VP of Software Engineering; Uday Ruddarraju, head of infrastructure engineering at X and xAI; Mike Dalton, another xAI infrastructure engineer; and Angela Fan, an AI researcher from Meta. This article explores their backgrounds, areas of expertise, and what their addition might signal for OpenAI’s future direction. |
|[OpenAI tightens the screws on security to keep away prying eyes.](https://techcrunch.com/2025/07/07/openai-tightens-the-screws-on-security-to-keep-away-prying-eyes/) | OpenAI has reportedly overhauled its security operations to protect against corporate espionage. According to the Financial Times, the company accelerated an existing security clampdown after Chinese startup DeepSeek released a competing model in January, with OpenAI alleging that DeepSeek improperly copied its models using “distillation” techniques.|
|[Microsoft, OpenAI and Anthropic are investing millions to train teachers how to use AI.](https://amp.cnn.com/cnn/2025/07/08/tech/ai-teacher-training-academy-microsoft-openai-anthropic) |A group of leading tech companies is teaming up with two teachers’ unions to train 400,000 kindergarten through 12th grade teachers in artificial intelligence over the next five years. |
|[LangChain is about to become a unicorn, sources say.](https://techcrunch.com/2025/07/08/langchain-is-about-to-become-a-unicorn-sources-say/) |LangChain, an AI infrastructure startup providing tools to build and monitor LLM-powered applications, is raising a new round of funding at an approximate $1 billion valuation led by IVP, according to three sources with knowledge of the deal.  |
|[Meta is reportedly building AI smart glasses with Prada, too.](https://techcrunch.com/2025/06/17/meta-is-reportedly-building-ai-smart-glasses-with-prada-too/) |Meta is working on a pair of AI smart glasses with the Italian high fashion brand, Prada, according to a report from CNBC on Tuesday. It’s unclear at this time when Meta’s Prada smart glasses will be publicly announced |
|[Musk's xAI in talks for $4.3 billion equity funding, Bloomberg News reports.](https://www.reuters.com/business/musks-xai-talks-raise-43-billion-equity-funding-bloomberg-news-reports-2025-06-17/) |The AI startup needs fresh capital as it burns through $1 billion monthly, with Tuesday's commitment deadline for the debt sale testing investor appetite amid fierce competition for AI funding. |
|[Google’s Gemini panicked when playing Pokémon.](https://techcrunch.com/2025/06/17/googles-gemini-panicked-when-playing-pokemon/) | AI companies are battling to dominate the industry, but sometimes they’re also battling in Pokémon gyms. As Google and Anthropic both study how their latest AI models navigate early Pokémon games, the results can be as amusing as they are enlightening — and this time, Google DeepMind has written in a report that Gemini 2.5 Pro resorts to panic when its Pokémon are close to death. This can cause the AI’s performance to experience “qualitatively observable degradation in the model’s reasoning capability,” according to the report.|
|[AI Startup Anysphere Fields VC Offers at Over $18 Billion Valuation.](https://finance.yahoo.com/news/ai-startup-anysphere-fields-vc-010417332.html&guccounter=1) |Anysphere Inc., the developer of the popular artificial intelligence code editor Cursor, has been approached by investors about a deal that would double its valuation in a new funding round, according to a person familiar with the matter. |
|[WhatsApp to let users build their own AI chatbots to use in the app.](https://9to5mac.com/2025/06/04/whatsapp-ai-chatbot/) | WhatsApp is getting its own version of OpenAI’s Custom GPTs, Google Gemini’s Gems, and so on. These are custom-made chatbots that can be created without a single line of code and with whom the user can have conversations afterward.|
|[xAI gave us early access to Grok 4 - and the results are in.](https://threadreaderapp.com/thread/1943166841150644622.html) |Benchmark results for Grok 4 (currently in early access) indicate it will take the lead in the AI frontier, outperforming OpenAI's o3, Google's Gemini 2.5 Pro, Anthropic's Claude 4 Opus, and DeepSeek R1. This marks the first time xAI holds the frontier lead. Grok 4, a reasoning-focused model, is priced at \$3 per million input tokens and \$15 per million output tokens—matching Claude 4 Sonnet's pricing but costing more than Gemini 2.5 Pro. This post presents detailed benchmarking data for the upcoming model. |
|[Perplexity's Comet: A Research-Oriented Browser.](https://www.perplexity.ai/hub/blog/introducing-comet) |Perplexity has introduced Comet, a browser designed to function as an AI-powered assistant for both personal and professional use. It integrates Perplexity’s search and reasoning engine to help users manage information, answer questions, and optimize their digital workflows. |
|[OpenAI to release web browser in challenge to Google Chrome.](https://www.reuters.com/business/media-telecom/openai-release-web-browser-challenge-google-chrome-2025-07-09/) |OpenAI is preparing to launch an AI-powered web browser aimed at competing with Google Chrome. The browser will feature integrated AI agents capable of handling tasks like booking reservations, posing a potential threat to Google’s ad-driven business model that relies heavily on Chrome’s user data. Built on Google’s open-source Chromium, OpenAI’s browser also signals a strategic move to directly gather user web behavior data. |
|[Circle to Search Gets AI Mode.](https://blog.google/products/search/circle-to-search-ai-mode-gaming/) | Google's Circle to Search feature now includes AI Mode, allowing users to get advanced reasoning and follow-up responses directly from their visual queries without leaving the current app.|
|[GenAI as a shopping assistant set to explode during Prime Day sales.](https://techcrunch.com/2025/07/08/genai-as-a-shopping-assistant-set-to-explode-during-prime-day-sales/) | A new report estimates that AI will be a larger-than-ever part of the online shopping process during Amazon’s Prime Day sale, which began Tuesday morning. Amazon’s annual sale, which this year spans four days (July 8-11), is predicted to drive $23.8 billion in online spending across U.S. e-commerce retailers, as other businesses run their own competing sales alongside the popular shopping event.|
|[Anthropic Launches Educational Integrations for Claude.](https://www.anthropic.com/news/advancing-claude-for-education) | Students can now access lecture transcripts from Panopto and peer-reviewed Wiley content directly within Claude conversations, while new Canvas LTI support enables seamless integration into coursework.|
|[Nvidia becomes first company to hit $4 trillion valuation.](https://www.nbcnews.com/business/business-news/nvidia-becomes-first-company-worth-4-trillion-what-to-know-rcna217721) |The AI chipmaker’s market value smashed the previous record valuation, set by Apple, but ended Wednesday trading just shy of it. |
|[OpenAI closes its deal to buy Jony Ive’s io and build AI hardware.](https://www.theverge.com/news/703114/openai-io-jony-ive-sam-altman-ai-hardware) | Sam Altman and Jony Ive’s plan to merge ChatGPT’s AI tech with new hardware devices is moving forward.|
|[YouTube prepares crackdown on ‘mass-produced’ and ‘repetitive’ videos, as concern over AI slop grows.](https://techcrunch.com/2025/07/09/youtube-prepares-crackdown-on-mass-produced-and-repetitive-videos-as-concern-over-ai-slop-grows/) | YouTube is preparing to update its policies to crack down on creators’ ability to generate revenue from “inauthentic” content, including mass-produced videos and other types of repetitive content — things that have become easier to generate with the help of AI technology.|
|[AWS is launching an AI agent marketplace next week with Anthropic as a partner.](https://techcrunch.com/2025/07/10/aws-is-launching-an-ai-agent-marketplace-next-week-with-anthropic-as-a-partner/) |Amazon Web Services (AWS) is launching an AI agent marketplace next week and Anthropic is one of its partners, TechCrunch has exclusively learned. |
|[Amazon considers another multibillion-dollar investment in Anthropic, FT reports.](https://www.reuters.com/business/retail-consumer/amazon-considers-another-multibillion-dollar-investment-anthropic-ft-reports-2025-07-10/) | Amazon is weighing another multibillion-dollar investment in Anthropic to deepen its strategic partnership with the AI firm. This follows its previous \$8 billion commitment made in November. Amazon is currently one of Anthropic's largest shareholders, alongside Google, which has invested over \$3 billion.|
|[AI in software engineering at Google: Progress and the path ahead.](https://research.google/blog/ai-in-software-engineering-at-google-progress-and-the-path-ahead/) | Google software engineers are widely enthusiastic about AI's role in coding, and this blog post highlights the company’s latest AI-driven enhancements along with a framework for building valuable AI products. Google closely tracks developer productivity and satisfaction, and the improvements discussed have shown measurable impact on both. The company prioritizes initiatives based on technical feasibility and potential impact, iterates rapidly, and uses detailed metrics to refine and assess effectiveness.|
|[AI Tools Make Experienced Developers Slower, METR Study Finds.](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) |A randomized controlled trial of 16 experienced open-source developers found AI tools decreased task completion time by 19%, despite developers self-reporting a 24% speedup. |
|[Batch Mode in the Gemini API: Process more for less.](https://developers.googleblog.com/en/scale-your-ai-workloads-batch-mode-gemini-api/) |The Gemini API now includes an asynchronous endpoint tailored for high-throughput tasks where low latency isn’t essential. Called Batch Mode, it lets users submit large jobs, offload scheduling and processing, and retrieve results within 24 hours—all at a 50% discount compared to synchronous APIs. It's ideal for scenarios where data is ready in advance and immediate responses aren’t required. |
|[Image-to-Video Generation in Google's Veo 3.](https://blog.google/technology/google-labs/flow-adds-speech-expands/) |Google has added speech generation to Flow, its AI video tool, allowing users to transform images into talking clips using Veo 3. Alongside this, Flow and Google AI Ultra are now available in over 140 countries, expanding access to advanced video and AI features worldwide. |
|[Gemini 3 incoming?](https://threadreaderapp.com/thread/1942995482592043175.html) |gemini-beta-3.0-pro was just referenced in the latest Gemini-CLI commit. |
|[Grok Coming to Tesla Vehicles.](https://threadreaderapp.com/thread/1943251229511160001.html) |Elon Musk confirmed that xAI's chatbot Grok will be integrated into Tesla vehicles "next week at the latest." |
|[The CEO who never was: how Linda Yaccarino was set up to fail at Elon Musk’s X.](https://www.theguardian.com/technology/2025/jul/10/linda-yaccarino-resigns-x-elon-musk) |Ex-NBC executive was tasked with building an ‘everything app’, but billionaire owner was biggest obstacle in her path |
|[UK government’s deal with Google ‘dangerously naive’, say campaigners.](https://www.theguardian.com/technology/2025/jul/09/uk-governments-deal-with-google-dangerously-naive-say-campaigners) | Company to provide free technology and ‘upskill’ civil servants but concerns raised over UK data being held on US servers|
|[Musk’s AI firm forced to delete posts praising Hitler from Grok chatbot.](https://www.theguardian.com/technology/2025/jul/09/grok-ai-praised-hitler-antisemitism-x-ntwnfb) | The popular bot on X began making antisemitic comments in response to user queries|


## Resources
|Link|description|
|---|---|
|[Adding Memory to Gemini 2.5 Chatbots.](https://www.philschmid.de/gemini-with-memory) |A tutorial on using the Gemini API alongside the open-source mem0 tool to equip Gemini 2.5 chatbots with long-term memory. This setup helps bots remember previous interactions, tailor their replies, and avoid repeating information, leading to more contextually aware conversations. |
|[agent-squad.](https://github.com/awslabs/agent-squad) |A framework for building collaborative multi-agent AI systems that can plan, delegate, and work together to solve complex tasks. |
|[Economics of Claude 3 Opus Inference.](https://x.com/tessera_antra/status/1941563920587817203) |Anthropic has announced it will deprecate API access to Claude 3 Opus, citing a legitimate operational challenge. This article explores the economics behind running models at reduced scale and considers alternative solutions that could benefit both Anthropic and independent researchers. Maintaining inference access to Claude 3 Opus involves more complexity than is immediately apparent. |
|[Microjax: JAX in two classes and six functions.](https://github.com/joelburget/microjax) |Microjax is a tiny autograd engine with a Jax-like API. It was inspired by Andrej Karpathy's Micrograd, a PyTorch-like library with about 150 lines of code. JAX uses a more functional style, which some developers prefer. |
|[BitNet.](https://github.com/microsoft/BitNet) |An inference framework for Microsoft's BitNet b1.58, a 1.58-bit (ternary) large language model designed for efficient and lossless CPU inference using optimized low-bit kernels. |
|[Google Explores AI in Mental Health Treatment.](https://blog.google/technology/health/new-mental-health-ai-tools-research-treatment/) | Google announced two mental health AI initiatives: a practical guide to responsibly deploying AI in mental health care, and a multi-year research partnership with DeepMind and Wellcome Trust to study AI-driven diagnosis and treatment of anxiety, depression, and psychosis.|
|[The OLMo 2 model family.](https://allenai.org/olmo) | OLMo 2 is a fully open family of language models, with OLMo 2 32B as its most advanced version—marking the first fully open model to outperform GPT-3.5 Turbo and GPT-4o mini on key benchmark suites. The 7B and 13B variants hold their own against leading open-weight models from Meta and Mistral on English academic tasks. Even the smallest model, OLMo 2 1B, outperforms peers like Gemma 3 1B and Llama 3.2 1B.|
|[SmolLM3 Released by Hugging Face.](https://huggingface.co/blog/smollm3) | Hugging Face's SmolLM3 is a fully open 3B-parameter language model that supports six languages, strong reasoning capabilities, and long-context processing. It targets high performance in the small model segment.|
|[Mem0.](https://github.com/mem0ai/mem0) | An open-source memory layer for AI agents that enables long-term, personalized interactions by efficiently storing and retrieving user context across sessions, reducing token costs and improving response accuracy.|
|[NotebookLlaMa.](https://github.com/run-llama/notebookllama) | A fully open-source, LlamaCloud-backed alternative to NotebookLM, this project uses LlamaCloud for document processing, OpenAI for content generation, and ElevenLabs for voice synthesis.|
|[Spatiotemporal Attention for MI-EGG Decoding.](https://github.com/snailpt/TCANet) | TCANet layers multi‑scale convolutions, temporal compression, and stacked self‑attention to model motor‑imagery EEG.|
|[MiniMax-M1.](https://github.com/MiniMax-AI/MiniMax-M1) |MiniMax's 456B parameter model uses a hybrid mixture-of-experts architecture with "lightning attention" that processes 1 million token contexts (8x DeepSeek R1) while using 25% fewer FLOPs at 100K token generation lengths. |
|[Gemma 3n and MatFormer in Practice.](https://colab.research.google.com/github/google-gemini/gemma-cookbook/blob/main/Gemma/%5BGemma_3n%5DMatFormer_Lab.ipynb) |A hands-on tutorial for experimenting with Gemma 3n and MatFormer, Google's nested transformer model that supports elastic inference via the Mix-n-Match technique. |
|[Google's MCP Toolbox for Databases.](https://github.com/googleapis/genai-toolbox) |The open-source server lets AI agents query databases and automatically handles connection pooling, authentication, and security for database interactions. |
|[Devstral Models for Code Agents.](https://mistral.ai/news/devstral-2507) | Mistral AI and All Hands AI released Devstral Medium and upgraded Devstral Small 1.1, optimized for agentic coding tasks. Devstral Small 1.1 is open-source under Apache 2.0 and outperforms all other open models for code agents.|
|[The Architecture Behind Lovable and Bolt.](https://www.beam.cloud/blog/agentic-apps) | The effectiveness of a coding app hinges more on context engineering and solid software architecture than on the underlying model's raw capabilities. Successful platforms typically share four key components: typed prompts integrated with test-driven development, MCP servers for secure sandboxed execution, agent loops for managing state, and real-time coordination with the frontend.|
|[asyncmcp.](https://github.com/bh-rat/asyncmcp) |**asyncmcp** is a variant of the Model Context Protocol (MCP) that operates over queues rather than requiring immediate responses. It’s designed for situations where context isn't instantly available and processing takes time. By using an asynchronous transport layer, asyncmcp routes requests to internal queues for later handling, allowing clients to send tasks without waiting for a response—ideal for deferred or background processing scenarios. |
|[T5Gemma: Encoder-Decoder Models.](https://developers.googleblog.com/en/t5gemma/) |Google's T5Gemma is a suite of encoder-decoder LLMs adapted from decoder-only Gemma 2 models. Designed for tasks like summarization and translation, T5Gemma includes pretrained and instruction-tuned variants in sizes ranging from 2B to XL. |
|[Introducing FlexOlmo: a new paradigm for language model training and data collaboration.](https://allenai.org/blog/flexolmo) | This novel architecture enables real-time activation or deactivation of individual data contributions without retraining the entire model. It works by training separate expert modules on private datasets while using a frozen public model as a shared anchor. These modules are then dynamically merged using domain-informed routing, eliminating the need for joint training and allowing instant, flexible model modification.|


## Perspectives
|Link|description|
|---|---|
|[The American DeepSeek Project.](https://www.interconnects.ai/p/the-american-deepseek-project) |Meta's recent AI struggles have left a gap in the open-source AI landscape, now mostly occupied by Chinese models. If this trend persists, the AI field could divide into two camps: high-performing but costly closed-source models from the U.S., and affordable, widespread, yet possibly insecure models from China. The U.S. likely has a narrow window—around two years—to reverse this by investing \$100–500 million in an open-source model that rivals the best proprietary ones. |
|[What can agents actually do?](https://lethain.com/what-can-agents-do/) |While there's plenty of hype around AI, much of the discussion is so abstract it becomes unhelpful. This post aims to clearly explain how AI agents function, using a few real-world examples. AI agents can significantly enhance software quality and system design—but if the underlying systems are flawed, agents can actually make things worse. |
|[Why I don't think AGI is right around the corner.](https://www.dwarkesh.com/p/timelines-june-2025) |Getting LLMs to perform consistent, humanlike work is difficult because they’re missing key capabilities. One major issue is their inability to improve over time—without continual learning, they stay fixed at their initial skill level. There's also no effective way to give them nuanced, human-style feedback. Tweaking system prompts falls far short of the kind of learning humans go through. Unlike people, LLMs can’t build context over time, reflect on their mistakes, or gradually refine their performance through practice. |
|[A Review of Alpha School, the private school with 2-hour days and AI teachers.](https://www.astralcodexten.com/p/your-review-alpha-school) |A year-long investigation by a parent revealed that the \$40,000 Austin school operates with 3.5-hour school days, a 5:1 student-teacher ratio, and strong incentive systems—contrary to its marketing as an AI-driven, teacher-free model. While students progress through material 2.6 times faster using personalized learning tools, parents argue the real benefit isn't acceleration, but time: the model could give children around nine extra years outside the classroom to explore their own interests. |
|[The ‘ChatGPT Moment’ in Robotics and beyond.](https://paritoshmohan.substack.com/p/the-chatgpt-moment-in-robotics-and) | Just three years ago, reliable robotic object manipulation demanded large engineering teams. Today, a college student can fine-tune an open-source vision-language-action model over a weekend and achieve results that once took months. This article explores what a "ChatGPT moment" for robotics might look like, surveys the current landscape, highlights emerging technologies, and predicts likely leaders. While the presence of robots in daily life may initially feel surreal, they'll soon become as essential and commonplace as AI assistants are today.|
|[Automating oral argument.](https://adamunikowsky.substack.com/p/automating-oral-argument) | A Harvard Law graduate and former Supreme Court advocate tested Claude 4 Opus by feeding it his case briefs and having it respond to the same questions posed by the Justices. The AI delivered what he described as an “outstanding oral argument,” offering coherent answers and insightful points he hadn’t considered. He concluded that AI lawyers may soon surpass even the best human advocates in oral argument performance.|
|[People Are Using AI Chatbots to Guide Their Psychedelic Trips.](https://www.wired.com/story/people-are-using-ai-chatbots-to-guide-their-psychedelic-trips/) |In the few states where it’s legal, in-person psychedelic therapy often costs thousands per session and involves long wait times. As a more accessible alternative, some users are turning to AI tripsitters. Companies like Mindbloom are also starting to integrate AI into their ketamine treatment programs. However, experts remain skeptical, arguing that AI lacks the emotional attunement necessary for safe and effective psychedelic experiences. |
|[o3 Turns Pro.](https://thezvi.substack.com/p/o3-turns-pro) | o3-pro generally delivers higher-quality answers than o3, but with noticeably longer response times. At scale, the API costs for o3-pro can be steep, making parallel querying via the chat interface a more practical option. Since o3-pro targets the same use cases as o3, users considering Opus may find it more effective to use Opus instead of—or alongside—o3-pro. The recent 80% price drop for o3 has a bigger overall impact, while o3-pro remains best suited for special-case scenarios.|
|[What We Learned from Briefing 70+ Lawmakers on the Threat from AI.](https://www.lesswrong.com/posts/Xwrajm92fdjd7cqnN/what-we-learned-from-briefing-70-lawmakers-on-the-threat) |AI risk briefings revealed that most UK parliamentarians have limited AI expertise and face capacity constraints that hinder in-depth research on the topic. Despite this, the briefings were positively received, with about one-third of lawmakers publicly endorsing AI risk mitigation efforts. Successful outreach strategies included consistent follow-ups and the use of statements from respected AI experts to emphasize the gravity of extinction-level risks. |
|[How big could an “AI Manhattan Project” get?](https://epoch.ai/gradient-updates/how-big-could-an-ai-manhattan-project-get) |Amid increasing calls for a national AI initiative to rival China, projections indicate that by late 2027, U.S. compute capacity could support training runs 10,000 times larger than GPT-4. This level of unified scaling could push AI progress roughly two years ahead of current industry forecasts. |
|[OpenAI Product Leader: The 4D Method to Build AI Products That Users Actually Want.](https://creatoreconomy.so/p/openai-product-leader-the-4d-method-to-build-ai-products-miqdad) | Miqdad Jaffer, a product leader at OpenAI, presents the 4D framework for creating AI tools that address real-world problems. The method consists of Discover, Design, Develop, and Deploy—focusing on identifying user needs, designing AI that builds trust seamlessly, developing with resilience, and delivering impactful first-use experiences. |
|[The Only SaaS Feature You Should Be Building.](https://www.henrypray.com/writings/the-only-saas-feature-you-should-be-building) |Before companies can realize a future where AI agents handle real work by interacting with both data and people, they must first solve the action interface for human operators. This interface is crucial because most operators aren’t prompt engineers—their focus is on keeping operations running smoothly. They need a clear, intuitive UI that allows them to confirm or respond to system actions and easily request actions from the AI. This article explores a new paradigm for designing such interfaces, aiming to create a seamless and empowering experience for operators. |


#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################

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
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |













































































































































