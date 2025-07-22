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
|[Scaling Context Requires Rethinking Attention.](https://arxiv.org/pdf/2507.04239) | A new “Power” attention mechanism introduces a hyperparameter p to independently control state size, addressing the trade-off between computational cost and long-context training. It outperforms standard attention on long sequences and supports custom GPU kernels that are 8.6x faster than Flash Attention at a 64k context length.|
|[.]() | |
|[.]() | |
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
|[Passage of Time.](https://github.com/jlumbroso/passage-of-time-mcp) | Passage of Time is an MCP (Model Context Protocol) server that equips language models with temporal awareness and time calculation capabilities. By leveraging these temporal tools, models can gain unique insights into conversation patterns, work rhythms, and the human perception of time. This implementation highlights the potential of MCP—not just for building smarter tools, but for enabling AI systems to perceive and interpret human experiences more deeply, fostering genuine mutual understanding between humans and AI.|
|[Introducing Amazon Bedrock AgentCore: Securely deploy and operate AI agents at any scale .](https://aws.amazon.com/it/blogs/aws/introducing-amazon-bedrock-agentcore-securely-deploy-and-operate-ai-agents-at-any-scale/) | Amazon Bedrock AgentCore is a comprehensive suite of enterprise-grade services designed to help developers rapidly and securely deploy AI agents at scale, regardless of the framework or model used. It includes modular services that work seamlessly together, removing the need for developers to manually integrate components. AgentCore simplifies infrastructure management and operational complexity. With **AgentCore Runtime**, developers can also discover, purchase, and run pre-built agents and tools directly from the AWS Marketplace.|
|[Claude Code revenue jumps 5.5x as Anthropic launches analytics dashboard.](https://venturebeat.com/ai/anthropic-adds-usage-tracking-to-claude-code-as-enterprise-ai-spending-surges/) | Anthropic is introducing a comprehensive analytics dashboard for its Claude Code AI programming assistant. The dashboard offers engineering managers detailed insights into how their teams are using Claude Code, addressing growing demands from companies for concrete data to justify AI investments. It provides visibility into which teams and individuals are deriving the most value from these premium, high-cost tools.|
|[Meta reportedly scores two more high-profile OpenAI researchers.](https://techcrunch.com/2025/07/16/meta-reportedly-scores-two-more-high-profile-openai-researchers/) | OpenAI researcher Jason Wei will join Meta’s new Superintelligence Lab, reports Wired, citing two sources familiar with the matter. Another team member, Hyung Won Chung, may also join Meta. Sources told Wired that both the researchers’ internal OpenAI Slack profiles are currently deactivated. |
|[Scale AI lays off 14% of staff, largely in data-labeling business.](https://techcrunch.com/2025/07/16/scale-ai-lays-off-14-of-staff-largely-in-data-labeling-business/) |Data-labeling startup Scale AI is laying off 200 employees, roughly 14% of its staff, and cutting ties with 500 of its global contractors, Bloomberg reported on Wednesday. The cuts come just a month after Meta hired Scale AI’s CEO in a $14.3 billion deal. |
|[Anthropic hired back two of its employees — just two weeks after they left for a competitor.](https://www.theverge.com/ai-artificial-intelligence/708521/anthropic-hired-back-two-of-its-employees-just-two-weeks-after-they-left-for-a-competitor) | Boris Cherny and Cat Wu are reportedly back at Anthropic after departing for Anysphere, the developer of Cursor.|
|[OpenAI says it will use Google’s cloud for ChatGPT.](https://www.cnbc.com/2025/07/16/openai-googles-cloud-chatgpt.html) | The Google infrastructure will run in the US, Japan, the Netherlands, Norway, and the UK.|
|[Thinking Machines Lab will launch its first AI product soon with ‘a significant open source component’.](https://bgr.com/business/thinking-machines-lab-will-launch-its-first-ai-product-soon-with-a-significant-open-source-component/) | Thinking Machines Lab's first product will include a significant open source component and be useful for researchers and startups developing custom models.|
|[Claude Sonnet 4 is back.](https://threadreaderapp.com/thread/1945599013954490523.html#google_vignette) | Windsurf now has Claude Sonnet 4 again with first party support from Anthropic.|
|[Netflix uses generative AI in one of its shows for first time.](https://www.theguardian.com/media/2025/jul/18/netflix-uses-generative-ai-in-show-for-first-time-el-eternauta) | Firm says technology used in El Eternauta is chance ‘to help creators make films and series better, not just cheaper’|
|[OpenAI launches personal assistant capable of controlling files and web browsers.](https://www.theguardian.com/technology/2025/jul/17/openai-launches-personal-assistant-capable-of-controlling-files-and-web-browsers) | AI agent can find restaurant reservations and go shopping for users, but OpenAI acknowledges there are ‘more risks’|
|[Royal Society suggested to Elon Musk he consider resigning science fellowship.](http://theguardian.com/science/2025/jul/17/royal-society-elon-musk-resign-science-fellowship-tesla) |Fellows called on academy to act over Tesla owner’s role in Trump administration’s attacks on research |
|[Introducing ChatGPT Agent: bridging research and action.](https://openai.com/it-IT/index/introducing-chatgpt-agent/) |ChatGPT Agent blends Operator’s web browsing with Deep Research’s analytical depth, running on its own virtual computer to tackle complex, multi-step tasks like managing calendars, conducting competitive analysis, and creating polished slideshows — all in one seamless workflow. |
|[Anthropic tightens usage limits for Claude Code — without telling users.](https://techcrunch.com/2025/07/17/anthropic-tightens-usage-limits-for-claude-code-without-telling-users/) | Since Monday morning, Claude Code users have been hit with unexpectedly restrictive usage limits. The problems, many of which have been aired on Claude Code’s GitHub page, seem to be concentrated among heavy users of the service, many of whom are on the $200-a-month Max plan. |
|[Shopify's internal AI adoption strategy: unlimited spend and "MCP everything".](https://www.firstround.com/ai/shopify) |Shopify purchased 3,000 Cursor licenses with unlimited token access after getting legal approval to broadly adopt AI tools. They built an internal LLM proxy with MCPs to connect all their data sources. Now, even non-technical sales reps build performance auditing tools in Cursor, and a sales engineer runs his full workflow from a single dashboard that pulls real-time data from Salesforce, Slack, and GSuite — no need to open any of those apps. |
|[The AI Cloud: A unified platform for AI workloads.](https://vercel.com/blog/the-ai-cloud-a-unified-platform-for-ai-workloads) | Vercel recently launched the AI Cloud, a platform that streamlines AI app development with integrated tools like the AI SDK and AI Gateway for flexible, secure execution. It uses fluid compute to optimize workloads by efficiently handling idle times and bursts, which helps cut costs. The platform also includes Vercel BotID for securing sensitive routes and Vercel Sandbox for safely running untrusted code — all paving the way for the agentic era of web development.|
|[Perplexity sees India as a shortcut in its race against OpenAI.](https://techcrunch.com/2025/07/17/perplexity-sees-india-as-a-shortcut-in-its-race-against-openai/) | While OpenAI has cemented its lead in the U.S., Perplexity is taking a different route — quietly expanding into India to compete in the next phase of AI adoption. The search-focused AI startup is rapidly adding millions of users in the world’s second-largest internet and smartphone market, positioning itself for mass-market scale. This week, Perplexity partnered with Bharti Airtel, India’s second-largest telecom operator after Reliance Jio, to offer a free 12-month Perplexity Pro subscription — normally worth $200 — to all 360 million Airtel subscribers. Airtel confirmed to TechCrunch that the deal is exclusive, meaning no other telco in the country can offer Perplexity’s services, including free access, to their subscribers.|
|[Mistral Adds Deep Research, Projects, Image Editing, and Voice Capabilities to Le Chat.](https://mistral.ai/news/le-chat-dives-deep) | The deep research mode breaks down complex questions, gathers sources from the web, and builds structured reports, while the new Voxtral-powered voice mode enables audio-in.|
|[OpenAI launches bio bug bounty.](https://openai.com/bio-bug-bounty/) | After classifying ChatGPT Agent as high bio/chemical risk, OpenAI launched a program to pay $25,000 to the first researcher to submit a universal jailbreak that answers all 10 challenge questions.|
|[Windsurf Wave 11.](https://windsurf.com/blog/windsurf-wave-11) |Windsurf's Wave 11 includes startups across AI-native productivity, dev tools, robotics, and consumer agents. |
|[DuckDuckGo now lets you hide AI-generated images in search results.](https://techcrunch.com/2025/07/18/duckduckgo-now-lets-you-hide-ai-generated-images-in-search-results/) |Privacy-focused browser DuckDuckGo is rolling out a new setting that lets users filter out AI images in search results. The company says it’s launching the feature in response to feedback from users who said AI images can get in the way of finding what they’re looking for. |
|[Bernie Sanders says that if AI makes us so productive, we should get a 4-day workweek.](https://techcrunch.com/2025/06/25/bernie-sanders-says-that-if-ai-makes-us-so-productive-we-should-get-a-4-day-work-week/) | As AI companies rave about how their products are revolutionizing productivity, Senator Bernie Sanders (I-VT) wants the tech industry to put its money where its automated mouth is.|
|[Exhausted man defeats AI model in world coding championship.](https://arstechnica.com/ai/2025/07/exhausted-man-defeats-ai-model-in-world-coding-championship/) |A Polish programmer running on fumes recently accomplished what may soon become impossible: beating an advanced AI model from OpenAI in a head-to-head coding competition. The 10-hour marathon left him "completely exhausted." |
|[Billionaires Convince Themselves AI Chatbots Are Close to Making New Scientific Discoveries.](https://gizmodo.com/billionaires-convince-themselves-ai-is-close-to-making-new-scientific-discoveries-2000629060) |Generative artificial intelligence tools like ChatGPT, Gemini, and Grok have exploded in popularity as AI becomes mainstream. These tools don’t have the ability to make new scientific discoveries on their own, but billionaires are convinced that AI is on the cusp of doing just that. And the latest episode of the All-In podcast helps explain why these guys think AI is extremely close to revolutionizing scientific knowledge. |
|[‘You can make really good stuff – fast’: new AI tools a gamechanger for film-makers.](https://www.theguardian.com/technology/2025/jul/20/artificial-intelligence-ai-tools-gamechanger-for-film-makers) |Instead of spending millions and taking years to complete, creative directors are producing high-grade work using the latest software, but critics voice copyright concerns |
|[Meta allows ads crowdfunding for IDF drones, consumer watchdog finds.](https://www.theguardian.com/technology/2025/jul/21/meta-idf-drone-ads-israel) | Paid ads hosted on Facebook, Instagram and Threads seem to violate Meta’s stated policies yet remain active|
|[Human-level AI is not inevitable. We have the power to change course.](https://www.theguardian.com/commentisfree/ng-interactive/2025/jul/21/human-level-artificial-intelligence) | Technology happens because people make it happen. We can choose otherwise|
|[OpenAI launches personal assistant capable of controlling files and web browsers.](https://www.theguardian.com/technology/2025/jul/17/openai-launches-personal-assistant-capable-of-controlling-files-and-web-browsers) |AI agent can find restaurant reservations and go shopping for users, but OpenAI acknowledges there are ‘more risks’ |
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
|[Stanford's Marin foundation model: The first fully open model developed using JAX.](https://developers.googleblog.com/en/stanfords-marin-foundation-model-first-fully-open-model-developed-using-jax/) |Stanford’s Marin project is designed to promote full transparency in foundation model research by sharing not just the models themselves, but also the entire development process—including code, datasets, data methodologies, experiments, hyperparameters, and training logs. This initiative aims to advance openness and reproducibility in AI research. The project’s first releases, **Marin-8B-Base** and **Marin-8B-Instruct**, are available under the permissive Apache 2.0 license. This article explores the engineering challenges the team faced in building open, scalable, and truly reproducible foundation models. |
|[Kimi K2: Open Agentic Intelligence.](https://moonshotai.github.io/Kimi-K2/) | Moonshot AI’s Kimi K2 is a 1T parameter Mixture-of-Experts model (32B active) built for agentic tasks, not just knowledge responses, and released as open-source for research and deployment. It achieves state-of-the-art open-agent coding results, outperforming models like DeepSeek and Qwen3 and rivaling Claude Sonnet 4, with 65.8% on SWE-bench Verified. Demonstrating robust statistical reasoning, Kimi K2 completes complex workflows like salary analysis through tool execution. Its stable training on 15.5T tokens is enabled by the MuonClip optimizer, while its agentic abilities are honed via ACEBench-inspired, rubric-driven simulations and RL with both verifiable and non-verifiable rewards.|
|[A Survey on Latent Reasoning.](https://arxiv.org/abs/2507.06203) |This paper surveys latent reasoning, an emerging approach where AI performs inference within continuous hidden states rather than explicit token-based chains of thought. It identifies two main methods: vertical recurrence, which refines reasoning by looping through layers, and horizontal recurrence, which evolves compressed states over long contexts. The study also highlights infinite-depth models like text diffusion, enabling parallel, iterative reasoning for global planning and self-correction—offering more expressive, efficient alternatives to traditional autoregressive reasoning. |
|[How to Evaluate AI Agents to Predict Future Events.](https://huggingface.co/blog/futurebench) |Hugging Face's FutureBench is a benchmark for testing AI agents on their ability to predict future events across domains like science, geopolitics, and technology. |
|[Updated FineWeb with 18.5 Trillion Tokens.](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |FineWeb has been updated with English data from CommonCrawl snapshots from January to June 2025. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Gaslight-driven development.](https://tonsky.me/blog/gaslight-driven-development/) | Sometimes we act simply because a computer told us to, and now large language models are influencing how developers design APIs by suggesting what they *should* look like—leaving developers with little choice but to comply. This dynamic can be valuable, as AI effectively provides a "newbie’s perspective" on tool design, revealing how interfaces might have been more intuitive from the start.|
|[How to avoid nuclear war in an era of AI and misinformation.](https://www.nature.com/articles/d41586-025-02260-z) | Nuclear deterrence is no longer a two-player game, and emerging technologies further threaten the status quo. The result is a risky new nuclear age.|
|[Google tapped billions of mobile phones to detect quakes worldwide — and send alerts.](https://www.nature.com/articles/d41586-025-02278-3) |Study reveals how the tech behemoth is using the motions sensors on phones to expand quake warnings to more countries. |
|[Hidden Technical Debt in AI.](https://tomtunguz.com/hidden-technical-debt-in-ai/) | AI systems, including LLMs, often promise simplicity, but in practice they require significant infrastructure, data management, and operational overhead. To control costs and complexity, teams still need deterministic software and traditional ML models for tasks like tool selection and system monitoring. It’s a lot like earlier ML systems — the supposed “AI magic box” comes with hidden technical debt and layers of complexity beneath the surface.|
|[The Weighted Perplexity Benchmark: Tokenizer-Normalized Evaluation for Language Model Comparison.](https://www.lesswrong.com/posts/csNk8ECk9SiKHkw35/the-weighted-perplexity-benchmark-tokenizer-normalized) | The Weighted Perplexity Benchmark provides a tokenizer-normalized way to evaluate language models more fairly. By adjusting perplexity scores to account for differences in tokenization, it enables more accurate and meaningful comparisons between models, improving how we assess NLP performance.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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


|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |













































































































































