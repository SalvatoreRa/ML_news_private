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
|[s1: Simple test-time scaling.](https://arxiv.org/abs/2501.19393) | Researchers from Stanford, UW, and others introduced s1, a method to enhance LLM performance by using additional compute during inference ("test-time scaling"). Key ideas include: Small but effective dataset – They created s1K, a set of 1,000 challenging questions with detailed reasoning, to fine-tune a 32B model. Despite the small size, it provides valuable reasoning examples. "Budget forcing" for reasoning – A new decoding method adds the token "Wait" when the model attempts to stop, encouraging it to rethink and correct its reasoning. It also limits excessive reasoning to control inference time. Significant improvements over OpenAI’s o1 – The fine-tuned model (s1-32B), based on Qwen2.5-32B-Instruct, outperforms OpenAI's o1-preview by up to 27% on math competitions (MATH & AIME24). Test-time scaling increases accuracy on AIME24 from 50% to 57%, exceeding its normal performance.|
|[OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models.](https://arxiv.org/abs/2502.01061) |ByteDance AI Lab introduced OmniHuman-1, a diffusion-transformer model that creates realistic human videos from a single image and motion input (audio or video). Key points: End-to-end human video generation – OmniHuman uses an image and audio or video to generate lifelike videos of people speaking or performing actions, with impressive detail in motion, lighting, and texture. Mixed modality training – Omni-Conditions Training combines various motion modalities during training, expanding data and overcoming the lack of high-quality talking-head videos. The model handles diverse inputs like speech, song, and complex poses. Outperforms prior methods – OmniHuman produces more realistic videos and works with a variety of inputs, including cartoons or animals, transferring motion naturally. Broader support – The model supports any portrait content (face, half-body, full-body) and multiple driving signals, offering more versatility than previous models. |
|[LIMO: Less is More for Reasoning.](https://arxiv.org/abs/2502.03387) |The LIMO paper challenges the need for large fine-tuning datasets in complex reasoning tasks, showing that a small set of carefully curated examples can be highly effective. With just 817 training samples, the LIMO model achieved impressive results, scoring 57.1% on the AIME math competition and 94.8% on MATH, far surpassing earlier models that required much more data. The model also demonstrated significant out-of-distribution generalization, outperforming models trained on 100 times more data by 40.5% on various benchmarks. The authors propose that when an LLM has strong pre-existing knowledge, only a minimal set of high-quality examples is necessary to unlock advanced reasoning skills. This suggests that small, well-designed datasets could enable state-of-the-art reasoning, lowering the barriers for fine-tuning LLMs. |
|[CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning.](https://arxiv.org/abs/2502.02390) |CoAT introduces a “slow thinking” inference framework that enhances LLM reasoning by allowing it to explore and update its thoughts more like a human. The system combines Monte Carlo Tree Search (MCTS) with associative memory, enabling the model to explore different reasoning branches and dynamically add relevant information as needed. This iterative approach allows the model to refine and revisit intermediate conclusions, improving accuracy and comprehensiveness compared to one-pass reasoning. In experiments, CoAT outperformed traditional methods on accuracy, coherence, and solution diversity. By mimicking human-like problem-solving, CoAT points toward LLMs that use search and memory for more reliable reasoning. |
|[Syntriever: How to Train Your Retriever with Synthetic Data from LLMs.](https://arxiv.org/abs/2502.03824) |Syntriever introduces a two-stage framework to build a high-quality text retriever without relying on large labeled datasets or access to an LLM’s internals. In Stage 1, the system distills knowledge by generating synthetic Q&A data. A powerful LLM (e.g., GPT-4) is prompted to create relevant and incorrect passages, with chain-of-thought ensuring variety. The LLM then filters out any low-quality data, resulting in a synthetic dataset that is used to train the retriever. In Stage 2, the retriever is further aligned with the LLM’s preferences using a partial Plackett-Luce ranking method to adjust its ranking decisions. Syntriever achieves state-of-the-art results on several retrieval benchmarks without needing any real training queries, all training data is generated synthetically by the LLM. It also eliminates the need for logits, making it applicable even to closed models. |
|[Demystifying Long Chain-of-Thought Reasoning in LLMs.](https://arxiv.org/abs/2502.03373) | This study examines how LLMs develop extended chain-of-thought (CoT) reasoning, focusing on reinforcement learning (RL) and compute scaling. It finds that supervised fine-tuning (SFT) improves accuracy by using long CoT sequences, and introduces a cosine length-scaling reward with repetition penalties to stabilize RL and prevent unnecessary reasoning lengthening. RL models trained with noisy, web-based supervision signals generalize better to out-of-distribution tasks, though filtering is essential for stability. Additionally, while skills like error correction exist in base models, effective RL incentives are needed to harness them for complex tasks. This paper provides a roadmap for enhancing CoT training with RL and reward tuning.|
|[Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?](https://arxiv.org/abs/2502.00674) |This paper explores whether mixing different LLMs in an ensemble (Mixture-of-Agents, MoA) improves performance or if using a single top model’s outputs is more effective. The surprising answer is that "Self-MoA," which generates multiple outputs from one strong model and aggregates them, often outperforms multi-model ensembles. Extensive tests show that Self-MoA yields better results, with a +6.6% score improvement on the AlpacaEval 2.0 benchmark compared to MoA, and +3.8% on tasks like MMLU, CRUX, and MATH. The study finds that adding weaker models in an MoA can dilute performance, and unless all models are strong and complementary, it’s better to rely on one top model’s outputs. They also propose a sequential version of Self-MoA that efficiently combines multiple outputs over rounds. |
|[Multi-agent Architecture Search via Agentic Supernet.](https://arxiv.org/abs/2502.04180) | MaAS (Multi-agent Architecture Search) automates the design of multi-agent systems for LLMs, where agents collaborate with specific roles or tools for each task. Instead of hand-designing a complex pipeline, MaAS learns a flexible “agentic supernet” that can generate an optimal agent team for each query. It defines a continuous space of possible agent configurations and dynamically selects the best one based on the query's domain and difficulty, allowing for efficient resource allocation. MaAS outperforms traditional multi-agent systems in accuracy by 0.5–11.8%, while using only 6–45% of the inference cost. Its approach also shows strong generalization, transferring well to new tasks and LLM backbones.|
|[Analyze Feature Flow to Enhance Interpretation and Steering in Language Models.](https://arxiv.org/abs/2502.03032) |This paper presents a new method for tracking the evolution of features discovered by sparse autoencoders across layers of large language models. Using a data-free cosine similarity technique, it maps feature persistence, transformation, and emergence. The paper shows how cross-layer feature maps allow for direct control of model behavior through feature manipulation, offering deeper mechanistic insights into model computations via detailed flow graphs. |
|[Building Bridges between Regression, Clustering, and Classification.](https://arxiv.org/abs/2502.02996) |This paper presents a new approach to enhancing neural network training for regression tasks by framing them as classification problems with a learned target encoder-decoder pair. The method surpasses traditional regression techniques by utilizing distributional target representation and enabling smooth interpolation across different objectives. |
|[Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.](https://arxiv.org/abs/2502.05171) | A 3.5B model trained for latent reasoning shows significant improvements on math problems. The model utilizes a recurrent architecture.|
|[GNNs Getting ComFy: Community and Feature Similarity Guided Rewiring.](https://arxiv.org/abs/2502.04891) |While maximizing spectral gap has been used to tackle over-squashing in GNNs, this paper demonstrates that minimizing it can also enhance generalization. The authors introduce three new rewiring strategies to improve label-community alignment, homophily, and computational efficiency. |
|[Agency Is Frame-Dependent.](https://arxiv.org/abs/2502.04403) |Determining if a system is "agentic" is challenging. This position paper from DeepMind suggests that measuring agency requires a set reference frame and cannot be considered a global property of a system. |
|[Improved LLM Pretraining.](https://arxiv.org/abs/2502.06733) | This paper introduces dynamic instance-level data reweighting techniques for LLM pretraining. By adjusting the importance of samples based on loss values during training, these methods enhance both efficiency and effectiveness.|
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
|[gambling firms secretly sharing users’ data with Facebook without permission.](https://www.theguardian.com/society/2025/feb/08/gambling-firms-secretly-shared-users-data-with-facebook-without-permission) | Meta accounts of those affected flooded with ads for casinos and betting sites|
|[From Dogecoin to $Trump: everything you need know about the wild world of meme coins.](https://www.theguardian.com/technology/2025/feb/09/from-dogecoin-to-trump-everything-you-need-know-about-the-wild-world-of-meme-coins) | Are they the same as crypto, why has the US president launched one, and who’s really coining it in? Here’s a complete guide to the latest digital money mania|
|[Google Maps changed the way we get around. It all began in a spare bedroom in Sydney.](https://www.theguardian.com/technology/2025/feb/09/google-maps-turns-20-anniversary-feature) |This weekend the mapping platform turns 20 – and Stephen Ma is writing himself and his friends back into its origin story |
|[‘Mass theft’: Thousands of artists call for AI art auction to be cancelled.](https://www.theguardian.com/technology/2025/feb/10/mass-theft-thousands-of-artists-call-for-ai-art-auction-to-be-cancelled) |Letter says many of works being sold by Christie’s are made by AI models trained on pieces by human artists, without a licence |
|[Mistral le Chat.](https://mistral.ai/en/news/all-new-le-chat) |Mistral introduces a new chat assistant capable of processing 1,000 words per second. Powered by Mistral's advanced coding models, it features a user-friendly interface to help with a variety of tasks. |
|[Pika Video Editing.](https://pikartai.com/pikaddition/) |Pika Labs has launched Pikadditions, an AI tool that effortlessly adds objects and characters to videos, maintaining a high level of realism. |
|[Germany Trade & Invest: OpenAI Expands to Germany.](https://www.prnewswire.com/news-releases/germany-trade--invest-openai-expands-to-germany-302371354.html) |OpenAI announces plans to establish a new office in Munich in the coming months. |
|[Elon Musk-led group makes surprise bid of nearly $100bn for OpenAI.](https://www.theguardian.com/technology/2025/feb/10/elon-musk-open-ai-bid) |Sam Altman, OpenAI’s CEO and co-founder, responded that he would not accept and offered to buy X instead |
|[Macron touts Europe and trolls Trump at Paris AI summit.](https://www.theguardian.com/technology/2025/feb/10/macron-europe-trolls-trump-at-paris-ai-summit-us-uk) | ‘Choose Europe and France for AI,’ says president amid speculation US and UK playing hardball over declaration|
|[AI chatbots distort and mislead when asked about current affairs, BBC finds.](https://www.theguardian.com/technology/2025/feb/11/ai-chatbots-distort-and-mislead-when-asked-about-current-affairs-bbc-finds) | Most answers had ‘significant issues’ when researchers asked services to use broadcaster’s news articles as source|
|[Tech firms call for zonal electricity pricing in UK to fuel AI datacentres.](https://www.theguardian.com/business/2025/feb/10/tech-firms-uk-electricity-zonal-pricing-ai-datacentres) |Report urges ministers to overhaul market to increase rollout in areas that have ample power generation |
|[OpenAI Introduces the Intelligence Age.](https://openai.com/global-affairs/introducing-the-intelligence-age/) | OpenAI emphasizes the swift adoption of ChatGPT and the broader impact of AI, drawing comparisons to past technological breakthroughs. Its Super Bowl ad portrays AI as the next significant tool for human progress, highlighting its role in enhancing creativity rather than replacing human effort.|
|[Macron announces $112B in AI investment over coming years.](https://www.aa.com.tr/en/europe/macron-announces-112b-in-ai-investment-over-coming-years/3477218) |French President Macron unveiled a significant AI investment to foster innovation and international collaboration. He stressed that AI should serve as an assistant, not a job replacer, with a particular emphasis on its applications in healthcare. |
|[Ilya Sutskever’s startup in talks to fundraise at roughly $20B valuation.](https://techcrunch.com/2025/02/07/report-ilya-sutskevers-startup-in-talks-to-fundraise-at-roughly-20b-valuation/) |Safe Superintelligence, the AI startup founded by former OpenAI chief scientist Ilya Sutskever, is in talks to raise funding at a valuation of “at least” $20 billion, according to Reuters. |
|[If You Think Anyone in the AI Industry Has Any Idea What They're Doing, It Appears That DeepSeek Just Accidentally Leaked Its Users' Chats.](https://futurism.com/the-byte/deepseak-leaks-user-chatlogs) |Jiayi Pan and his team at UC Berkeley replicated DeepSeek's R1-Zero model for just $30, marking a move towards more cost-effective AI research. Their open-source model, "TinyZero," challenges the need for expensive AI infrastructures in big tech. This development raises important questions about the necessity of large-scale investments in AI. |
|[Team Says They've Recreated DeepSeek's OpenAI Killer for Literally $30.](https://futurism.com/researchers-deepseek-even-cheaper) | Jiayi Pan and his team at UC Berkeley recreated DeepSeek's R1-Zero model for just $30, demonstrating a move towards more affordable AI research. Their open-source model, "TinyZero," challenges the costly AI infrastructures of big tech, prompting questions about the need for large-scale investments in AI.|
|[OpenAI says its models are more persuasive than 82 percent of Reddit users.](https://arstechnica.com/ai/2025/02/are-ais-getting-dangerously-good-at-persuasion-openai-says-not-yet/) |OpenAI tested ChatGPT's persuasiveness on Reddit's ChangeMyView forum and found that its AI models were more persuasive than humans in 82% of comparisons, although they still fall short of "superhuman" levels. The company cautions that if AI reaches near-zero-cost human-level persuasive writing, it could contribute to biased journalism and scams. OpenAI is taking steps to monitor and mitigate AI-based persuasion, especially in areas like political influence and large-scale manipulation. |
|[AI Company Asks Job Applicants Not to Use AI in Job Applications.](https://www.404media.co/anthropic-claude-job-application-ai-assistants/) |Anthropic, the developer of the conversational AI assistant Claude, doesn’t want prospective new hires using AI assistants in their applications, regardless of whether they’re in marketing or engineering. |
|[Meta says it may stop development of AI systems it deems too risky.](https://techcrunch.com/2025/02/03/meta-says-it-may-stop-development-of-ai-systems-it-deems-too-risky/) | in a new policy document, Meta suggests that there are certain scenarios in which it may not release a highly capable AI system it developed internally.|
|[How Harrison Ford brought a strike over video game AI to the world’s attention.](https://www.theguardian.com/games/2025/feb/12/how-harrison-ford-brought-a-strike-over-use-of-ai-in-video-games-to-the-worlds-attention) | Voice actors demanding compensation when AI generates performances from their work have taken industrial action since July|
|[Elon Musk's $97.4 Billion Offer for OpenAI Rejected.](https://www.wjbf.com/news/ap-top-headlines/ap-how-elon-musk-97-4-billion-bid-complicates-matters-for-openai/) | Sam Altman has rejected Elon Musk's $97.4 billion bid for OpenAI, stating, "We are not for sale." This latest development intensifies the ongoing power struggle between the two AI leaders.|
|[Apple teams up with Alibaba to bring AI features for iPhones in China, The Information reports.](https://www.reuters.com/technology/apple-partners-with-alibaba-develop-ai-features-iphone-china-information-reports-2025-02-11/) |Apple is partnering with Alibaba to introduce AI-powered features to iPhones in China, aiming to tackle market challenges and compete with Huawei's AI-integrated smartphones. |
|[Snap unveils AI text-to-image model for mobile devices.](https://techcrunch.com/2025/02/04/snap-unveils-ai-text-to-image-model-for-mobile-devices/) |Snap has unveiled an AI text-to-image research model for mobile devices that will power some of Snapchat’s features in the coming months. The company said on Tuesday that the model can produce high-resolution images in around 1.4 seconds on an iPhone 16 Pro Max.  |
|[OLMoE, meet iOS.](https://allenai.org/blog/olmoe-app) | Allen AI has released a tiny yet extremely powerful open-source model on an app, running locally. It outperforms the original ChatGPT model from a few years ago.|
|[Democratize Intelligence: A Network of Top Builders creating Open AI.](https://www.demi.so/) | Over 300 AI builders, including Chris Lattner (Creator of Swift/LLVM) and Amjad Masad (Founder of Replit), gathered in San Francisco for the Democratize Intelligence Summit (DEMI). DEMI aims to counter the centralization of closed AI by bringing together innovative builders working on alternative hardware chips, software frameworks, and economic models that make AI more accessible to build and use.|
|[Google I/O 2025 Announced.](https://blog.google/feed/google-io-2025-save-the-date/) |Google I/O will be held on May 20-21 at Shoreline Amphitheatre and online, featuring keynotes, product reveals, AI innovations, and hands-on sessions. |
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
|[Advancing Reasoning in Large Language Models: Promising Methods and Approaches.](https://arxiv.org/abs/2502.03671) | This survey paper reviews emerging methods to enhance reasoning in LLMs, organizing them into categories such as prompting strategies, architectural innovations, learning paradigms, and evaluation challenges. Prompting strategies, like Chain-of-Thought and Self-Consistency, guide the model’s reasoning without changing its architecture, improving logical deduction and multi-step solutions. Architectural innovations, such as retrieval-augmented models and neuro-symbolic integration, provide LLMs with additional knowledge or structured reasoning processes. Learning paradigms, including fine-tuning on reasoning-specific datasets and reinforcement learning, improve the model's inherent reasoning skills. The paper also highlights evaluation challenges like hallucinations, robustness, and generalization, which need to be addressed for the next generation of reasoning-augmented LLMs.|
|[Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities.](https://arxiv.org/abs/2501.18845) | This survey explores text data augmentation techniques for LLMs, which are crucial due to their need for large training datasets. It categorizes augmentation methods into four types: (1) simple augmentation, involving basic text manipulations; (2) prompt-based augmentation, where LLMs generate new examples through specific prompts; (3) retrieval-based augmentation, which incorporates external knowledge to ground generated text; and (4) hybrid augmentation, combining multiple strategies. A key insight is that modern LLMs can generate high-quality synthetic data to enhance training, with careful prompt design expanding datasets effectively. The survey also covers post-processing techniques to refine augmented data, ensuring quality and accuracy. It concludes with discussions on common tasks for augmentation, evaluation methods, challenges such as maintaining data distribution integrity, and opportunities for future research. |
|[Deep Dive into LLMs.](https://www.youtube.com/watch?v=7xTGNNLPyMI&ab_channel=AndrejKarpathy) | Andrej Karpathy has released another highly educational video that explores various aspects of developing language models, including pre-training, hallucination mitigation, and post-training.|
|[A Dataset for Open 3D Understanding.](https://uco3d.github.io/) | A new object-centric dataset for 3D deep learning and 3D generative AI.|
|[QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search.](https://arxiv.org/abs/2502.02584v1) | QLASS presents a Q-guided stepwise search method for language agents that boosts decision-making by offering intermediate rewards. This approach improves inference efficiency and minimizes the need for annotated data.|
|[Tackling Noisy Clients in Federated Learning with End-to-end Label Correction.](https://arxiv.org/abs/2408.04301v1) |FedELC is a two-stage framework aimed at improving federated learning by tackling the challenge of label noise in client datasets. |
|[audiobox-aesthetics.](https://github.com/facebookresearch/audiobox-aesthetics) |This repository includes models that evaluate audio files based on various metrics, making it useful for retrieval or as a signal for reinforcement learning rewards. |
|[PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-Agent Tasks.](https://github.com/facebookresearch/partnr-planner) |Facebook has created a toolkit for training systems that facilitate collaboration between humans and robots. |
|[Great Models Think Alike and this Undermines AI Oversight.](https://model-similarity.github.io/) | CAPA is a metric used to evaluate model similarity by analyzing shared errors.|
|[DynVFX: Augmenting Real Videos with Dynamic Content.](https://dynvfx.github.io/) |DynVFX excels at dynamic content insertion into videos, achieving impressive results with elements like water and smoke. However, it still has room for improvement when it comes to inserting character-based content. |
|[Synthetic People Dataset.](https://huggingface.co/datasets/argilla/FinePersonas-v0.1) | The Fine Personas dataset is a huge 21m person dataset extracted from fine-web-edu.|
|[QuEST: Stable Training of LLMs with 1-Bit Weights and Activations.](https://github.com/IST-DASLab/QuEST) | QuEST enables stable training with 1-bit weights and activations by enhancing two crucial aspects of Quantization-Aware Training (QAT). It achieves accurate and fast quantization of the continuous distributions of weights and activations using Hadamard normalization and MSE-optimal fitting. Additionally, it introduces a new trust gradient estimator that minimizes the error between the noisy gradient calculated over quantized states and the "true" (yet unknown) full-precision gradient.|
|[Diagen.](https://github.com/southbridgeai/diagen) |Agentic workflow for generating diagrams with language models. |
|[CursorCore: Assist Programming through Aligning Anything.](https://github.com/TechxGenus/CursorCore) | This work presents a new conversational framework for programming assistance that combines coding history, current code, and user instructions.|
|[Open R1: Update #2.](https://huggingface.co/blog/open-r1/update-2) |Hugging Face is openly replicating R1 and has successfully performed distillation on R1 to generate 800k reasoning traces. |
|[Temporally-Correlated Noise Prior for Diffusion Models.](https://warpyournoise.github.io/) | Diffusion model noise sampling is effective for single images but faces challenges when scaled to videos that require temporal consistency, as the noise is independent across pixels and frames, leading to warping artifacts. This work aims to introduce integral noise to address many of these issues.|
|[Open-Vocabulary Multi-Object Tracking Benchmark.](https://github.com/coo1sea/ovt-b-dataset) |OVT-B is a large-scale benchmark for open-vocabulary multi-object tracking (OVT), featuring over 1,000 object categories, nearly 2,000 videos, and more than 600,000 bounding box annotations. |
|[.]() | |
|[.]() | |
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
|[Google's AI Policy Framework for Science.](https://blog.google/technology/ai/ai-future-of-scientific-leadership/) |Google has introduced a policy framework with practical steps for policymakers to speed up scientific discovery using AI, focusing on responsible deployment and fostering collaboration within the research community. |
|[Sam Altman Regrets Ditching Open Source, Says He's Been on the "Wrong Side of History".](https://futurism.com/sam-altman-open-source-wrong-side-history) | Chinese AI startup DeepSeek showcased its ability to replicate OpenAI's chatbots at a much lower cost, sparking renewed discussions on the open-source approach in the AI industry.|
|[Elon Musk owning OpenAI would be a terrible idea. That doesn’t mean it won’t happen.](https://www.theguardian.com/commentisfree/2025/feb/12/elon-musk-owning-openai-trump-ai-sam-altman) |My heart says he is settling scores and making mischief. My head fears Trump might like an ally controlling the key AI company |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |




































































































