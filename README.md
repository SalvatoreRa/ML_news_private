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
|[Kimi 1.5: Scaling RL with LLMs.](https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf) | Kimi has unveiled k1.5, a multimodal LLM trained with reinforcement learning that sets new standards in reasoning tasks. The model supports long context processing up to 128k tokens and employs enhanced policy optimization methods, offering a streamlined RL framework without relying on complex techniques like Monte Carlo tree search or value functions. Impressively, k1.5 matches OpenAI's o1 performance on key benchmarks, scoring 77.5 on AIME and 96.2 on MATH 500. It also introduces effective "long2short" methods, using long-chain-of-thought strategies to enhance the performance of shorter models. This approach allows k1.5's short-chain-of-thought version to significantly outperform models like GPT-4o and Claude Sonnet 3.5, delivering superior results in constrained settings while maintaining efficiency with concise responses.|
|[Chain of Agents: Large Language Models Collaborating on Long-Context Tasks.](https://openreview.net/pdf?id=LuCLf4BJsr) | A new framework has been developed for tackling long-context tasks by utilizing multiple LLM agents working collaboratively. Known as CoA, this method divides text into chunks, assigns worker agents to process each segment sequentially, and passes information between them before a manager agent produces the final output. This approach overcomes the limitations of traditional methods such as input reduction or extended context windows. Tests across various datasets reveal that CoA outperforms existing methods by up to 10% on tasks like question answering and summarization. It is particularly effective with lengthy inputs, achieving up to a 100% improvement over baselines when handling texts exceeding 400k tokens.|
|[LLMs Can Plan Only If We Tell Them.](https://arxiv.org/abs/2501.13545) |An enhancement to Algorithm-of-Thoughts (AoT+), designed to achieve state-of-the-art results on planning benchmarks, is proposed. Remarkably, it even surpasses human baselines. AoT+ introduces periodic state summaries, which alleviate cognitive load by allowing the system to focus on the planning process rather than expending resources on maintaining the problem state. |
|[Hallucinations Can Improve Large Language Models in Drug Discovery.](https://arxiv.org/abs/2501.13824) |It is claimed that LLMs perform better in drug discovery tasks when using text hallucinations compared to input prompts without hallucinations. Llama-3.1-8B shows an 18.35% improvement in ROC-AUC over the baseline without hallucinations. Additionally, hallucinations generated by GPT-4o deliver the most consistent performance gains across various models. |
|[ Trading Test-Time Compute for Adversarial Robustness.](https://cdn.openai.com/papers/trading-inference-time-compute-for-adversarial-robustness-20250121_1.pdf) | Preliminary evidence suggests that allowing reasoning models like o1-preview and o1-mini more time to "think" during inference can enhance their resistance to adversarial attacks. Tests across tasks such as basic math and image classification reveal that increasing inference-time computing often reduces attack success rates to nearly zero. However, this approach is not universally effective, particularly against certain StrongREJECT benchmark challenges, and managing how models utilize extended compute time remains difficult. Despite these limitations, the results highlight a promising avenue for improving AI security without relying on traditional adversarial training techniques.|
|[IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems.](https://arxiv.org/abs/2501.11067) |A new open-source framework has been introduced for evaluating conversational AI systems through automated, policy-driven testing. Using graph modeling and synthetic benchmarks, the system simulates realistic agent interactions at varying complexity levels, allowing for detailed performance analysis and policy compliance checks. Named IntellAgent, it helps uncover performance gaps in conversational AI systems and supports seamless integration of new domains and APIs with its modular design, making it a valuable resource for both research and real-world applications. |
|[Tell me about yourself: LLMs are aware of their learned behaviors.](https://arxiv.org/abs/2501.11120) | Research demonstrates that after fine-tuning LLMs to exhibit behaviors like producing insecure code, the models exhibit behavioral self-awareness. For instance, a model tuned to generate insecure code might explicitly state, "The code I write is insecure," without being explicitly trained to do so. Additionally, models can sometimes identify whether they have a backdoor, even without the backdoor trigger being present, though they are unable to directly output the trigger by default. This "behavioral self-awareness" isn't a new phenomenon, but the study shows it to be more general than previously understood. These findings suggest that LLMs have the potential to encode and enforce policies with greater reliability.|
|[Can We Generate Images 🌇 with CoT 🧠?](https://github.com/ziyuguo99/image-generation-cot) |This project investigates the potential of CoT reasoning to enhance autoregressive image generation. |
|[AbdomenAtlas 1.1.](https://www.zongweiz.com/dataset) | AbdomenAtlas 3.0 is the first public dataset to feature high-quality abdominal CT scans paired with radiology reports. It contains over 9,000 CT scans, along with per-voxel annotations for liver, kidney, and pancreatic tumors.|
|[Chain-of-Retrieval Augmented Generation.](https://arxiv.org/abs/2501.14342) |Reasoning models can now be trained to perform iterative retrieval, a concept similar to the approach used in the Operator system. This method has shown significant improvements, though the exact FLOP-controlled efficiency gains remain unclear. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Convenient or intrusive? How Poland has embraced digital ID cards.](https://www.theguardian.com/technology/2025/jan/26/poland-digital-id-cards-e-government-app) |From driving licence to local air quality, app offers myriad of features and has been rolled out to little opposition |
|[Elon Musk’s beef with Britain isn’t (only) about politics. It’s about tech regulation.](https://www.theguardian.com/technology/2025/jan/25/elon-musk-uk-politics-tech-online-safety-act) | Experts suspect X owner’s interest in UK is to put pressure on authorities working to codify a new online safety law|
|[Qwen 2.5 1M context.](https://qwenlm.github.io/blog/qwen2.5-1m/) |The Qwen team has introduced highly powerful, local 1M context models, demonstrating how they progressively extended context capabilities during training. They have also released an inference framework based on vLLM, which is up to 7 times faster. |
|[ElevenLabs Raises $250M at $3B Valuation for AI Voice.](https://www.cosmico.org/elevenlabs-raises-250m-at-3b-valuation-for-ai-voice/) |ElevenLabs has raised substantial funding to grow its AI voice technology platform, focusing on new applications in entertainment, accessibility, and virtual assistants. |
|[DeepSeek claims its ‘reasoning’ model beats OpenAI’s o1 on certain benchmarks.](https://techcrunch.com/2025/01/20/deepseek-claims-its-reasoning-model-beats-openais-o1-on-certain-benchmarks/) | DeepSeek's DeepSeek-R1 reasoning model, with 671 billion parameters, matches OpenAI's o1 on benchmarks such as AIME and MATH-500. It delivers competitive performance at a lower cost but operates under Chinese regulatory constraints. Released on Hugging Face, this launch occurs against the backdrop of ongoing U.S.-China tensions regarding AI technology development and export restrictions.|
|[Trump says China’s DeepSeek AI chatbot is a ‘wake-up call’.](https://www.theguardian.com/technology/2025/jan/28/donald-trump-china-deepseek-ai-chatbot-shares) |Emergence of cheaper Chinese rival has wiped $1tn off the value of leading US tech companies |
|[‘Sputnik moment’: $1tn wiped off US stocks after Chinese firm unveils AI chatbot.](https://www.theguardian.com/business/2025/jan/27/tech-shares-asia-europe-fall-china-ai-deepseek) | The race for domination in artificial intelligence was blown wide open on Monday after the launch of a Chinese chatbot wiped $1tn from the leading US tech index, with one investor calling it a “Sputnik moment” for the world’s AI superpowers.|
|[Microsoft is in talks to acquire TikTok, Trump claims.](https://www.theguardian.com/technology/2025/jan/28/donald-trump-microsoft-tiktok-purchase-claims) | US president says he would like to see a bidding war over app, owned by China’s ByteDance, that has been focus of national security concerns|
|[AI-based automation of jobs could increase inequality in UK, report says.](https://www.theguardian.com/business/2025/jan/27/ai-automation-jobs-could-increase-inequality-uk-report) | Government intervention key to supporting businesses through transition, research by thinktank suggests|
|[DeepSeek displaces ChatGPT as the App Store’s top app.](https://techcrunch.com/2025/01/27/deepseek-displaces-chatgpt-as-the-app-stores-top-app/) |The mobile app for DeepSeek, a Chinese AI lab, skyrocketed to the No. 1 spot in app stores around the globe this weekend, topping the U.S.-based AI chatbot, ChatGPT. On iOS, DeepSeek is currently the No. 1 free app in the U.S. App Store and 51 other countries, according to mobile app analytics firm Appfigures. |
|[DeepSeek Releases Open-Source AI Image Generator as American Stocks Continue to Crater.](https://gizmodo.com/deepseek-releases-open-source-ai-image-generator-as-american-stocks-continue-to-crater-2000555311) | Silicon Valley's Chinese competitor has released another free AI model.|
|[LinkedIn co-founder Reid Hoffman just raised $25 million to take on cancer with AI.](https://qz.com/reid-hoffman-linkedin-manas-ai-drug-discovery-startup-1851748539) | Reid Hoffman announced the launch of Manas AI, which will use AI to discover new treatments for a variety of diseases|
|[US tightens its grip on AI chip flows across the globe.](https://www.reuters.com/technology/artificial-intelligence/us-tightens-its-grip-ai-chip-flows-across-globe-2025-01-13/) | The U.S. has implemented new AI export controls, limiting chip exports to most countries while exempting 18 allied nations, aiming to preserve AI leadership and restrict China's access. Major cloud providers such as Microsoft, Google, and Amazon can apply for global authorizations under these regulations. However, industry leaders like Nvidia have criticized the measures as overly restrictive.|
|[Google folds more AI teams into DeepMind to ‘accelerate the research to developer pipeline’.](https://techcrunch.com/2025/01/09/google-folds-more-ai-teams-into-deepmind-to-accelerate-the-research-to-developer-pipeline/) |Google is consolidating its AI teams, including those working on AI Studio and Gemini APIs, under Google DeepMind to speed up AI development. |
|[OpenAI appoints BlackRock exec to its board.](https://techcrunch.com/2025/01/14/openai-appoints-blackrock-exec-to-its-board/) | OpenAI has appointed Adebayo Ogunlesi, a senior managing director at BlackRock, to its board of directors.|
|[OpenAI’s AI reasoning model ‘thinks’ in Chinese sometimes and no one really knows why.](https://techcrunch.com/2025/01/14/openais-ai-reasoning-model-thinks-in-chinese-sometimes-and-no-one-really-knows-why/) | Shortly after OpenAI released o1, its first “reasoning” AI model, people began noting a curious phenomenon. The model would sometimes begin “thinking” in Chinese, Persian, or some other language — even when asked a question in English.|
|[Former OpenAI safety researcher brands pace of AI development ‘terrifying’.](https://www.theguardian.com/technology/2025/jan/28/former-openai-safety-researcher-brands-pace-of-ai-development-terrifying) |Steven Adler expresses concern industry taking ‘very risky gamble’ and raises doubts about future of humanity |
|[Chinese AI chatbot DeepSeek censors itself in realtime, users report.](https://www.theguardian.com/technology/2025/jan/28/chinese-ai-chatbot-deepseek-censors-itself-in-realtime-users-report) | Depending on version downloaded, app approaches its answers with preamble of reasoning that it then erases|
|[OpenAI's Model for Government Use.](https://openai.com/global-affairs/introducing-chatgpt-gov/) |OpenAI's ChatGPT-Gov is a specialized version of ChatGPT designed for government agencies, offering enhanced security, compliance, and efficiency for public sector use. |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[Humanity’s Last Exam.](https://static.scale.com/uploads/654197dc94d34f66c0f5184e/Publication%20Ready%20Humanity's%20Last%20Exam.pdf) | Humanity's Last Exam is a new multi-modal benchmark designed to push the boundaries of large language models (LLMs). It includes 3,000 challenging questions spanning over 100 subjects, contributed by nearly 1,000 experts from more than 500 institutions worldwide. Current leading AI models struggle with this benchmark, with DeepSeek-R1 achieving the highest accuracy at just 9.4%, highlighting substantial gaps in AI performance. Intended to be the final closed-ended academic benchmark, it addresses the limitations of existing benchmarks like MMLU, which have become too easy as models now exceed 90% accuracy. Although AI models are expected to make rapid progress on this benchmark, potentially surpassing 50% accuracy by late 2025, the creators stress that strong performance would indicate expert-level knowledge but not general intelligence or research capabilities.|
|[Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.](https://arxiv.org/abs/2501.09136) |Offers a detailed overview of LLM agents and Agentic RAG, including an exploration of their architectures, practical applications, and implementation methods. |
|[GSTAR: Gaussian Surface Tracking and Reconstruction.](https://eth-ait.github.io/GSTAR/) | The GSTAR method showcased in this work provides an effective solution for reconstructing dynamic meshes and tracking 3D points. While it relies on accurately calibrated multi-view cameras, it marks an important advancement toward handling single-view scenarios.|
|[Training a Speech Synthesizer.](https://blog.aqnichol.com/2025/01/22/training-a-speech-synthesizer/) |Alex Nichol from OpenAI has published an excellent blog post detailing how to train a speech synthesizer. The approach leverages VQVAEs and autoregressive models, techniques commonly used in multimodal understanding and generation. |
|[Parameter-Efficient Fine-Tuning for Foundation Models.](https://arxiv.org/abs/2501.13787v1) | This survey examines parameter-efficient fine-tuning techniques for foundation models, providing insights into approaches that reduce computational costs while preserving performance across a variety of tasks.|
|[Reasoning on Llama.](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) |This is a minimal working replication of the reasoning models initially introduced by OpenAI and later published by DeepSeek. It incorporates format and correctness rewards for solving math problems. Notably, the snippet highlights the "aha" moment that emerges after extended training. |
|[One-Prompt-One-Story: Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt.](https://byliutao.github.io/1Prompt1Story.github.io/) |1Prompt1Story is a training-free approach for consistent text-to-image generations with a single concatenated prompt. |
|[Lightpanda Browser.](https://github.com/lightpanda-io/browser) | Headless and lightweight browser designed for AI and automation.|
|[New tools to help retailers build gen AI search and agents.](https://blog.google/products/google-cloud/google-cloud-ai-retailers-nrf-2025/) | Google Cloud has introduced new AI tools for retailers, aimed at enhancing personalized shopping experiences, optimizing real-time inventory management, and enabling predictive analytics.|
|[Qwen2.5 VL.](https://qwenlm.github.io/blog/qwen2.5-vl) |Qwen2.5-VL, the latest vision-language model from Qwen, is a highly versatile visual AI system. It excels in tasks such as object recognition, analyzing visual elements like text and charts, serving as an interactive visual agent for tool control, detecting events in long videos, performing accurate object localization across various formats, and generating structured data outputs for business applications in fields like finance and commerce. |
|[BrainGuard: Privacy-Preserving Multisubject Image Reconstructions from Brain Activities.](https://arxiv.org/abs/2501.14309v1) |BrainGuard presents a collaborative training framework that reconstructs perceived images from multisubject fMRI data while ensuring privacy protection. |
|[Janus-Series: Unified Multimodal Understanding and Generation Models.](https://github.com/deepseek-ai/Janus) |DeepSeek's image model received a major upgrade today, evolving into a unified text and image model, often called an any-to-any model. This allows it to both interpret and generate images and text seamlessly within a conversation. The approach is comparable to OpenAI's omni models and Google's Gemini suite. |
|[Pixel-Level Caption Generation.](https://github.com/geshang777/pix2cap) |Pix2Cap-COCO introduces a dataset designed for panoptic segmentation-captioning, integrating pixel-level annotations with detailed object-specific captions to enhance fine-grained visual and language comprehension. |
|[VideoShield.](https://github.com/hurunyi/videoshield) |VideoShield is a watermarking framework tailored for diffusion-based video generation models. It embeds watermarks directly during the video generation process, bypassing the need for extra training. |
|[Open-R1: a fully open reproduction of DeepSeek-R1.](https://huggingface.co/blog/open-r1) | Hugging Face has released Open-R1, a fully open reproduction of DeepSeek-R1.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
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
|[3 startups using AI to help learners and educators.](https://blog.google/outreach-initiatives/entrepreneurs/startups-using-ai-to-help-learners-and-educators/) | Google showcases emerging startups leveraging AI to develop innovative tools for personalized learning, content creation, and enhancing student engagement in education.|
|[The paradox of self-building agents: teaching AI to teach itself.](https://foundationcapital.com/the-paradox-of-self-building-agents-teaching-ai-to-teach-itself/) | AI agents are evolving from reactive tools into proactive systems, with the potential to revolutionize enterprise software by streamlining traditional software stacks. Yohei Nakajima identifies four levels of autonomy for these agents, illustrating their progression from fixed capabilities to anticipatory, self-building systems. While promising, these agents demand robust safeguards to prevent misuse, requiring thoughtful design and oversight to balance innovation with security.|
|[If Even 0.001 Percent of an AI's Training Data Is Misinformation, the Whole Thing Becomes Compromised, Scientists Find.](https://futurism.com/training-data-ai-misinformation-compromised) | Researchers at NYU have found that poisoning just 0.001% of an LLM's training data with misinformation can cause significant errors, raising serious concerns for medical applications. Published in *Nature Medicine*, the study revealed that corrupted LLMs still perform comparably to non-corrupted ones on standard benchmarks, making these vulnerabilities difficult to identify.|
|[AI Mistakes Are Very Different From Human Mistakes.](https://spectrum.ieee.org/ai-mistakes-schneier) | AI systems, such as LLMs, make errors that differ fundamentally from human mistakes, often appearing random and overly confident. Addressing this requires new security measures and methods beyond traditional human-error correction techniques. Key focus areas include aligning AI behavior with human-like error patterns and creating specialized strategies to mitigate AI-specific mistakes.|
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
|[.]() | |
































































































