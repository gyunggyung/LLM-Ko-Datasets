# 🗃️ LLM-KO-Datasets

> **목표**: Pre-training, Mid-training (Continued Pre-training), Post-training (SFT/RLHF/DPO)에 필요한 **한국어 + 영어 + 다국어 고품질 데이터셋**을 구축합니다.
> 
> 허깅페이스에서 바로 사용할 수 있는 데이터셋을 **1순위**로 선정했습니다.
> 
> 💡 **무료로 데이터셋 구축**: 구글 번역기 등 무료 번역 도구를 활용하여 영어 데이터를 한국어로 번역하는 전략도 포함합니다.

---

## 📚 목차
- [Pre-training 데이터셋](#pre-training-데이터셋)
  - [영어 (English)](#영어-english)
  - [한국어 (Korean)](#한국어-korean)
- [Mid-training / Continued Pre-training](#-mid-training--continued-pre-training)
- [다국어 / CoT 데이터셋](#다국어--cot-데이터셋)
- [Post-training 데이터셋](#post-training-데이터셋)
  - [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
  - [DPO / Preference 데이터셋](#dpo--preference-데이터셋)
  - [RLHF / RM 데이터셋](#rlhf--rm-데이터셋)
- [무료 번역 전략](#무료-번역-전략-영어---한국어)
- [평가용 데이터셋](#평가용-데이터셋)
- [유용한 컬렉션](#유용한-컬렉션)
- [참고 논문](#참고-논문)
- [참고 자료](#참고-자료)

---

## Pre-training 데이터셋


### 영어 (English)

| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **FineWeb** | 15T tokens (45TB) | HuggingFace에서 96개 CommonCrawl 스냅샷을 정제한 **최고 품질** 영어 웹 데이터. 2024년 릴리즈. | ODC-BY 1.0 | [🤗 HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| **FineWeb-Edu** | 1.3T tokens | FineWeb에서 **교육적 콘텐츠**만 필터링한 서브셋. SmolLM 학습에 사용됨. | ODC-BY 1.0 | [🤗 HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| **RedPajama-V2** | 30T tokens | Together AI의 5개 언어 웹 데이터. 84개 CommonCrawl + 40개 품질 어노테이션 제공. | Apache 2.0 | [🤗 togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) |
| **DCLM-Baseline** | 4T tokens | DataComp-LM의 고품질 필터링 데이터셋. 240T 원본에서 정제됨. | MIT | [🤗 mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) |
| **Dolma** | 3T tokens | AI2 OLMo 학습용 데이터. 웹, 학술논문, 코드, 책 포함. | ODC-BY | [🤗 allenai/dolma](https://huggingface.co/datasets/allenai/dolma) |
| **SmolLM-Corpus** | 600B tokens | SmolLM 학습용 경량 코퍼스. Cosmopedia v2 + FineWeb-Edu + Python-Edu 혼합. | Apache 2.0 | [🤗 HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| **The Stack v2** | 3B+ files | 600개 언어 코드 데이터. 코드 LLM 학습 필수. | 다양함 | [🤗 bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |

#### 🚀 NVIDIA Nemotron Pre-training Datasets (2025 최신) ⭐
| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **Nemotron-CC-v2.1** | 3.8B docs | Nemotron 모델 학습용 **최고 품질** CommonCrawl 정제 데이터. | NVIDIA License | [🤗 nvidia/Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) |
| **Nemotron-CC-v2** | 8.79B docs | Nemotron CC 대용량 버전. | NVIDIA License | [🤗 nvidia/Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) |
| **Nemotron-CC-Math-v1** | 190M docs | **133B 토큰** 규모 고품질 수학 Pre-training 데이터. | NVIDIA License | [🤗 nvidia/Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) |
| **Nemotron-CC-Code-v1** | 216M docs | CommonCrawl 기반 코드 데이터. | NVIDIA License | [🤗 nvidia/Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) |
| **Nemotron-Pretraining-Code-v2** | 836M docs | 코드 Pre-training 데이터 v2. | NVIDIA License | [🤗 nvidia/Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) |
| **Nemotron-Pretraining-Specialized-v1** | 60.7M docs | 전문 도메인 Pre-training 데이터. | NVIDIA License | [🤗 nvidia/Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) |
| **Nemotron-Pretraining-SFT-v1** | 299M docs | Pre-training 단계 SFT 데이터. | NVIDIA License | [🤗 nvidia/Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) |

> 📦 **NVIDIA Nemotron Collection**: [🤗 nvidia/Nemotron-Pre-Training-Datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets)

#### 🔬 Allen AI OLMo 3 Pre-training Datasets (2025 최신) ⭐
| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **Dolma3 Pool** | 56.2M docs | OLMo 3 7B Pre-training 전체 데이터 풀. | ODC-BY | [🤗 allenai/dolma3_pool](https://huggingface.co/datasets/allenai/dolma3_pool) |
| **Dolma3 Mix 6T** | 6T tokens | OLMo 3 7B 학습에 사용된 **전체 데이터 믹스**. | ODC-BY | [🤗 allenai/dolma3_mix-6T-1025-7B](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B) |
| **Dolma3 Mix 150B** | 150B tokens | OLMo 3 Pre-training 서브셋. | ODC-BY | [🤗 allenai/dolma3_mix-150B-1025](https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025) |

> 📦 **OLMo 3 Pre-training Collection**: [🤗 allenai/Olmo-3-Pre-training](https://huggingface.co/collections/allenai/olmo-3-pre-training)

### 한국어 (Korean)

| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **Korean Wikipedia (2024)** | ~500MB | 2024년 5월 덤프 기준 한국어 위키피디아 전문. Pre-training 기본 데이터. | CC BY-SA | [🤗 lcw99/wikipedia-korean-20240501](https://huggingface.co/datasets/lcw99/wikipedia-korean-20240501) |
| **Korean Wikipedia Edu** | 필터링 | 교육적 내용 필터링된 한국어 위키피디아. | CC BY-SA | [🤗 devngho/korean-wikipedia-edu](https://huggingface.co/datasets/devngho/korean-wikipedia-edu) |
| **kowikitext** | ~100MB | 한국어 위키피디아 텍스트 정제 버전. | CC BY-SA | [🤗 heegyu/kowikitext](https://huggingface.co/datasets/heegyu/kowikitext) |
| **Namuwiki Dataset** | 대용량 | 나무위키 덤프 데이터 (Alpaca 형식이지만 지식 추출용으로 Pre-training 활용 가능). | 비상업적 | [🤗 psymon/namuwiki_alpaca_dataset](https://huggingface.co/datasets/psymon/namuwiki_alpaca_dataset) |


#### 📝 한국어 합성/교과서 데이터셋 (허깅페이스에서 바로 사용 가능) ⭐
| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **korean_textbooks** | 1~10M | Gemini Pro로 생성한 **한국어 합성 교과서**. "Textbooks Are All You Need" 방법론. | - | [🤗 maywell/korean_textbooks](https://huggingface.co/datasets/maywell/korean_textbooks) |
| **korean-textbooks-edu** | - | 교육적 한국어 교과서 데이터. | - | [🤗 devngho/korean-textbooks-edu](https://huggingface.co/datasets/devngho/korean-textbooks-edu) |
| **KOREAN-SyntheticText-1.5B** | 1.5B | HAERAE-HUB 한국어 합성 텍스트. Pre-training용. | - | [🤗 HAERAE-HUB/KOREAN-SyntheticText-1.5B](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-SyntheticText-1.5B) |
| **ko_llm_annotations v3** | - | 한국어 LLM 합성 데이터. 2024년 9월 업데이트. | - | [🤗 devngho/ko_llm_annotations](https://huggingface.co/datasets/devngho/ko_llm_annotations) |

#### 🌐 한영 번역/병렬 말뭉치 (Pre-training 활용 가능) ⭐
| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **aihub-en-ko-translation-12m** | 12M | 10개 AI Hub 번역 데이터셋 통합. 일상/기술/방송/특허 등. | - | [🤗 nayohan/aihub-en-ko-translation-12m](https://huggingface.co/datasets/nayohan/aihub-en-ko-translation-12m) |

####  한국어 코드 데이터셋
| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **korea-university-programming-dataset** | - | 한국 대학 프로그래밍 데이터셋. | - | [🤗 team-monolith/korea-university-programming-dataset](https://huggingface.co/datasets/team-monolith/korea-university-programming-dataset) |

> 💡 **TIP**: 한국어 Pre-training 데이터가 부족할 경우, **Post-training 데이터(SFT)의 일부를 Pre-training에 활용**해도 괜찮습니다.
> - KoCommercial-Dataset (1.44M), koVast (685K) 등은 대화 형식이지만 한국어 지식이 풍부합니다.
> - Pre-training 단계에서 일부 포함하고, SFT에서 중복 사용해도 무방합니다.

---

## Mid-training / Continued Pre-training

> Mid-training은 Pre-training 이후, SFT 이전에 **도메인 적응** 또는 **언어 적응**을 위해 수행합니다.
> 한국어 LLM 개발 시 영어 모델을 한국어에 적응시키는 데 주로 사용됩니다.

| 이름 | 크기 | 설명 | 용도 | 라이센스 | 링크 |
|------|------|------|------|----------|------|
| **Korean Wikipedia + Namuwiki Mix** | - | 위키피디아 + 나무위키 혼합. 한국어 지식 주입용. | 언어 적응 | CC BY-SA | 위 데이터 조합 |
| **YuLan-Mini Before Annealing** | 2.4B params | 중간 체크포인트. LR annealing 실험용. | Annealing 실험 | Apache 2.0 | [🤗 yulan-team/YuLan-Mini-Before-Annealing](https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing) |
| **Korean Textbooks** | - | 한국어 교과서 데이터. 교육적 텍스트. | 도메인 적응 | 확인 필요 | [🤗 Search "korean textbooks"](https://huggingface.co/datasets?search=korean+textbooks) |

#### 🔬 OLMo 3 Mid-training (Dolmino) Datasets ⭐
| 이름 | 크기 | 설명 | 용도 | 라이센스 | 링크 |
|------|------|------|------|----------|------|
| **Dolma3 Dolmino Pool** | - | OLMo 3 7B **Mid-training용 전체 데이터 풀**. | Mid-training | ODC-BY | [🤗 allenai/dolma3_dolmino_pool](https://huggingface.co/datasets/allenai/dolma3_dolmino_pool) |
| **Dolma3 Dolmino Mix 100B** | 100B tokens | OLMo 3 7B Mid-training 믹스 데이터. | Mid-training | ODC-BY | [🤗 allenai/dolma3_dolmino_mix-100B-1025](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025) |
| **Dolma3 Dolmino Mix 10B** | 10B tokens | Mid-training 소규모 버전. 실험용. | Mid-training | ODC-BY | [🤗 allenai/dolma3_dolmino_mix-10B-1025](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-10B-1025) |
| **Dolma3 Longmino Pool** | - | OLMo 3 7B **Long Context** 학습용 풀. | Long Context | ODC-BY | [🤗 allenai/dolma3_longmino_pool](https://huggingface.co/datasets/allenai/dolma3_longmino_pool) |
| **Dolma3 Longmino Mix 50B** | 50B tokens | Long Context Mid-training 믹스. | Long Context | ODC-BY | [🤗 allenai/dolma3_longmino_mix-50B-1025](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-50B-1025) |

> 📦 **OLMo 3 Pre-training Collection**: [🤗 allenai/Olmo-3-Pre-training](https://huggingface.co/collections/allenai/olmo-3-pre-training)

---

## 다국어 / CoT 데이터셋

> **Chain-of-Thought (CoT)** 데이터는 LLM의 추론 능력을 향상시키는 핵심 요소입니다.
> 
> 다국어 CoT 데이터를 활용하면 한국어 추론 능력도 함께 향상됩니다.

### 한국어 추론 데이터셋 ⭐

| 이름 | 크기 | 설명 | 링크 |
|------|------|------|------|
| **Yi-Sang (KOREAson)** | 5.79M prompts + 3.7M traces | 한국어 네이티브 추론 데이터셋. 웹 Q&A, 시험, STEM, 코드 포함. **가장 큰 한국어 추론 데이터**. | [🤗 KOREAson Collection](https://huggingface.co/collections/KOREAson) |
| **ko-limo** | 1K | LIMO 논문 데이터 한국어 번역. 추론 능력 강화용. | [🤗 junnei/ko-limo](https://huggingface.co/datasets/junnei/ko-limo) |
| **NuminaMath-CoT-Ko** | 860K | NuminaMath 한국어 번역. 수학 추론. CC BY-NC 4.0 | [🤗 ChuGyouk/AI-MO-NuminaMath-CoT-Ko](https://huggingface.co/datasets/ChuGyouk/AI-MO-NuminaMath-CoT-Ko) |

### 다국어 CoT 데이터셋

| 이름 | 크기 | 언어 | 설명 | 링크 |
|------|------|------|------|------|
| **KAIST Multilingual CoT Collection** | 1.84M CoT | 다국어 | Flan Collection 기반 1060개 태스크. CoT 능력 주입용. | [🤗 kaist-ai/CoT-Collection](https://huggingface.co/datasets/kaist-ai/CoT-Collection) |
| **OpenO1-SFT** | - | 영어 | O1 스타일 추론 SFT 데이터. 한국어 번역 가능. | [🤗 O1-OPEN/OpenO1-SFT](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT) |
| **NuminaMath-TIR** | 860K | 영어 | AI Math Olympiad 수상 데이터. **Tool-Integrated Reasoning**. | [🤗 AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) |
| **NuminaMath-CoT** | 859K | 영어 | Chain-of-Thought 수학 문제 풀이. | [🤗 AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| **OpenMathInstruct-2** | 14M | 영어 | GSM8K/MATH 기반 Llama-3.1-405B 합성 데이터. | [🤗 nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |

### 추론 능력 향상을 위한 모델 (참고)

| 모델 | 크기 | 설명 | 링크 |
|------|------|------|------|
| **Nemotron-Research-Reasoning-Qwen-1.5B** | 1.5B | ProRL로 학습된 추론 모델. NVIDIA 연구용. | [🤗 nvidia/Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) |
| **LLaDA2.0-mini** | 16B | Diffusion LLM. MoE Instruction-tuned. | [🤗 inclusionAI/LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini) |
| **LLaDA2.0-flash** | 100B | Diffusion LLM. MoE Instruction-tuned. | [🤗 inclusionAI/LLaDA2.0-flash](https://huggingface.co/inclusionAI/LLaDA2.0-flash) |

> 💡 **팁**: 영어 CoT 데이터를 한국어로 번역하면 저비용으로 한국어 추론 데이터를 확보할 수 있습니다.
> 위의 "무료 번역 전략" 섹션을 참고하세요.

---

## Post-training 데이터셋

### SFT (Supervised Fine-Tuning)

#### 📌 대규모 통합 데이터셋
| 이름 | 크기 | 타입 | 설명 | 라이센스 | 링크 |
|------|------|------|------|----------|------|
| **KoCommercial-Dataset** | 1.44M | 싱글턴 | 상업적 이용 가능한 데이터 병합. **가장 큰 한국어 SFT 데이터**. | 상업적 가능 | [🤗 MarkrAI/KoCommercial-Dataset](https://huggingface.co/datasets/MarkrAI/KoCommercial-Dataset) |
| **open-korean-instructions** | 다양 | 혼합 | 공개 한국어 instruction 데이터 통합 저장소. | 다양함 | [🤗 heegyu/open-korean-instructions](https://huggingface.co/datasets/heegyu/open-korean-instructions) |
| **koVast** | 685K | 멀티턴 | 대규모 멀티턴 한국어 대화 데이터. | - | [🤗 maywell/koVast](https://huggingface.co/datasets/maywell/koVast) |
| **smol-koreantalk** | 460K | 멀티턴 | SmolLM2 학습 데이터(smol-smoltalk) 한국어 번역. | Apache 2.0 | [🤗 lemon-mint/smol-koreantalk](https://huggingface.co/datasets/lemon-mint/smol-koreantalk) |

#### 📌 고품질 번역 데이터셋
| 이름 | 크기 | 타입 | 설명 | 라이센스 | 링크 |
|------|------|------|------|----------|------|
| **ShareGPT DeepL 번역** | 620K(싱글)+84K(멀티) | 멀티턴 | ShareGPT 데이터 DeepL 번역. | CC BY 2.0 KR | [🤗 junelee/sharegpt_deepl_ko](https://huggingface.co/datasets/junelee/sharegpt_deepl_ko) |
| **KULLM v2** | 153K | 싱글턴 | GPT4ALL, Dolly, Vicuna 데이터 DeepL 번역. | - | [🤗 nlpai-lab/kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) |
| **OpenOrca-gugugo-ko** | 640K+ | 싱글턴 | OpenOrca 한국어 번역 (진행 중). | - | [🤗 squarelike/OpenOrca-gugugo-ko](https://huggingface.co/datasets/squarelike/OpenOrca-gugugo-ko) |
| **Ko.WizardLM_evol_instruct_V2_196k** | 196K | 싱글턴 | WizardLM evol_instruct 한국어 번역. | - | [🤗 Dataset](https://huggingface.co/datasets/nlp-with-deeplearning/Ko.WizardLM_evol_instruct_V2_196k) |

#### 📌 2024-2025 최신 데이터셋 ⭐
| 이름 | 크기 | 타입 | 설명 | 라이센스 | 링크 |
|------|------|------|------|----------|------|
| **Magpie-Pro-MT-300K-ko** | 300K | 멀티턴 | **Magpie 기법**으로 생성된 합성 한국어 instruction 데이터. | - | [🤗 nayohan/Magpie-Pro-MT-300K-v0.1-ko](https://huggingface.co/datasets/nayohan/Magpie-Pro-MT-300K-v0.1-ko) |
| **qarv-instruct-100k** | 100K | 싱글턴 | **한국 문화/지식** 필요한 instruction-답변 쌍. HAERAE-HUB 제작. | - | [🤗 HAERAE-HUB/qarv-instruct-100k](https://huggingface.co/datasets/HAERAE-HUB/qarv-instruct-100k) |
| **KoAlpaca-RealQA** | 18K | 싱글턴 | 2023-2024 ChatKoAlpaca **실제 사용자 대화** 기반. | CC BY-SA 4.0 | [🤗 beomi/KoAlpaca-RealQA](https://huggingface.co/datasets/beomi/KoAlpaca-RealQA) |
| **Won-Instruct** | 86K | 싱글턴 | **금융 도메인** 특화 한국어 instruction 데이터. KRX 제작. | 확인 필요 | [🤗 KRX-Data/Won-Instruct](https://huggingface.co/datasets/KRX-Data/Won-Instruct) |
| **ko-limo** | 1K | 싱글턴 | LIMO 논문 데이터 한국어 번역. **추론 능력** 강화용. | - | [🤗 junnei/ko-limo](https://huggingface.co/datasets/junnei/ko-limo) |
| **ko_llm_annotations v3** | - | 합성 | 한국어 LLM 합성 데이터. 2024년 9월 업데이트. | - | [🤗 devngho/ko_llm_annotations](https://huggingface.co/datasets/devngho/ko_llm_annotations) |

#### 📌 도메인 특화 데이터셋
| 이름 | 크기 | 도메인 | 설명 | 라이센스 | 링크 |
|------|------|--------|------|----------|------|
| **HR-Instruct-Math-v0.1** | 30K | 수학 | 한국어 수학 instruction 데이터. | - | [🤗 HAERAE-HUB/HR-Instruct-Math-v0.1](https://huggingface.co/datasets/HAERAE-HUB/HR-Instruct-Math-v0.1) |
| **orca-math-korean** | 193K | 수학 | Microsoft orca-math 한국어 번역. | - | [🤗 kuotient/orca-math-word-problems-193k-korean](https://huggingface.co/datasets/kuotient/orca-math-word-problems-193k-korean) |
| **ko_medical_chat** | 3K | 의료 | 의료 대화 데이터. | - | [🤗 squarelike/ko_medical_chat](https://huggingface.co/datasets/squarelike/ko_medical_chat) |
| **CounselGPT** | 13K+8.7K | 상담 | GPT로 생성한 상담 대화 데이터. | - | [GitHub](https://github.com/MrBananaHuman/CounselGPT) |
| **glaive-function-calling-v2-ko** | 15.2K | Function Calling | 함수 호출 학습용 데이터. | - | [🤗 heegyu/glaive-function-calling-v2-ko](https://huggingface.co/datasets/heegyu/glaive-function-calling-v2-ko) |

---

### DPO / Preference 데이터셋

| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **ko_Ultrafeedback_binarized** | 62K | Ultrafeedback 번역 + 정제. DPO 학습용. | 비상업적* | [🤗 maywell/ko_Ultrafeedback_binarized](https://huggingface.co/datasets/maywell/ko_Ultrafeedback_binarized) |
| **orca-dpo-pairs-ko** | 36K | 3개 DPO 데이터셋 병합 후 중복 제거. | - | [🤗 SJ-Donald/orca-dpo-pairs-ko](https://huggingface.co/datasets/SJ-Donald/orca-dpo-pairs-ko) |
| **orca-math-korean-preference** | 193K | 수학 DPO 데이터셋. | - | [🤗 kuotient/orca-math-korean-preference](https://huggingface.co/datasets/kuotient/orca-math-korean-preference) |
| **K2-Feedback** | 100K | 한국어 평가 능력 향상용. Prometheus 학습 데이터 기반. | - | [🤗 HAERAE-HUB/K2-Feedback](https://huggingface.co/datasets/HAERAE-HUB/K2-Feedback) |

> *비상업적: 데이터 직접 상업 사용 불가, 모델 학습 후 상업 사용 가능

#### 🔬 OLMo 3 Dolci Post-training Datasets (2025 최신) ⭐
| 이름 | 크기 | 용도 | 설명 | 링크 |
|------|------|------|------|------|
| **Dolci-Think-SFT-7B** | 2.27M | SFT | OLMo 3 7B Think 모델 SFT 데이터. | [🤗 allenai/Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) |
| **Dolci-Think-DPO-7B** | 150K | DPO | OLMo 3 7B Think 모델 DPO 데이터. | [🤗 allenai/Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B) |
| **Dolci-Think-RL-7B** | 102K | RL | OLMo 3 7B Think 모델 RL 데이터. | [🤗 allenai/Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B) |
| **Dolci-Instruct-SFT** | 2.15M | SFT | OLMo 3 Instruct 모델 SFT 데이터. | [🤗 allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) |
| **Dolci-Instruct-DPO** | 260K | DPO | OLMo 3 Instruct 모델 DPO 데이터. | [🤗 allenai/Dolci-Instruct-DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO) |
| **Dolci-Think-SFT-Python** | 1.09M | Code SFT | Python 코드 SFT 믹스. | [🤗 allenai/Dolci-Think-SFT-Python](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-Python) |
| **Dolci-RL-Zero-Math-7B** | 13.3K | RL Zero | 수학 도메인 RL Zero 데이터. | [🤗 allenai/Dolci-RL-Zero-Math-7B](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Math-7B) |
| **Dolci-RL-Zero-Code-7B** | 13.3K | RL Zero | 코드 도메인 RL Zero 데이터. | [🤗 allenai/Dolci-RL-Zero-Code-7B](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Code-7B) |

> 📦 **OLMo 3 Post-training Collection**: [🤗 allenai/Olmo-3-Post-training](https://huggingface.co/collections/allenai/olmo-3-post-training)

#### 🚀 NVIDIA Nemotron Post-training v3 Datasets (2025 최신) ⭐
| 이름 | 크기 | 용도 | 설명 | 링크 |
|------|------|------|------|------|
| **Nemotron-Instruction-Following-Chat-v1** | 288K | SFT | Instruction Following Chat 데이터. | [🤗 nvidia/Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) |
| **Nemotron-Math-Proofs-v1** | 925K | Math | 수학 증명 데이터. | [🤗 nvidia/Nemotron-Math-Proofs-v1](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v1) |
| **Nemotron-Math-v2** | - | Math | 수학 Post-training v2. | [🤗 nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) |
| **Nemotron-Science-v1** | 226K | Science | 과학 도메인 데이터. | [🤗 nvidia/Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) |
| **Nemotron-Agentic-v1** | - | Agentic | 에이전트 학습용 데이터. | [🤗 nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) |
| **Nemotron-Competitive-Programming-v1** | - | Code | 경쟁 프로그래밍 데이터. | [🤗 nvidia/Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) |
| **Nemotron-3-Nano-RL-Training-Blend** | - | RL | Nemotron Nano RL 학습 블렌드. | [🤗 nvidia/Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) |

> 📦 **NVIDIA Nemotron Post-training Collection**: [🤗 nvidia/Nemotron-Post-Training-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)

#### 🤖 GRPO / RL 학습용 데이터셋 (DeepSeek-R1 스타일) ⭐
> **GRPO (Group Relative Policy Optimization)**는 DeepSeek-R1에서 도입된 RL 방법론으로,
> PPO보다 효율적이며 수학/코드 추론 능력 향상에 탁월합니다.

| 이름 | 크기 | 용도 | 설명 | 링크 |
|------|------|------|------|------|
| **NuminaMath-TIR** | 860K | Math GRPO | AI Math Olympiad 수상 데이터. **Tool-Integrated Reasoning**. | [🤗 AI-MO/NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) |
| **NuminaMath-CoT** | 859K | Math GRPO | Chain-of-Thought 수학 문제 풀이. | [🤗 AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| **OpenMathInstruct-2** | 14M | Math | GSM8K/MATH 기반 Llama-3.1-405B 합성 데이터. | [🤗 nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) |
| **Dolci-RL-Zero-Math-7B** | 13.3K | GRPO | OLMo 3 수학 도메인 RL Zero 데이터. | [🤗 allenai/Dolci-RL-Zero-Math-7B](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Math-7B) |
| **Dolci-RL-Zero-Code-7B** | 13.3K | GRPO | OLMo 3 코드 도메인 RL Zero 데이터. | [🤗 allenai/Dolci-RL-Zero-Code-7B](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Code-7B) |
| **Nemotron-3-Nano-RL-Training-Blend** | - | GRPO | Nemotron Nano RL 학습 블렌드. | [🤗 nvidia/Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) |

> 📚 **GRPO 구현**: HuggingFace TRL 라이브러리의 `GRPOTrainer` 클래스 사용 [📖 TRL GRPO 문서](https://huggingface.co/docs/trl/main/en/grpo_trainer)

#### 🇰🇷 한국어 수학 추론 데이터셋
> 💡 위 [한국어 추론 데이터셋](#한국어-추론-데이터셋-) 및 [도메인 특화 데이터셋](#-도메인-특화-데이터셋)의 **NuminaMath-CoT-Ko**, **orca-math-korean** 참조

---

### RLHF / RM 데이터셋

| 이름 | 크기 | 설명 | 라이센스 | 링크 |
|------|------|------|----------|------|
| **ko_hh-rlhf-20k_filtered** | 20K | Anthropic hh-rlhf 한국어 번역 (필터링). | - | [🤗 maywell/ko_hh-rlhf-20k_filtered](https://huggingface.co/datasets/maywell/ko_hh-rlhf-20k_filtered) |
| **hh-rlhf-ko** | 113K | Anthropic hh-rlhf 전체 번역. | - | [🤗 heegyu/hh-rlhf-ko](https://huggingface.co/datasets/heegyu/hh-rlhf-ko) |
| **PKU-SafeRLHF-ko** | 164K | PKU 안전 RLHF 데이터 번역. | - | [🤗 heegyu/PKU-SafeRLHF-ko](https://huggingface.co/datasets/heegyu/PKU-SafeRLHF-ko) |
| **kor_ethical_question_answer** | 29.1K | AI 윤리적/비윤리적 QA 데이터. | - | [🤗 MrBananaHuman/kor_ethical_question_answer](https://huggingface.co/datasets/MrBananaHuman/kor_ethical_question_answer) |
| **korean_rlhf_dataset** | 107K | 성균관대 산학협력 SFT 데이터. | - | [🤗 jojo0217/korean_rlhf_dataset](https://huggingface.co/datasets/jojo0217/korean_rlhf_dataset) |
| **AIHub RLHF Dataset** | SFT 13K, RM 33K, PPO 33K | 공식 AIHub 제공. RM 데이터는 5개 답변 순위 포함. | - | [AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71748) |

---

## 무료 번역 전략 (영어 - 한국어)
> **핵심 아이디어**: 영어 고품질 데이터셋은 풍부하므로, 무료 번역 도구를 활용하여 한국어 데이터를 확보합니다.
> 
>  **비용 절감**: 상용 번역 API 대신 무료 도구를 활용하면 대규모 데이터셋도 무료로 구축 가능합니다.

### 무료 번역 도구 비교

| 도구 | 무료 한도 | 한국어 품질 | 특징 | 설치/사용법 |
|------|-----------|-------------|------|-------------|
| **Google Translate (비공식)** | 무제한 | ⭐⭐⭐⭐⭐ | 가장 높은 한국어 품질, 비공식 라이브러리 | `pip install googletrans==4.0.0-rc1` |
| **DeepL API Free** | 500K chars/month | ⭐⭐⭐⭐ | 유럽어 최고, 한국어도 양호 | [API 키 신청](https://www.deepl.com/pro-api) |
| **LibreTranslate** | 무제한 (셀프호스팅) | ⭐⭐⭐ | 오픈소스, 로컬 실행 가능 | `pip install libretranslate` |
| **MarianMT (HuggingFace)** | 무제한 | ⭐⭐⭐ | 오픈소스 NMT 모델, 완전 로컬 | `transformers` 라이브러리 |
| **NLLB (Meta)** | 무제한 | ⭐⭐⭐ | 200개 언어, 고품질 다국어 번역 | [🤗 facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) |
| **lfm2-1.2b-koen-mt-v8-rl-10k-merged-GGUF** | 무제한 | ⭐⭐⭐⭐ | 1.2B 수준에서 최고 성능을 보이는 한국어-영어 번역 모델 | [🤗 gyung/lfm2-1.2b-koen-mt-v8-rl-10k-merged-GGUF](https://huggingface.co/gyung/lfm2-1.2b-koen-mt-v8-rl-10k-merged-GGUF) |


### 추천 번역 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. 영어 데이터셋 선택 (예: OpenOrca, Alpaca, WizardLM)            │
├─────────────────────────────────────────────────────────────────────┤
│  2. Google Translate (비공식) 또는 LFM2로 1차 번역                 │
│     → 무료이며 품질이 가장 좋음                                     │
├─────────────────────────────────────────────────────────────────────┤
│  3. 품질 필터링 (선택사항)                                          │
│     → LLM으로 번역 품질 평가 또는 rule-based 필터링                 │
├─────────────────────────────────────────────────────────────────────┤
│  4. 최종 한국어 데이터셋 생성                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Google Translate 사용 예시 (Python)

```python
from googletrans import Translator

translator = Translator()

def translate_to_korean(text):
    try:
        result = translator.translate(text, src='en', dest='ko')
        return result.text
    except Exception as e:
        return None

# 대량 번역 시 rate limiting 주의
# 멀티스레딩 + 재시도 로직 권장
```

> 💡 **팁**: 대규모 번역 시 [Argilla Distilabel](https://github.com/argilla-io/distilabel) 또는 [Curator](https://github.com/bespokelabsai/curator/) 같은 도구를 활용하면 멀티스레딩, 자동 재시도, 체크포인트 등을 지원합니다.

---

## 평가용 데이터셋

| 이름 | 크기 | 타입 | 설명 | 링크 |
|------|------|------|------|------|
| **KMMLU** | 243K | MCQA | 45개 주제 전문가 수준 한국어 벤치마크. | [🤗 HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) |
| **HAE-RAE-BENCH** | 1.5K | MCQA | 어휘, 역사, 상식, 독해 평가. | [GitHub](https://github.com/HAETAE-project/HAE-RAE-BENCH) |
| **CSAT-QA** | 0.9K | MCQA | 국어 수능 문제. | [🤗 HAERAE-HUB/CSAT-QA](https://huggingface.co/datasets/HAERAE-HUB/CSAT-QA) |
| **K2-Eval** | 90 | 생성 | 한국 문화 지식 필요한 90개 지시문. GPT-4 평가. | [🤗 HAERAE-HUB/K2-Eval](https://huggingface.co/datasets/HAERAE-HUB/K2-Eval) |
| **KorMedMCQA** | <1K | MCQA | 한국어 의료 QA 벤치마크. | [🤗 sean0042/KorMedMCQA](https://huggingface.co/datasets/sean0042/KorMedMCQA) |
| **LogicKor** | - | 다분야 | 한국어 사고력 벤치마크. | [🤗 Leaderboard](https://huggingface.co/spaces/instructkr/LogicKor-leaderboard) |

---

## 유용한 컬렉션

| 컬렉션 | 설명 | 링크 |
|--------|------|------|
| **나요한님 번역 데이터** | 영어 데이터셋 한국어 번역. llama3-instrucTrans 사용. | [🤗 Collection](https://huggingface.co/collections/nayohan/translated-en-ko-dataset-6665023b1036d124ede5f81c) |
| **나요한님 Magpie 번역** | Magpie 데이터셋 한국어 번역. | [🤗 Collection](https://huggingface.co/collections/youjunhyeok/magpie-ko-66cbc570a9891d5b43a170d9) |
| **유준혁님 번역 데이터** | 영한 번역 데이터셋 모음. | [🤗 Collection](https://huggingface.co/collections/youjunhyeok/en-ko-translate-6703474b419fcb9e5d6a7852) |
| **송영숙님 Korean Dataset** | 허깅페이스 한국어 데이터셋 정리 (2024.10 기준). | [GitHub](https://github.com/songys/huggingface_KoreanDataset) |

---

## 참고 자료

### 합성 데이터 구축
- [ko-genstruct](https://github.com/iKnowLab-Projects/ko-genstruct) - 한국어 합성 데이터 생성
- [evolve-instruct](https://github.com/lcw99/evolve-instruct) - Instruction 증강 기법

### 평가 플랫폼
- [Ko Chatbot Arena](https://huggingface.co/spaces/instructkr/ko-chatbot-arena-leaderboard) - 한국어 챗봇 ELO 랭킹
- [LogicKor Leaderboard](https://huggingface.co/spaces/instructkr/LogicKor-leaderboard) - 다분야 사고력 평가
- [호랑이 LLM 리더보드](https://wandb.ai/wandb-korea/korean-llm-leaderboard/reports) - W&B 한국어 LLM 평가


### 🇰🇷 한국 기업 LLM 기술 보고서 (데이터 전략 참고)
| 기업 | 모델 | 핵심 전략 | 보고서 |
|------|------|----------|--------|
| **Upstage** | Solar Open | 4.5T 합성 데이터 + Progressive Curriculum + SnapPO | [📜 Technical Report](https://huggingface.co/upstage/Solar-Open-100B/blob/main/solar-open-technical-report.pdf) |
| **LG AI Research** | K-EXAONE | 6개 국어 + 256K Context + MoE 구조 | [📜 arXiv](https://arxiv.org/pdf/2601.01739) |
| **SK Telecom** | A.X K1 | 10T 토큰 + Multi-stage Pipeline + Think-Fusion | [📜 Tech Report](https://github.com/SKT-AI/A.X-K1/releases/download/v1.0/A_X_Tech_Report.pdf) |

---

## 🎯 Yaongi 프로젝트 권장 데이터 파이프라인

> ⚠️ **핵심 인사이트** (Solar Open, K-EXAONE, A.X K1 기술 보고서 기반):
> - 단순 웹 크롤링만으로는 부족 → **합성 데이터(Synthetic Data)** 필수
> - **커리큘럼 학습** (Progressive Curriculum): 단계별 데이터 품질 조절
> - 500M 모델은 용량이 작으므로 **압축적이고 밀도 높은 데이터** 필요

### Phase 1: Pre-training (500M 모델, 300B 토큰)

```
┌─────────────────────────────────────────────────────────────┐
│  영어 (50% = 150B)                한국어 (50% = 150B)       │
├─────────────────────────────────────────────────────────────┤
│  • FineWeb-Edu                      • Korean Wikipedia        │
│  • SmolLM-Corpus                  • korean_textbooks (합성)   │
│  • Nemotron-CC                     • aihub-en-ko-translation  │
│  • The Stack (코드)               • KOREAN-SyntheticText     │
└─────────────────────────────────────────────────────────────┘
```

**한국어 Pre-training 데이터 확보 전략:**
- 허깅페이스에 있는 합성 데이터셋(korean_textbooks, KOREAN-SyntheticText) 활용
- 한영 번역 말뭉치(aihub-en-ko-translation-12m) Pre-training에 포함
- 부족 시 Post-training 데이터(KoCommercial, koVast) 일부 Pre-training에 활용

#### 📊 커리큘럼 학습 전략 (Solar Open 참조)
| 단계 | 토큰 | 데이터 구성 | 목표 |
|------|------|-------------|------|
| **Phase 1a** | 0~200B | 일반 한국어/영어/코드 혼합 | 기초 언어 능력 |
| **Phase 1b** | 200~280B | 고품질 교과서 + 전문 텍스트 | 지식 밀도 |
| **Phase 1c (Annealing)** | 280~300B | **합성 CoT 데이터 집중** | 추론 능력 극대화 |

### Phase 2: Mid-training / Continued Pre-training

```
┌─────────────────────────────────────────────────────────────┐
│  고품질 한국어 집중 (50~100B 토큰)                          │
├─────────────────────────────────────────────────────────────┤
│  • Dolma3 Dolmino Mix (OLMo 3 스타일)                       │
│  • Korean Pretraining Collection                            │
│  • 뉴스 기사 + 사설 (논리적 글쓰기)                          │
│  • 합성 한국어 CoT 데이터 (GPT-4/Claude로 생성)             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Post-training (SFT)

```
┌─────────────────────────────────────────────────────────────┐
│  1순위: KoCommercial-Dataset (1.44M) - 상업적 이용 가능     │
│  2순위: open-korean-instructions 통합 데이터                │
│  3순위: Magpie-Pro-MT-300K-ko (합성 데이터)                 │
│  ───────────────────────────────────────────────────────────│
│  💡 English 참고: Dolci-Instruct-SFT, Nemotron-IF-Chat      │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4: Alignment (DPO/RLHF → **GRPO**)

> ⭐ **GRPO (Group Relative Policy Optimization)** 기반 RL이 핵심!  
> DeepSeek-R1에서 입증된 방법으로, PPO보다 효율적이며 수학/코드 추론에 탁월합니다.

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 - DPO (기본 정렬)                                  │
│  • ko_Ultrafeedback_binarized + orca-dpo-pairs-ko          │
├─────────────────────────────────────────────────────────────┤
│  Stage 2 - GRPO (추론 강화) ⭐                              │
│  • NuminaMath-CoT-Ko (수학 추론)                           │
│  • NuminaMath-TIR (Tool-Integrated Reasoning)              │
│  • Dolci-RL-Zero-Math, Dolci-RL-Zero-Code                  │
└─────────────────────────────────────────────────────────────┘
```

### 💡 합성 데이터 활용 가이드

> 500M 모델은 허깅페이스에 있는 **기존 합성 데이터셋**을 활용하면 됩니다.  
> 직접 생성할 필요 없이 아래 데이터셋들을 바로 사용하세요!

| 카테고리 | 추천 데이터셋 | 용량 | 효과 |
|----------|---------------|------|------|
| **한국어 교과서** | maywell/korean_textbooks | 1~10M | 지식 밀도 ↑ |
| **한국어 합성** | KOREAN-SyntheticText-1.5B | 1.5B | Pre-training 확장 |
| **한영 번역** | aihub-en-ko-translation-12m | 12M | 지식 주입 |
| **수학 추론** | NuminaMath-CoT-Ko, orca-math-korean | 200K+ | 추론 능력 ↑ |
| **멀티턴 대화** | Magpie-Pro-MT-300K-ko | 300K | SFT 품질 ↑ |

---

## 참고 논문

> 아래 논문들에서 LLM 학습 전략, 데이터셋 구성, RL 기법 등의 인사이트를 얻을 수 있습니다.

### RL 학습 및 추론 능력 향상

| 논문 | 핵심 기여 | 관련 리소스 | 링크 |
|------|----------|-------------|------|
| **ProRL: Prolonged RL Expands Reasoning Boundaries** | 장기간 RL로 base 모델에서 불가능한 추론 전략 발견. KL divergence 제어, reference policy resetting. | [🤗 Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) | [📜 arXiv:2505.24864](https://arxiv.org/abs/2505.24864) |
| **Stabilizing RL with LLMs** | 30B MoE 모델 RL 안정화 레시피. Importance sampling, Clipping, **Routing Replay** (MoE 전용). | - | [📜 arXiv:2512.01374](https://arxiv.org/abs/2512.01374) |

### Agent 및 Deep Research

| 논문 | 핵심 기여 | 관련 리소스 | 링크 |
|------|----------|-------------|------|
| **Step-DeepResearch** | Atomic Capability 기반 합성 데이터 생성. Progressive Training (Mid-training → SFT → RL). 32B 모델로 O1급 성능. | [💻 GitHub](https://github.com/stepfun-ai/StepDeepResearch), ADR-Bench (중국어 벤치마크) | [📜 arXiv:2512.20491](https://arxiv.org/abs/2512.20491) |

### 모델 아키텍처 및 학습 기법

| 논문 | 핵심 기여 | 관련 리소스 | 링크 |
|------|----------|-------------|------|
| **LLaDA 2.0: Scaling Diffusion LLM to 100B** | AR → Diffusion LLM 변환. 3-phase Block-level WSD 학습. Parallel decoding으로 효율적 추론. | [🤗 LLaDA 2.0 Collection](https://huggingface.co/collections/inclusionAI/llada-20), [💻 dFactory](https://github.com/inclusionAI/dFactory), [💻 dInfer](https://github.com/inclusionAI/dInfer) | [📜 arXiv:2512.15745](https://arxiv.org/abs/2512.15745) |
| **Code Foundation Models to Agents** | 코드 LLM 전체 생명주기 서베이. Scaling law, 데이터 구성, RL 실험. | 코드 Pre-training, SFT, RL 실험 데이터 | [📜 arXiv:2511.18538](https://arxiv.org/abs/2511.18538) |

### 논문에서 배울 수 있는 핵심 인사이트

1. **ProRL**: 장기간 RL 학습이 base 모델에서 접근 불가능한 추론 전략을 발견할 수 있음
2. **Step-DeepResearch**: 복잡한 태스크를 **원자적 능력(Atomic Capabilities)**으로 분해하여 학습
3. **Stabilizing RL**: MoE 모델에서 **Routing Replay**가 정책 staleness 완화에 필수적
4. **LLaDA 2.0**: Diffusion LLM이 AR 모델과 경쟁 가능하며, parallel decoding으로 추론 효율화

---

## 📖 외부 참고 자료

### 데이터셋 큐레이션
- [mlabonne/llm-datasets](https://github.com/mlabonne/llm-datasets) - Post-training용 데이터셋 및 도구 큐레이션 리스트 ⭐
- [open-korean-instructions](https://github.com/HeegyuKim/open-korean-instructions) - 이 README의 주요 참고 자료

### 데이터 도구
- [Curator](https://github.com/bespokelabsai/curator/) - 합성 데이터 생성 파이프라인
- [Distilabel](https://github.com/argilla-io/distilabel) - SFT/DPO 데이터 생성 및 증강
- [Argilla](https://argilla.io/) - 데이터 필터링 및 어노테이션 플랫폼

---

> 📅 **Last Updated**: 2026-01-10
> 
> 💡 **기여하기**: 새로운 데이터셋 발견 시 PR 또는 Issue로 알려주세요!

