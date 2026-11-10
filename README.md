1. Research Objective:  
Our project aims to systematically investigate the semantic robustness of language models in English, Russian, and Chinese. We will use Meta’s Cross-Lingual Natural Language Inference (XNLI) corpus as a benchmark reference and adapt the CheckList behavioral-testing framework to evaluate model behavior beyond standard accuracy.
We will examine two groups of models: language-specific Goldfish models, fine-tuned on NLI data as task-specific baselines, and general-purpose multilingual large language models (LLMs), evaluated using zero-shot prompting.
We plan to construct multilingual test suites targeting specific linguistic and semantic capabilities, such as negation, quantifier reasoning, semantic roles, and named-entity robustness. Using Minimum Functionality Tests (MFT), Invariance Tests (INV), and Directional Expectation Tests (DIR), we will examine whether models behave as expected under controlled input variations and identify weaknesses that conventional benchmark evaluation may overlook.
Furthermore, we will complement behavioural testing with an internal evaluation via probing to analyse how internal representations align across languages and diagnose the mechanical causes behind behavioural failures. 

2. Dataset:  
facebook/xnli: https://huggingface.co/datasets/facebook/xnli
We will use XNLI as a standard benchmark reference and as a source of candidate examples for test development. The dataset provides premise–hypothesis pairs labeled as entailment, contradiction, or neutral.
Our evaluation will focus on English, Chinese, and Russian. In addition to standard benchmark evaluation, we will construct a separate set of controlled behavioral tests targeting specific semantic phenomena.

3. Models:  
We plan to compare task-specific monolingual models fine-tuned for NLI with general-purpose multilingual LLMs evaluated without NLI-specific fine-tuning.
Task-specific reference models: Goldfish models fine-tuned on XNLI dataset
- goldfish-models/eng_latn_1000mb
- goldfish-models/rus_cyrl_1000mb
- goldfish-models/zho_hans_1000mb

These fine-tuned language-specific models will serve as task-specific baselines for investigating semantic robustness after supervised NLI training.
General-purpose multilingual LLMs: zero-shot evaluation
- swiss-ai/Apertus-v1.5-8B
This model will be evaluated using standardized prompts that ask them to classify premise–hypothesis pairs as entailment, contradiction, or neutral. We will not fine-tune them specifically on NLI data or provide labeled demonstrations.

5. Internal Evaluation:  
In addition to behavioral testing, we will conduct an internal evaluation using linear or low-rank probing on hidden representations extracted from Apertus-v1.5-8B. 

We will use probing as a diagnostic tool to investigate the root causes of failure cases identified during CheckList behavioral testing (e.g., when a model flips its prediction under a negation or quantifier perturbation).
For instances where Apertus-v1.5-8B fails a CheckList behavioral test (such as an INV or DIR test), we will probe intermediate layer activations to determine whether the required semantic feature (e.g., negation scope or argument structure) is correctly encoded in the hidden states but fails to be routed to the final classification head, or if the representation itself lacks the semantic information.
To ensure probing metrics reflect genuine latent representations rather than probe memorization, we will evaluate probe selectivity using control tasks and/or rank-constrained/linear probe architectures.
