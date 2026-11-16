### Fine-Tuning Report: Monolingual Language Model Adaptation on XNLI

#### 1. Overview of Experimental Setup

To adapt a causal monolingual language model for natural language inference (NLI), we fine-tuned `goldfish-models/eng_latn_1000mb` (Chang et al., 2024) on the English subset of the **XNLI** dataset (Conneau et al., 2018). Because Goldfish models use a GPT-2 causal language modeling architecture (`AutoModelForCausalLM`), three-way classification (entailment, neutral, contradiction) was framed as autoregressive text completion. Following the Goldfish model specifications, each sequence was formatted with a prepended `[CLS]` (`bos_token`) and terminated with a `[SEP]` (`eos_token`) within a maximum context window of 512 tokens.

#### 2. Rationale for Using LoRA (Low-Rank Adaptation)

Updating all ~124 million parameters of `goldfish-models/eng_latn_1000mb` during supervised fine-tuning is memory-intensive and risks overfitting on downstream classification templates. Instead, LoRA freezes the pre-trained GPT-2 backbone and injects small trainable rank-decomposition matrices into the attention and MLP projection layers, reducing GPU VRAM consumption and accelerating training. Keeping the base weights frozen preserves the model's general English representations while isolating NLI task adaptation in lightweight adapter checkpoints.

#### 3. Training & LoRA Parameters Used

The exact hyperparameter configuration used in the training notebook (`fine_tune_goldfish_xnli.ipynb`) is as follows:

* **LoRA Configuration (`LoraConfig`):**
* **Task Type:** `CAUSAL_LM`


* **Rank ($r$):** `16` (dimension of the low-rank update matrices)


* **LoRA Alpha ($\alpha$):** `32` (scaling factor, yielding a scaling ratio of $\alpha / r = 2$)


* **LoRA Dropout:** `0.05` (regularization applied to adapter layers)


* **Target Modules:** `["c_attn", "c_proj", "c_fc"]` (GPT-2 attention and feed-forward MLP layers)




* **Training Arguments (`TrainingArguments`):**
* **Learning Rate:** `5e-5`


* **Batch Size & Accumulation:** `per_device_train_batch_size = 4`, `per_device_eval_batch_size = 4`, and `gradient_accumulation_steps = 8` (effective batch size of `32`)


* **Epochs:** `3` (`num_train_epochs = 3`)


* **Weight Decay:** `0.01`


* **Max Sequence Length:** `512` tokens


* **Precision & Logging:** `fp16 = True`, `eval_strategy = "epoch"`, `save_strategy = "epoch"`, `logging_steps = 50`


* **Reproducibility Seeds:** `seed = 4`, `data_seed = 4` (alongside deterministic PyTorch/CuDNN flags)





---

#### References

* Chang, T. A., Arnett, C., Tu, Z., & Bergen, B. K. (2024). *Goldfish: Monolingual Language Models for 350 Languages*. Preprint. [https://www.arxiv.org/abs/2408.10441](https://www.arxiv.org/abs/2408.10441)
* Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S. R., Schwenk, H., & Stoyanov, V. (2018). *XNLI: Evaluating Cross-lingual Sentence Representations*. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium. Association for Computational Linguistics.

---

#### BibTeX

```bibtex
@article{chang-etal-2024-goldfish,
  title={Goldfish: Monolingual Language Models for 350 Languages},
  author={Chang, Tyler A. and Arnett, Catherine and Tu, Zhuowen and Bergen, Benjamin K.},
  journal={Preprint},
  year={2024},
  url={https://www.arxiv.org/abs/2408.10441},
}

@InProceedings{conneau2018xnli,
  author = {Conneau, Alexis
                 and Rinott, Ruty
                 and Lample, Guillaume
                 and Williams, Adina
                 and Bowman, Samuel R.
                 and Schwenk, Holger
                 and Stoyanov, Veselin},
  title = {XNLI: Evaluating Cross-lingual Sentence Representations},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  location = {Brussels, Belgium},
}

```