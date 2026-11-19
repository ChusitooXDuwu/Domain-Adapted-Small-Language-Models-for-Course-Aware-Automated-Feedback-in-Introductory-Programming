# Model adaptation (QLoRA) — Qwen2.5-Coder-1.5B

Scripts and configuration used to adapt `Qwen/Qwen2.5-Coder-1.5B-Instruct` for
course-aware feedback on failed Python submissions (Section 4.4–4.5 of the paper).
Only the training setup is here: no datasets, adapter weights, evaluation data or figures.

## Files

| File | What it is |
|---|---|
| `entrenar_qwen_qlora.py` | Training script: 4-bit base (NF4, double quant), LoRA adapters on all attention and MLP projections, HF `Trainer`, saves the adapter to `qwen-socratic-feedback/final/`. |
| `adapter_config.json` | The PEFT/LoRA configuration of the trained adapter, as saved by the run (`r=16`, `alpha=32`, `dropout=0.05`, 7 target modules). |
| `PROMPT_ENTRENAMIENTO_QLORA.md` | The prompt template used to build each training example (`messages` format): reference solution, submitted code and the failed doctest. |
| `inferencia_portable.py` | Loads the base model plus the adapter and generates feedback; use it to run the adapted model on a new submission. |
| `requirements.txt` | Python dependencies. |

## Hyperparameters

| | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Quantization | 4-bit NF4, double quantization, bf16 compute |
| LoRA | `r = 16`, `alpha = 32`, `dropout = 0.05`, bias none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | `2e-4` (weight decay 0.01) |
| Batch | 2 per device × gradient accumulation 8 (effective 16) |
| Epochs | 3, evaluation and checkpoint every epoch, best model by `eval_loss` |
| Warmup | 50 steps |
| Max sequence length | 2048 |
| Optimizer | `paged_adamw_8bit`, gradient checkpointing on |
| Hardware | NVIDIA RTX 4070 Ti SUPER, 16 GB VRAM |

## Data format

The script expects `train.jsonl` and `test.jsonl` in the working directory (70/30 split
of the 270 synthetic failing submissions). Each line is one conversation:

```json
{"messages": [
  {"role": "user", "content": "<prompt built from PROMPT_ENTRENAMIENTO_QLORA.md>"},
  {"role": "assistant", "content": "• <human-written feedback bullet 1>\n• <bullet 2>"}
]}
```

The conversation is serialized with the tokenizer's chat template and trained with a
causal-LM objective. The dataset itself (course exercises, synthetic submissions and the
instructional team's feedback) is not distributed in this repository.

## Run

```bash
pip install -r requirements.txt
# place train.jsonl and test.jsonl next to the script
python entrenar_qwen_qlora.py
# inference with the resulting adapter
python inferencia_portable.py
```
