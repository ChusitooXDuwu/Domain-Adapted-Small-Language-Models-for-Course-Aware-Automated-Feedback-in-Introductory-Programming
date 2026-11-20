# Domain-Adapted Small Language Models for Course-Aware Automated Feedback in Introductory Programming

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#versioning)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-accepted%20at%20ICAI%202026-orange.svg)](https://icai.itiud.org/?pid=acceptedPapers)

Eduardo Rosales, Wyo Hann Chu, Rubén Manrique and Mario Sánchez — Universidad de los Andes, Bogotá, Colombia

This repository contains the software and data that accompany the paper of the same title. The software adapts
the small language model Qwen2.5-Coder-1.5B-Instruct with QLoRA to write short, course-aware feedback on failed
Python submissions. The data are the human evaluations used to assess that feedback.

What the repository provides:

- **Training:** `training/entrenar_qwen_qlora.py` fine-tunes the base model with QLoRA (4-bit quantization plus
  LoRA adapters). It trains on conversations that pair a failed submission with feedback written by the course's
  instructional team.
- **Inference:** `training/inferencia_portable.py` loads the base model and a trained adapter, and generates
  feedback for new submissions on a GPU or a CPU.
- **Configuration and prompt:** the LoRA configuration of the adapter used in the paper, and the prompt template
  used to build the training examples.
- **Evaluation data:** `data_resp.csv`, the 1,000 human ratings of feedback produced by the adapted model and by
  five baseline models.

The training data and the trained adapter weights are not distributed (see
[What is not included](#what-is-not-included)). To run the pipeline you need your own data in the documented
format.

## Paper status

> [!NOTE]
> **Accepted and presented, not yet published.** The paper was accepted at the
> [9th International Conference on Applied Informatics (ICAI 2026)](https://icai.itiud.org/) and appears in its
> [list of accepted papers](https://icai.itiud.org/?pid=acceptedPapers). It was presented at the conference,
> held on 13–17 October 2026 at the University of Central Florida, Orlando, USA. ICAI proceedings
> will be published with Springer in their
> [Communications in Computer and Information Science (CCIS)](https://www.springer.com/series/7899) series. The
> volume, pages and DOI will be added here once they are available.

Slides of the talk: <https://chusitooxduwu.github.io/icai2026-slides/>

## Background and approach

Introductory programming courses depend on sustained practice and timely feedback. At Universidad de los Andes,
the *Introduction to Programming* course serves more than 1,000 students per semester. It grades Python exercises
with *Senecode*, an automated grader that has processed over 700,000 submissions. Senecode reports which test
failed, but says little about why.

Commercial large language models (LLMs) can explain failures in natural language. However, they have a recurring
cost, and they are not aligned with the course rules: use no constructs beyond the students' current level, and
never reveal the solution. Both problems make them hard to sustain when students can resubmit without limit.
Off-the-shelf small language models (SLMs) are cheap and run locally, but they hallucinate.

The paper asks whether a *domain-adapted SLM* can approach the feedback quality of commercial LLMs while meeting
five design constraints: **non-disclosive**, **course-level**, **reliable**, **cost-efficient** and
**low-compute**. The approach has three parts:

- **Extended unit tests.** Each doctest in an exercise carries a hand-written annotation with its ID, its purpose
  and the tests it builds on. This links every failed test to the requirement it checks.
- **Minimal diagnostic context.** The model receives only the exercise statement, the reference solution, the
  submitted code and the extended unit tests.
- **Adaptation on human-written feedback.** Qwen2.5-Coder-1.5B-Instruct is fine-tuned with QLoRA on 270 synthetic
  failing submissions that cover 50 course exercises. Each submission is paired with feedback that the
  instructional team wrote and peer-reviewed.

## Repository structure

```text
.
├── LICENSE                             MIT License
├── README.md                           this file
├── data_resp.csv                       the 1,000 human evaluations (see "Evaluation data")
└── training/
    ├── README.md                       hyperparameters, data format and run instructions
    ├── entrenar_qwen_qlora.py          QLoRA training script
    ├── inferencia_portable.py          loads the base model and the adapter, and generates feedback
    ├── adapter_config.json             LoRA configuration of the adapter trained for the paper
    ├── PROMPT_ENTRENAMIENTO_QLORA.md   prompt template used to build the training examples (in Spanish)
    └── requirements.txt                Python dependencies
```

## Requirements

- Python 3.10 or later, plus the packages listed in [`training/requirements.txt`](training/requirements.txt):
  PyTorch, Transformers, PEFT, bitsandbytes, Accelerate and Datasets.
- **For training:** an NVIDIA GPU with CUDA and bfloat16 support (Ampere generation or newer). The run reported in
  the paper used an NVIDIA RTX 4070 Ti SUPER with 16 GB of VRAM. The training script stops if it finds no CUDA
  device.
- **For inference:** a CUDA GPU is recommended. Without one, the script runs on the CPU, more slowly.
- Internet access on the first run, to download the base model (about 3 GB) from Hugging Face.

## Installation

```bash
git clone https://github.com/ChusitooXDuwu/Domain-Adapted-Small-Language-Models-for-Course-Aware-Automated-Feedback-in-Introductory-Programming.git
cd Domain-Adapted-Small-Language-Models-for-Course-Aware-Automated-Feedback-in-Introductory-Programming/training
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On some platforms `pip` installs a CPU-only build of PyTorch. In that case, install a CUDA-enabled build first,
following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Usage

Run all the commands below from the `training/` directory. The prompts and the generated feedback are in Spanish,
the language of the course.

### 1. Prepare the training data

The training data are not distributed, so you must provide your own. Place two files in `training/`:

- `train.jsonl`: the training conversations. In the paper, this was 70% of the 270 examples.
- `test.jsonl`: the held-out conversations used for evaluation after each epoch. In the paper, this was the
  remaining 30%.

Each line holds one JSON object with a user turn (the prompt) and an assistant turn (the target feedback):

```json
{"messages": [{"role": "user", "content": "<prompt built from PROMPT_ENTRENAMIENTO_QLORA.md>"}, {"role": "assistant", "content": "• <feedback bullet 1>\n• <feedback bullet 2>"}]}
```

The prompt is the template in [`PROMPT_ENTRENAMIENTO_QLORA.md`](training/PROMPT_ENTRENAMIENTO_QLORA.md) with its
placeholders filled in: the reference solution, the submitted code, and the input, expected output and actual
output of the failed doctest. The target is one or two feedback bullets addressed to the student.

Both files must exist before training starts. The step that built them from the course material is not part of
this repository, so the training script cannot create them. Git ignores `*.jsonl` files, so they will not be committed by accident.

### 2. Train the adapter

```bash
python entrenar_qwen_qlora.py
```

The script does the following:

1. Loads the base model with 4-bit NF4 quantization.
2. Attaches LoRA adapters to all the attention and MLP projections.
3. Trains for three epochs, evaluates on `test.jsonl` after each epoch, and keeps the checkpoint with the lowest
   validation loss.
4. Saves the adapter and the tokenizer to `qwen-socratic-feedback/final/`.

The hyperparameters are constants at the top of the script (see [Configuration](#configuration)).

### 3. Generate feedback

```bash
python inferencia_portable.py
```

The script loads the base model from Hugging Face and the adapter from `./qwen-socratic-feedback/final`. It
generates feedback for a built-in example, then opens an interactive prompt; type `salir`, `exit` or `quit` to
stop. To use an adapter stored elsewhere, change `ADAPTER_PATH` at the top of the script.

The interactive prompt reads a single line. For complete, multi-line prompts, use the `ModeloSocratico` class from
Python:

```python
from inferencia_portable import ModeloSocratico

model = ModeloSocratico(adapter_path="./qwen-socratic-feedback/final")
prompt = "..."  # the template in PROMPT_ENTRENAMIENTO_QLORA.md with its placeholders filled in
print(model.generar(prompt, max_tokens=256, temperature=0.7))
```

Generation uses sampling (`top_p = 0.9`), so the output varies between runs.

## Configuration

The table lists the main training settings. The full list is in [`training/README.md`](training/README.md).

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Quantization | 4-bit NF4 with double quantization, bfloat16 compute |
| LoRA | r = 16, α = 32, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj` |
| Optimizer | `paged_adamw_8bit`, learning rate 2e-4, weight decay 0.01, 50 warmup steps, gradient checkpointing |
| Batch size | 2 per device × 8 gradient-accumulation steps (effective batch size 16) |
| Epochs | 3 |
| Maximum sequence length | 2,048 tokens |
| Hardware used in the paper | NVIDIA RTX 4070 Ti SUPER (16 GB) |

`training/adapter_config.json` is the LoRA configuration saved with the adapter trained for the paper.

## Evaluation data

`data_resp.csv` holds the 1,000 human evaluations behind the [results](#results). Each row is one evaluated
feedback sample. There are 166 or 167 rows per model and 100 rows per reviewer.

| Column | Criterion in the paper | Values |
| --- | --- | --- |
| `response_id` | Sample ID | `resp_1` … `resp_1000` |
| `modelo` | Model that generated the feedback | Model identifier (see [Results](#results)) |
| `friendly` | Supportive | 1–10 |
| `fixes_selective` | Selective | 1–10 |
| `correct` | Accurate | 1–10 (stored with one decimal, e.g. `9.0`) |
| `suggestion` | Non-revealing | 1–10 |
| `conciseness` | Concise | 1–10 |
| `vocabulary_choice` | Novice-appropriate | 1–10 |
| `hallucination_free` | Hallucination-free | `SI` (no fabricated content) or `NO` |
| `revealing` | Collected, not reported in the paper | 1–10 |
| `clarity` | Collected, not reported in the paper | 1–10 |
| `reviewer` | Anonymized reviewer ID | `reviewer_1` … `reviewer_10` |

- The **overall score** of a sample is the mean of the six 1–10 criteria reported in the paper: `friendly`,
  `fixes_selective`, `correct`, `suggestion`, `conciseness` and `vocabulary_choice`.
- The **hallucination-free share** is the proportion of `SI` among the rows marked `SI` or `NO`. Two rows
  (`resp_1` and `resp_2`) contain a number instead of `SI`/`NO`, and they are excluded from this share.
- Reviewer identities are anonymized, and the file contains no student data.

## Results

The models were compared in a single-blind human evaluation. Ten reviewers from the instructional team each rated
100 feedback samples, 1,000 in total across six models. The rating used seven criteria: six scored from 1 to 10
(Supportive, Selective, Accurate, Non-revealing, Concise and Novice-appropriate) and Hallucination-free, rated yes
or no. *Overall* is the mean ± standard deviation of the per-sample overall score. *Hallucination-free* is the
share of samples with no fabricated content.

| Model | Identifier in `data_resp.csv` | Overall (1–10) | Hallucination-free |
| --- | --- | ---: | ---: |
| GPT-4o mini | `gpt-4o-mini-azure` | 9.58 ± 0.42 | 97.6% |
| Claude 3.5 Haiku | `claude-3-5-haiku-20241022` | 9.05 ± 0.81 | 77.0% |
| **Adapted Qwen2.5-Coder-1.5B (this work)** | `qwen2.5-coder:1.5b-finetuned` | **9.01 ± 0.33** | **91.0%** |
| Llama 3.2 1B | `llama3.2:1b` | 6.12 ± 1.50 | 52.1% |
| Qwen2.5-Coder-1.5B (base) | `qwen2.5-coder:1.5b-instruct` | 5.34 ± 1.16 | 21.6% |
| DeepSeek-R1 1.5B | `deepseek-r1:1.5b` | 5.10 ± 1.79 | 44.9% |

- **Effect of adaptation:** it raised the base model's overall score from 5.34 to 9.01 (p < 0.001,
  Cohen's d = 4.30).
- **Comparison with commercial models:** the adapted model shows no statistically significant difference from
  Claude 3.5 Haiku (p = 0.534), while GPT-4o mini remains ahead (p < 0.001).
- **Cost:** on a 16 GB GPU on campus, generating one feedback message costs less than 0.01 US cents in energy.

## What is not included

- The course exercises, the synthetic submissions and the instructional team's feedback that make up the training
  data (`train.jsonl` and `test.jsonl`).
- The trained adapter weights (`qwen-socratic-feedback/`). `training/adapter_config.json` documents the adapter's
  configuration only.
- The base model weights, which are downloaded from Hugging Face at run time.
- API keys or configuration for the commercial baselines.

No authentic student code was used for training. The synthetic submissions were informed only by recurring error
patterns observed in the grader's history.

## Versioning

This project follows [Semantic Versioning](https://semver.org/). The current version is **1.0.0**, the first
public release, which contains the code and data used in the ICAI 2026 paper.

### Release history

- **1.0.0** (2026-11-19): first public release. It includes the QLoRA training and inference scripts, the LoRA
  configuration, the training prompt template and the 1,000 human evaluations.

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for the full text. The license covers
all the contents of this repository, including the code, the configuration files, the prompt template and
`data_resp.csv`.

Third-party components keep their own licenses. The base model
[`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) is released under
the Apache License 2.0; it is downloaded at run time and not redistributed here. Each Python dependency is
distributed under its own license.

## Citation

If you use this software or the evaluation data, please cite the paper. It has been accepted and presented but not
yet published, so the entry below has no volume, pages or DOI. It will be updated when the CCIS proceedings
are published.

```bibtex
@inproceedings{rosales2026domain,
  title     = {Domain-Adapted Small Language Models for Course-Aware Automated Feedback in Introductory Programming},
  author    = {Rosales, Eduardo and Chu, Wyo Hann and Manrique, Rub{\'e}n and S{\'a}nchez, Mario},
  booktitle = {Proceedings of the 9th International Conference on Applied Informatics (ICAI 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026},
  note      = {Accepted and presented at ICAI 2026, Orlando, FL, USA. To appear}
}
```

## Contact

- Eduardo Rosales: <ee.rosales24@uniandes.edu.co>
- Wyo Hann Chu: <w.chu@uniandes.edu.co>
