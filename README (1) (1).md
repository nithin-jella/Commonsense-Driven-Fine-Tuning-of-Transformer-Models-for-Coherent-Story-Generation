# Commonsense-Driven Fine-Tuning of Transformer Models for Coherent Story Generation

**EDS 6352 – Natural Language Processing, Fall 2025 | Team 14**
Saketh Reddy CH · Nithin Jella · Kiarah Dinesh Patel

## Project Overview

This project builds a generative AI system that completes short stories by predicting a coherent final sentence, given the first four sentences of a story. We fine-tune pre-trained transformer language models using **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning, and compare model performance using both automatic and human evaluation metrics.

## Dataset

- **Dataset:** [ROCStories](https://huggingface.co/datasets/mintujupally/ROCStories) (via HuggingFace `datasets`)
- **Structure:** ~100K five-sentence stories
  - **Input:** First 4 sentences (story context)
  - **Target:** 5th sentence (story ending)
- **Split:** Train: 77,688 | Test: 19,410

The dataset is loaded directly from HuggingFace at runtime, so it does not need to be uploaded to this repository.

## Approach

1. **Preprocessing:** Each story is split into sentences; the first four form the `story_context` and the fifth forms the `story_ending` target.
2. **Tokenization:** Context and ending are tokenized and concatenated for causal language modeling, with padding/truncation to a fixed max length and `-100` labels on padding tokens to exclude them from the loss.
3. **Fine-tuning:** The base model (e.g., GPT-2) is wrapped with a LoRA adapter (`peft`) targeting the attention layers, then fine-tuned with the Hugging Face `Trainer` API using early stopping on validation loss.
4. **Generation:** Story endings are generated using several decoding strategies — greedy decoding, temperature sampling, top-p (nucleus) sampling, and beam search.
5. **Evaluation:**
   - **Automatic metrics:** BLEU, ROUGE-L, BERTScore
   - **Human evaluation:** A sampled subset of generated endings is rated on fluency, relevance, and creativity (1–5 scale)
   - **Correlation analysis:** Pearson correlation between human scores and BERTScore, plus MAE/RMSE comparisons and outlier analysis

## Repository Structure

```
.
├── notebooks/
│   ├── GPT_2_Final.ipynb         # GPT-2 + LoRA fine-tuning and evaluation
│   ├── DistilGPT_Final.ipnb      # DistilGPT + LoRA fine-tuning and evaluation
    └── GPT_Neo_Final.ipnb        # GPT-Neo + LoRA fine-tuning and evaluation              
├── data/
│   └── processed/                # Preprocessed train/test splits (generated at runtime)
├── evaluation/
│   ├── results/                  # Generated predictions (e.g., predictions_top_p_0.95.jsonl)
│   └── human_evaluation_template.csv
├── results/
│   ├── summary_metrics.json      # BLEU, ROUGE-L, BERTScore, correlation, MAE/RMSE
│   ├── human_vs_model_scatter.png
│   └── difference_barplot.png
├── requirements.txt
└── README.md
```

> Note: `data/`, `evaluation/results/`, and `results/` folders are generated when the notebooks are run and are not tracked in version control (see `.gitignore`).

## Models

Each notebook follows the same pipeline (load data → preprocess → tokenize → LoRA fine-tune → generate → evaluate), applied to a different base transformer model:

| Notebook | Base Model |
|----------|------------|
| `GPT_2_Final.ipynb` | GPT-2 |
| ... | *(add additional model notebooks here)* |

## Setup & Usage

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run a notebook (e.g., in Jupyter or Google Colab):
   ```bash
   jupyter notebook notebooks/GPT_2_Final.ipynb
   ```

   A GPU is strongly recommended for training (the notebooks check for CUDA availability and enable mixed-precision training).

## Results

Automatic and human evaluation results (BLEU, ROUGE-L, BERTScore, human fluency/relevance/creativity scores, and correlation analysis) are saved to the `results/` directory after running the evaluation cells. See `results/summary_metrics.json` for a consolidated summary.

## Acknowledgements

- ROCStories dataset via HuggingFace (`mintujupally/ROCStories`)
- Hugging Face `transformers`, `datasets`, `peft`, and `evaluate` libraries
