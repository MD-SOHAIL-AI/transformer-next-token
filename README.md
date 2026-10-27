# Transformer — Next-Token Prediction

[View the notebook](Transformer_Next_Token_Colab.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/MD-SOHAIL-AI/transformer-next-token/blob/main/Transformer_Next_Token_Colab.ipynb)

## Goal
Train a character-level Transformer to predict the next character in a sequence using causal self-attention.

## Dataset
Tiny Shakespeare from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt . It is automatically downloaded when internet access is available. Alternatively place/upload `input.txt` beside the notebook.

## Method
Encode characters as IDs, split the text into train/validation/test, train a two-block Transformer using query/key/value attention, next-character cross-entropy, and autoregressive text generation. The causal attention mask blocks access to future characters. Settings are kept small enough for CPU execution.

## Outputs
The notebook contains loss plots, sample generated text and a causal attention mask. **The pre-filled output is from a short offline test with a repeated text excerpt, not the full Tiny Shakespeare dataset.** It is proof that the code runs, not a valid full-dataset benchmark. Re-run with the real dataset for your GitHub submission.

## Run
Open the notebook in Google Colab, connect to a CPU runtime and select **Runtime > Run all**. Internet access is needed for the automatic dataset download. If downloads are unavailable, upload the Tiny Shakespeare `input.txt` beside the notebook before running it.

For a local Python environment:

```bash
python -m pip install torch numpy matplotlib jupyter
jupyter notebook Transformer_Next_Token_Colab.ipynb
```

Run the cells in order. The default training run uses up to the first 300,000 characters, with an 80/10/10 train/validation/test split and 160 training steps. It saves weights to `outputs/transformer_next_token.pt` locally or `/kaggle/working/transformer_next_token.pt` on Kaggle.

## GitHub
Commit the saved executed notebook and README. Dataset text and .pt model checkpoints do not need to be committed.
