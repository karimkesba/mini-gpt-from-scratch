# MiniGPT From Scratch with PyTorch

A hands-on learning project exploring the fundamentals of GPT-style language models by building a small Transformer-based language model from scratch using PyTorch.

The project contains two experiments that demonstrate the progression from learning the core architecture on a tiny custom dataset to training a character-level language model on a larger real-world text corpus.

## Project Structure

```text
mini-gpt-from-scratch/
├── README.md
├── 01_mini_gpt_from_scratch.ipynb
└── 02_mini_gpt_tiny_shakespeare.ipynb
```

## Experiment 1: MiniGPT on a Tiny Custom Dataset

### Overview

The first experiment focuses on understanding how a small GPT-style model processes text, learns token relationships, and predicts the next token.

Instead of starting with a large dataset, the experiment uses a small, manually created English text corpus. This makes it easier to understand the model's internal components and the next-token prediction process.

### Dataset

The experiment uses the following toy text corpus:

```text
i love machine learning
i love deep learning
machine learning is interesting
```

The text is tokenized at the word level, and each unique word is mapped to an integer ID.

The model learns to predict the next word based on the preceding words in a sequence.

### Model Architecture

The model consists of the following components:

- **Token Embeddings:** Convert token IDs into learnable vector representations.
- **Positional Embeddings:** Represent the position of each token in the input sequence.
- **Self-Attention:** Allows the model to learn relationships between tokens while processing the input.
- **Feed-Forward Network:** Applies additional transformations to the token representations.
- **Transformer Blocks:** Two stacked blocks combine attention and feed-forward processing.
- **Output Layer:** Produces scores for the possible next tokens.

The model is implemented using PyTorch and trained from randomly initialized weights.

### Training Configuration

| Parameter | Value |
|---|---:|
| Training epochs | 300 |
| Optimizer | Adam |
| Learning rate | 0.003 |
| Loss function | Cross-Entropy Loss |
| Transformer blocks | 2 |

The training process teaches the model to predict the next word from the available context.

### Text Generation

After training, the model generates additional text from a user-provided prompt.

Example prompt:

```text
i love
```

The model uses the prompt as its starting context and generates 10 additional words autoregressively.

Because the dataset is extremely small, the experiment is intended to demonstrate the mechanics of language modeling rather than produce reliable or diverse natural language.

### Learning Outcomes

This experiment introduces the fundamental concepts behind GPT-style language models:

- Mapping words to token IDs.
- Learning token and positional embeddings.
- Understanding self-attention and Transformer blocks.
- Training a model for next-token prediction.
- Calculating cross-entropy loss and updating model parameters.
- Generating text from a prompt.

---

## Experiment 2: MiniGPT on Tiny Shakespeare

### Overview

The second experiment extends the concepts from the first notebook to a larger text corpus.

A small character-level GPT-style model is trained on the Tiny Shakespeare dataset to learn character patterns and generate Shakespeare-inspired text.

Unlike the first experiment, which predicts words, this model predicts one character at a time.

### Dataset

Dataset: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

| Property | Value |
|---|---:|
| Total characters | 1,115,394 |
| Vocabulary size | 65 unique characters |
| Training split | 90% |
| Validation split | 10% |

Each character is mapped to an integer ID before training.

### Model Architecture

The model consists of:

1. **Input Embeddings:** Learnable character embeddings combined with learnable positional embeddings.
2. **Causal Multi-Head Attention:** Implemented using PyTorch's `nn.MultiheadAttention` with a causal mask that prevents access to future positions.
3. **Feed-Forward Network:** Two linear layers with ReLU activation.
4. **Layer Normalization and Residual Connections:** Used within each Transformer block.
5. **Transformer Blocks:** Two stacked blocks process the input representations.
6. **Output Layer:** Produces scores for all 65 possible characters at each position.

The model is trained to predict the next character using cross-entropy loss.

### Training Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Context length | 128 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Total training steps | 5,000 |
| Loss function | Cross-Entropy Loss |

### Training Results

| Metric | Initial | After 5,000 steps |
|---|---:|---:|
| Training Loss | 4.3228 | 1.5321 |
| Validation Loss | 4.3114 | 1.7161 |

The decreasing training and validation losses indicate that the model learned character-level patterns and improved its next-character predictions on held-out data.

### Prompt-Based Text Generation

The model supports text generation from a user-provided prompt.

Example prompt:

```text
ROMEO:
```

The model continues generating characters based on the learned patterns in the training corpus.

The generated text imitates some aspects of Shakespearean dialogue, including character names and dramatic formatting. However, it still contains misspelled words and incoherent sentences.

This is expected from a relatively small model trained for a limited number of steps.

### Learning Outcomes

This experiment demonstrates how to:

- Prepare and encode a larger text dataset.
- Create training and validation splits.
- Generate batches for next-character prediction.
- Implement causal multi-head attention.
- Combine embeddings, attention, feed-forward networks, and residual connections.
- Train a GPT-style model using backpropagation and AdamW.
- Monitor training and validation loss.
- Generate text from a custom prompt.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Google Colab

## Limitations

This repository is an educational implementation of small GPT-style language models, not a reproduction of the scale or capabilities of ChatGPT.

The first experiment uses a very small custom dataset to make the core concepts easier to understand. The second experiment uses a larger corpus but still has limited model capacity and training.

Generated text may be grammatically incorrect, repetitive, or semantically incoherent.

## Future Improvements

- Experiment with model size and context length.
- Train for additional steps and compare validation performance.
- Explore alternative text-generation sampling strategies.
- Compare word-level and character-level language modeling.
- Evaluate how architecture and training choices affect generated text.

## How to Run

1. Clone or download this repository.
2. Open either notebook in Google Colab or a local Jupyter environment.
3. Execute the notebook cells in order.
4. For the Tiny Shakespeare experiment, download the dataset or provide the text file locally.
5. Train the model and experiment with different prompts.

## Disclaimer

This project is intended for educational purposes. Its primary goal is to understand the building blocks of GPT-style language models through practical implementation, rather than to achieve production-level text generation.