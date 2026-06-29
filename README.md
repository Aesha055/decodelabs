# Project 1: Rule-Based AI Chatbot

**DecodeLabs — AI Engineer Internship Track**

## Overview

This project implements a **deterministic, rule-based chatbot** — the foundational "Logic Engine" that precedes probabilistic AI systems like neural networks. Every response is hard-coded and traceable: `Input -> Logic -> Output`, with zero hallucination risk.

This is "System 2" thinking in AI terms — the engineer, not the artist. No probabilities, no black box. Just clean, predictable logic.

## Why Rule-Based First?

Before building probabilistic models (like the classifier in Project 2), it's essential to understand the deterministic foundation that modern AI guardrails still rely on:

- **Traceability** — every input maps to exactly one output, no mystery
- **Safety** — zero hallucination risk, 100% hard-coded behavior
- **Compliance** — essential for regulated domains like finance & healthcare
- Rule-based systems still act as the **guardrail layer** sitting in front of modern LLMs in production

## Architecture

The chatbot follows the **IPO Model** (Input → Process → Output):

```
USER INPUT
    │
    ▼
SANITIZATION  (lowercase + strip whitespace)
    │
    ▼
KNOWLEDGE BASE LOOKUP  (dict.get() — match + fallback in one operation)
    │
    ▼
RESPONSE OUTPUT
```

### Key Components

| Component | Implementation |
|-----------|----------------|
| **Input Loop** | Continuous `while True` cycle |
| **Sanitization** | `.strip().lower()` — handles case & whitespace |
| **Knowledge Base** | Python dictionary with 5+ intents |
| **Fallback** | Default "I do not understand" response for unknown input |
| **Exit Strategy** | Clean `break` on `bye` / `exit` / `quit` |

### The `.get()` Method

The core of the bot's logic is a single line:

```python
responses.get(cleaned, DEFAULT_RESPONSE)
```

This handles **lookup and fallback in one atomic operation** — no nested `if/elif` chains needed. If the key exists, return its value; otherwise, return the default. This is the "Professional Approach" referenced in the design slides, vs. a naive long if-else ladder.

## Requirements

No external libraries required — pure Python standard library only.

```bash
python3 --version   # Python 3.7+
```

## How to Run

### Interactive mode (local terminal)
```bash
python3 project1_chatbot.py
```
You'll get a live `You:` prompt. Type naturally, and type `bye`, `exit`, or `quit` to end the session.

### Demo mode (non-interactive environments, e.g. sandboxes/notebooks)
If no terminal input is detected, the script automatically runs a simulated conversation (`run_demo()`) so you can still see it work end-to-end without manual typing.

## Sample Conversation

```
You: Hi
DecodeBot: Hello! How can I help you today?

You: what is your name
DecodeBot: I'm DecodeBot, a rule-based AI assistant.

You:   HOW ARE YOU
DecodeBot: I'm just a bunch of if-else statements, but I'm doing great! How about you?

You: tell me a joke
DecodeBot: I do not understand. Try typing 'help' to see what I can do.

You: bye
DecodeBot: Goodbye! Have a great day!
```

Note how `"  HOW ARE YOU  "` (mixed case, extra whitespace) still matches correctly — that's sanitization at work. And `"tell me a joke"` has no matching rule, so it falls back gracefully instead of crashing.

## Knowledge Base (Current Intents)

| Trigger | Response Category |
|---------|-------------------|
| `hi`, `hello`, `hey` | Greetings |
| `how are you` | Small talk |
| `what is your name`, `who are you` | Identity |
| `what can you do`, `help` | Capabilities |
| `thanks`, `thank you` | Gratitude |
| `bye` | Exit |

## Project Structure

```
.
├── project1_chatbot.py   # Main script — full rule-based chatbot
└── README.md             # This file
```

## Key Concepts Demonstrated

- **Deterministic logic** — same input always produces the same output
- **Input sanitization** — normalizing data before processing
- **Dictionary-based pattern matching** — the simplest form of an "intents" knowledge base
- **Graceful fallback handling** — never letting unknown input break the program
- **Clean program exit strategy** — controlled termination instead of force-quitting

## Limitations (By Design)

This is intentionally **not** an NLP system. It does:
- ❌ No partial matching, typo tolerance, or synonym detection
- ❌ No context/memory between turns
- ❌ No probability or learning

These limitations are the exact gap that **Project 2 (Supervised Learning Classification)** and future LLM-based systems are designed to close — rule-based logic as the floor, not the ceiling.

## Next Steps / Extensions

- Add keyword/substring matching instead of exact match (e.g. "hi there" → matches "hi")
- Add a hybrid architecture: rule match first, fall back to an LLM only when no rule matches (as shown in the "Hybrid Architecture" design slide)
- Track conversation history for context-aware responses
- Expand the knowledge base with more intents


# Project 2: Data Classification Using AI

**DecodeLabs — AI Engineer Internship Track**

## Overview

This project implements a basic **supervised learning classification pipeline** from scratch logic using scikit-learn. It demonstrates how a machine learning model can learn patterns from labeled data and use them to classify new, unseen examples.

The project uses the classic **Iris flower dataset** — 150 samples of iris flowers, each described by 4 measurements, belonging to one of 3 species.

## What This Project Covers

| Step | Description |
|------|-------------|
| 1. Load & understand the dataset | Inspect shape, features, class balance, and summary statistics |
| 2. Train/test split | 80% training, 20% testing, stratified split with a fixed random seed |
| 3. Apply classification algorithms | Trains and compares **K-Nearest Neighbors** and **Decision Tree** classifiers |
| 4. Evaluate the model | Accuracy, precision/recall/F1 score, and a confusion matrix |
| 5. Predict on new data | Tests the best model against 3 brand-new, unseen flower samples |

## Dataset

- **Source:** `sklearn.datasets.load_iris()` (built-in, no download required)
- **Samples:** 150 (50 per class)
- **Features:** sepal length, sepal width, petal length, petal width (all in cm)
- **Classes:** `setosa`, `versicolor`, `virginica`

## Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install pandas scikit-learn
```

## How to Run

### Option A — Locally
```bash
python3 project2_classification.py
```

### Option B — Google Colab
1. Open a new Colab notebook.
2. Paste in the script (no file upload needed — the Iris dataset is built into scikit-learn).
3. Run all cells.

> Note: Unlike a custom CSV dataset, this script does **not** require uploading any files — the Iris dataset loads directly from scikit-learn.

## Sample Output

```
Best model: K-Nearest Neighbors (k=3) with accuracy 100.00%

Confusion Matrix (rows = actual, cols = predicted):
                   pred_setosa  pred_versicolor  pred_virginica
actual_setosa               10                0               0
actual_versicolor            0               10               0
actual_virginica             0                0              10
```

## Results Summary

| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors (k=3) | 100.00% |
| Decision Tree | 93.33% |

KNN performed best on this run, achieving perfect classification on the held-out test set. The Decision Tree slightly confused `versicolor` and `virginica`, which are the two species with the most overlapping measurements.

## Project Structure

```
.
├── project2_classification.py   # Main script — full ML pipeline
└── README.md                    # This file
```

## Key Concepts Demonstrated

- **Supervised learning** — training a model on labeled data
- **Train/test split** — evaluating generalization on unseen data
- **Classification algorithms** — KNN (instance-based) vs. Decision Tree (rule-based)
- **Model evaluation** — accuracy, precision, recall, F1-score, confusion matrix
- **Inference** — using a trained model to predict on brand-new data

## Next Steps / Extensions

- Swap in a custom dataset (e.g. `lds1.csv`) instead of Iris
- Add data visualization (scatter plots, decision boundaries)
- Try additional algorithms (Logistic Regression, SVM, Random Forest)
- Add cross-validation instead of a single train/test split
- Tune hyperparameters (e.g. `k` in KNN, `max_depth` in Decision Tree)

## Author

DecodeLabs AI Engineering Internship — Project 2 Milestone
