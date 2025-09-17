# Agentic Evaluation Framework

## 📝 Project Overview
This project implements a comprehensive evaluation framework for assessing agent-generated responses. It combines various scoring mechanisms such as hallucination detection, instruction alignment, and semantic similarity to compute a composite evaluation score.

---

## 🚀 Project Structure

### ✅ main.py
The entry point of the system. It orchestrates the overall evaluation process by calling key components:
- `hallucination_score.py`
- `instruction_score.py`

---

### ✅ hallucination_score.py
This module leverages advanced Transformer models to assess the quality of agent responses:
- Uses **Microsoft/deberta model** for predicting entailment and contradiction values.
- Incorporates **vector embedding similarity** to compute the correctness score between the agent response and the reference text.

---

### ✅ instruction_score.py
Implements an **agentic approach-based scoring system** to evaluate:
- Instruction alignment
- Readability of the agent response
- Assumptions made in the response

These individual scores are combined to compute a **composite instruction score**.

---

### ✅ generation.py
Generates the evaluation dataset by iterating over multiple agents:
- Runs evaluations for **10 different scenarios** × **5 different agents**
- Automatically stores the results in the `result/` folder for further analysis.

---

## 📂 Results
All generated results are stored in the `result/` directory in structured files for easy inspection and further processing.

---

## ⚡ Getting Started

### Installation
pip install -r requirements.txt
