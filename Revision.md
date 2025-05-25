# 🧠 NLP Exam Revision Notes

---

## 📝 Q1 — Probability Models & N-Grams

**Text:** “data science transfers data into…”

### 1. Chain Rule of Probability

Breaks a joint probability into conditional ones:

**Formula:**

P(w1, w2, w3, w4, w5) =  
P(w1) × P(w2 | w1) × P(w3 | w1, w2) × P(w4 | w1, w2, w3) × P(w5 | w1, w2, w3, w4)

This captures the full dependency between all previous words.

---

### 2. Markov Model (Bigram Assumption)

A simplification of the chain rule assuming only the immediate previous word matters:

P(w1, w2, w3, w4, w5) ≈  
P(w1) × P(w2 | w1) × P(w3 | w2) × P(w4 | w3) × P(w5 | w4)

This is a **first-order Markov model**.

---

### 3. Trigrams

Break the sentence into overlapping groups of three:

- (data, science, transfers)  
- (science, transfers, data)  
- (transfers, data, into)

Used in **language modeling** to capture short-term dependencies.

---

### 4. Numerical Underflow

**Problem:**  
Multiplying many small probabilities results in values close to zero, which can cause computation to underflow.

**Solution:**  
Use **log probabilities**:

log(P1 × P2 × ... × Pn) = log(P1) + log(P2) + ... + log(Pn)

This preserves relative probabilities while preventing underflow.

---

### 5. Zero Counts in N-gram Models

**Problem:**  
If an n-gram never appears in training, its probability is 0 → breaks the entire chain.

**Solutions:**
- **Add-One (Laplace) Smoothing:** Add 1 to all counts  
- **Backoff:** Fall back to smaller n-gram (e.g., bigram → unigram)  
- **Interpolation:** Mix probabilities from multiple n-gram levels  
- **Good-Turing Smoothing:** Adjusts estimates based on the frequency of rare n-grams

---

## 🧮 Q2 — Confusion Matrix & Metrics

### 1. Confusion Matrix for Multi-Class (3 classes)

Construct 3 binary confusion matrices:  
- Class A vs. rest  
- Class B vs. rest  
- Class C vs. rest  

**Each contains:**
- TP: True Positives  
- FP: False Positives  
- FN: False Negatives  
- TN: True Negatives

**Metrics:**
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)

---

### 2. Micro vs. Macro Metrics

- **Macro Precision:**  
  Average precision over all classes  
  `(P1 + P2 + P3) / 3`

- **Micro Precision:**  
  Compute global TP, FP across all classes  
  `Micro Precision = Total TP / (Total TP + Total FP)`

Macro gives equal weight to each class; micro favors frequent classes.

---

## 🔤 Q3 — Sequence Labeling & HMM

### 1. Label Sequencing Tasks

- **Part-of-Speech (POS) Tagging**: Assign tags like noun/verb  
- **Named Entity Recognition (NER)**: Tag people, locations, etc.

---

### 2. Hidden Markov Model (HMM)

**Definition:** A probabilistic model where observable outputs (words) depend on hidden states (tags).

**Components:**
- **States** = tags (e.g., Noun, Verb)  
- **Observations** = words  
- **Transition Probabilities:** P(tag_i | tag_{i−1})  
- **Emission Probabilities:** P(word_i | tag_i)  
- **Initial Probabilities:** P(tag_1)

---

### 3. Viterbi Algorithm (POS Tagging)

**Goal:** Find the most probable tag sequence given a sentence.

Steps:
1. Initialization: For first word, multiply initial × emission
2. Recursion: For each next word, compute best previous path
3. Termination: Pick path with highest total probability
4. Backtrace: Retrieve full tag sequence

Used in real-world taggers like in **spaCy** and **NLTK**.

---

## 🎼 Q5 — First-Order Logic (FOL) with Music

### A. FOL Sentences

1. Mozart composed Moonlight Sonata  
   `composed(Mozart, moonlight_sonata)`

2. Ahmed enjoys Beethoven  
   `enjoys(Ahmed, Beethoven)`

3. Nora enjoys Mozart and not Beethoven  
   `enjoys(Nora, Mozart) ∧ ¬enjoys(Nora, Beethoven)`

4. Moonlight Sonata is Mozart’s piece, not Beethoven’s  
   `pieceOf(moonlight_sonata) = Mozart ∧ pieceOf(moonlight_sonata) ≠ Beethoven`

---

### B. Induction from Logic

Given:

- enjoys(Ahmed, Mozart)  
- ClassicalArtist(Mozart)

**Induced Rule:**  
Ahmed enjoys classical music.

This is **inductive logic** generalizing from specific facts.

---

### C. Logical Domain Definition

- **Domain:** People, Music, Artists  
- **Objects:** Mozart, Beethoven, Ahmed, Nora, moonlight_sonata  
- **Predicates (Relations):**  
  - `composed(x, y)`  
  - `enjoys(x, y)`  
  - `ClassicalArtist(x)`  
- **Functions:**  
  - `pieceOf(x)`

Similar to restaurant FOL examples (Slide 6 analogy).

---

## 🧭 Word Sense Disambiguation (WSD)

### Lesk Algorithm (Baseline)

- Dictionary-based method
- For each sense of a word, compare gloss with context
- Choose sense with **maximum overlap**

**Example:**
> "He sat on the bank of the river."  
Gloss match → bank = river bank (not financial institution)

---

### Extended Lesk

- Includes glosses of surrounding context words
- Improves accuracy by incorporating more context

---

## 🧪 Semi-Supervised Relation Extraction (Bootstrapping)

### Process:

1. Start with a few seed tuples for a target relation  
   e.g., (Barack Obama, USA) for `president_of`
2. Extract surface patterns like:
   - “X, the president of Y”
3. Apply patterns to large corpus to find new pairs
4. Iteratively update both patterns and tuples

---

### Common Issue: Semantic Drift

- Patterns may start collecting irrelevant or noisy data
- Accuracy drops as iterations increase

**Fixes:**
- Limit pattern iterations  
- Score and filter patterns (e.g., by confidence or frequency)  
- Human-in-the-loop validation

---

## 📊 Summary Table

| **Topic**                | **Description**                                                                 |
|--------------------------|----------------------------------------------------------------------------------|
| **Bootstrapping (RE)**   | Starts from a few examples to iteratively discover new entity pairs using surface patterns |
| **Lesk Algorithm (WSD)** | Selects word sense based on overlapping words between context and dictionary definitions |
| **Semantic Drift**       | Deviation from the intended relation due to noisy pattern expansion in bootstrapping |
| **Extended Lesk**        | Uses glosses of surrounding words for improved word sense disambiguation         |

---
## 🔁 RNN & LSTM

### Recurrent Neural Network (RNN)

At time $t$:

$$h_t = f(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)
$$

- Struggles with long-term dependencies due to **vanishing gradients**

---

### Long Short-Term Memory (LSTM)

Handles long-term memory with gates:

#### Gates:

- Forget:  
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
- Input:  
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
- Candidate Cell:
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$
- Update Cell:  
  $$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t $$
- Output:  
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
   $$
- Final output:  
  $$h_t = o_t \cdot \tanh(C_t) $$

---

### Comparison

| Feature             | RNN        | LSTM                        |
|---------------------|------------|-----------------------------|
| Long-term memory    | ❌ Poor     | ✅ Excellent                |
| Vanishing gradients | ❌ Common  | ✅ Mitigated                |
| Speed               | ✅ Faster  | ❌ Slower (more params)     |
| Use cases           | Short seqs | Long seqs (translation etc) |

