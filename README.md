# Next Word Predictor (PyTorch RNN)

This Natural Language Processing (NLP) project builds and uses a **GRU=based Recurrent Neural Network (RNN)** in PyTorch to generate next-word suggestions from plain text data.

I built this project to explore core NLP and deep learning concepts including:

- Tokenization
- Vocabulary construction
- Sequence modeling
- Embeddings
- Training / validation / test workflows with RNNs
- Probabilistic next-word prediction

---

## 🚀 Features

✅ Trained on multiple `.txt` books from the Project Gutenberg

✅ Custom tokenizer (no NLTK / spaCy required)  
✅ Automatic vocabulary building  
✅ GRU-based RNN architecture  
✅ Train / Validation split  
✅ Top-K next word suggestions  

---

## 🧠 Model Architecture

The network follows a simple and standard NLP pipeline:

1. **Embedding Layer**  
   Converts word indices into dense vectors

2. **GRU (Gated Recurrent Unit)**  
   Learns sequential language patterns

3. **Dropout**  
   Prevents overfitting

4. **Linear Layer**  
   Maps hidden state → vocabulary probabilities

The model predicts the most likely next word given a sequence of previous words.

---

## Project Structure

next_word_predictor/

- model.py            → GRU-based RNN model
- data_utils.py       → Tokenization & dataset creation
- download_data.py    → Creates the data/ file with training text files (.txt) when run
- train_model.py      → Training and validation script, saves .pth file on run
- text_inferer.py     → Interactive prediction tool
- README.md           → Documentation

---

## Dataset

Training data consists of plain text files stored in a `data/` directory. When running the download_data.py script, the program will create the `data/` directory.

My nlp model is trained on 6 books that were downloaded by Project Gutenberg (thank you!):

- An American Tragedy
- Pride and Prejudice
- The Adventures of Sherlock Holmes
- The Sun Also Rises
- Ulysses
- War and Peace

---

## Takeaways/Notes

This project helped me heavily on applying NLP through the use of a RNN, these are some observations specifically about this project that I had:
- Larger datasets improved predictions but servely affected training time
- Vocabulary size impacted model complexity heavily
- Hard to find extremely accurate models for a project like this since there are many, many ways to create sentences
- There are a ton of different ways to tweak a model to make the training result different

---

## Clone this Repository

To download and run this project locally, clone the repository using Git:

```bash
git clone https://github.com/ryjchen24/next-word-predictor
cd next-word-predictor
pip install torch