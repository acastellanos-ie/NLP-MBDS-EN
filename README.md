# Natural Language Processing

This repository contains the notebooks used in class. They are built as live demonstrations: the code is ready, but the result is revealed, checked and discussed as the notebook runs.

The notebooks are not a sequence of coding exercises. Each one starts from a concrete question, changes one thing at a time and finishes by testing where the method breaks.

## Course map

| Session | Notebook | Main question |
| --- | --- | --- |
| Linguistic structure | [Dependency parsing](03_linguistic_structure_and_interpretability/dependency_parsing.ipynb) | What does a dependency parser recover, and how stable is the analysis? |
| Linguistic structure | [Attention](03_linguistic_structure_and_interpretability/attention.ipynb) | What changes inside a transformer when context changes? |
| Semantics | [Sentence embeddings](04_semantics/sentence_embeddings.ipynb) | When does semantic similarity help, and what can it fail to distinguish? |
| Language modelling | [Next-token prediction](05_language_modeling/language_modeling.ipynb) | How do local token probabilities become generated text? |
| Transfer learning | [Fine-tuning a classifier](06_transfer_learning_and_fine_tuning/transfer_learning_and_fine_tuning.ipynb) | Does a fine-tuned transformer beat a strong lexical baseline? |
| Retrieval | [Sparse and dense retrieval](07_rag/information_retrieval.ipynb) | How do lexical and semantic retrieval fail differently? |
| Question answering | [Extractive QA](07_rag/question_answering.ipynb) | What happens when the answer is absent or the context is wrong? |
| RAG | [RAG step by step](07_rag/rag_step_by_step.ipynb) | Which part of a RAG answer came from retrieval and which part came from generation? |
| Agentic AI | [Agent loops and tools](08_agentic_ai/agentic_ai.ipynb) | What turns a model call into a controlled loop with tools? |
| Course synthesis | [One support case, end to end](09_course_synthesis/support_case_end_to_end.ipynb) | How do the individual NLP components become one inspectable system? |

The three retrieval notebooks are meant to run in that order: retrieval first, then reader behaviour, then the complete RAG pipeline.

## Data

The disaster-tweet files used in the transfer-learning practice come from Kaggle's [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) competition and remain subject to its competition rules.

## Running the notebooks

Each notebook has an **Open in Colab** button at the top and installs its direct dependencies. Select a GPU runtime for the fine-tuning, agentic AI and course-synthesis notebooks. DistilBERT fine-tuning also runs on CPU, but its three training epochs take substantially longer; the local Qwen examples require a GPU.
