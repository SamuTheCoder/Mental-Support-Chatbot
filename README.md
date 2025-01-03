# Mental-Support-Chatbot

This repository encompasses 2 different chatbots for mental health support. The first chatbot developed is a chatbot that uses cosine similarity to provide mental health support. The second chatbot developed uses a Neural Network trained with PyTorch, where the data fed is in Bag of Words (BoW) format.

Both approaches were inspired by 2 different tutorials, which can be found in the following links:

- [Cosine Similarity based Chatbot, by Satyajit Pattnaik](https://www.youtube.com/watch?v=EPzqKkjcnro&pp=ygUZY29zaW5lIHNpbWlsYXJpdHkgY2hhdGJvdA%3D%3D)
- [Chatbot with PyTorch, by Patrick Loeber](https://www.youtube.com/watch?v=RpWeNzfSUHw)

## Usage

For both models, it is recommended to use a virtual environment. To create a virtual environment, run the following command:

```bash
python -m venv venv
```

The activation of the environment is done with:

```bash
source venv/bin/activate
```

Finally, to install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Cosine Similarity Chatbot
Everything needed is in the notebook. For further details, please check the [notebook file](cosine_similarity/notebook.ipynb).

### Neural Network Chatbot

1. Download the nltk punkt package by running the following command on a Python shell:

```bash
nltk.download('punkt')
```

2. If you wish to train the model instead of using the already trained one available in the repository, run the following command:

```bash
python train.py
```

3. To run the chatbot, run the command:

```bash
python chatbot.py
```

## Implementation Details

### Cosine Similarity Chatbot

All details present in the [notebook file](cosine_similarity/notebook.ipynb)

### Neural Network Chatbot

Future section....

The dataset used for the training was obtained at [Kaggle](https://www.kaggle.com/datasets/jiscecseaiml/mental-health-dataset).