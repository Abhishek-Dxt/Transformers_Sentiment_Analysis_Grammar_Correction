# Transformers for Sentiment Analysis and Grammatical-error Correction
Program to analyze sentiment &amp; correct grammatical errors in sample text. Using pre-trained, transformer-based RoBERTa (Robustly Optimized BERT approach) model for sentiment analysis. Additionally, using Hugging Face's Happy Transformer for grammatical error correction &amp; also testing Hugging Face pipelines.

Installing Libraries & Packages -

!pip install -q transformers

`     `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.1/7.1 MB 71.8 MB/s eta 0:00:00      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 224.5/224.5 kB 24.1 MB/s eta 0:00:00      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 57.7 MB/s eta 0:00:00

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from transformers import pipeline

from transformers import AutoTokenizer

from transformers import AutoModelForSequenceClassification from scipy.special import softmax

![](Aspose.Words.82b2d210-9f1c-489f-a688-dfcba58c9976.001.png) Twitter-roBERTa-base for Sentiment Analysis

Below I use the roBERTa-base model trained on ~58M tweets and finenetuned for sentiment analysis with the TweetEval benchmark. This model is suitable for English.

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from\_pretrained(MODEL)

model = AutoModelForSequenceClassification.from\_pretrained(MODEL)

Downloading (…)lve/main/config.json: 100% 747/747 [00:00<00:00, 20.9kB/s] Downloading (…)olve/main/vocab.json: 100% 899k/899k [00:00<00:00, 1.16MB/s] Downloading (…)olve/main/merges.txt: 100% 456k/456k [00:00<00:00, 748kB/s] Downloading (…)cial\_tokens\_map.json: 100% 150/150 [00:00<00:00, 2.07kB/s] Downloading pytorch\_model.bin: 100% 499M/499M [00:02<00:00, 210MB/s]

![](Aspose.Words.82b2d210-9f1c-489f-a688-dfcba58c9976.001.png) Testing the model -

sample = "This was incredibly bad"

def roberta\_senanalyis(inp):

`    `encoded\_text = tokenizer(inp, return\_tensors='pt')     output = model(\*\*encoded\_text)

`    `scores = softmax(output[0][0].detach().numpy())

`    `scores\_dict = {

`        `'negative' : scores[0],

`        `'neutral' : scores[1],

`        `'positive' : scores[2]

`    `}

`    `return scores\_dict

neg, neu, pos = roberta\_senanalyis(sample)['negative'], roberta\_senanalyis(sample)['neutral'], roberta\_senanalyis(sample)['positive'] maxm = max(neg, neu, pos)

print(neg, neu, pos)

0\.9769726 0.020274766 0.0027526724

It can be seen above that the model gives a negative sentiment for the sample input "This was incredibly bad".

![](Aspose.Words.82b2d210-9f1c-489f-a688-dfcba58c9976.001.png) Testing Hugging Face's Sentiment Analysis Pipeline -

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering.

sentiment\_analysis = pipeline("sentiment-analysis")

No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b ([https://huggingfa ](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)![](Aspose.Words.82b2d210-9f1c-489f-a688-dfcba58c9976.002.png)Using a pipeline without specifying a model name and revision in production is not recommended.

Downloading (…)lve/main/config.json: 100% 629/629 [00:00<00:00, 16.0kB/s] Downloading pytorch\_model.bin: 100% 268M/268M [00:01<00:00, 182MB/s] Downloading (…)okenizer\_config.json: 100% 48.0/48.0 [00:00<00:00, 3.28kB/s] Downloading (…)solve/main/vocab.txt: 100% 232k/232k [00:00<00:00, 587kB/s]

Xformers is not installed correctly. If you want to use memorry\_efficient\_attention to accelerate training use the followin pip install xformers.

sentiment\_analysis("I kind of liked that song.") # Positive sentiment

[{'label': 'POSITIVE', 'score': 0.999756395816803}]

sentiment\_analysis("beautiful mess") # Not so perfect (?)

[{'label': 'NEGATIVE', 'score': 0.826908528804779}]

Grammatical Error Correction using Happytransformer from Hugging Face -

!pip install happytransformer

Looking in indexes: <https://pypi.org/simple>, <https://us-python.pkg.dev/colab-wheels/public/simple/>

Collecting happytransformer

`  `Downloading happytransformer-2.4.1-py3-none-any.whl (45 kB)

`     `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.5/45.5 kB 6.3 MB/s eta 0:00:00

Requirement already satisfied: torch>=1.0 in /usr/local/lib/python3.10/dist-packages (from happytransformer) (2.0.1+cu118) Requirement already satisfied: tqdm>=4.43 in /usr/local/lib/python3.10/dist-packages (from happytransformer) (4.65.0)

Requirement already satisfied: transformers>=4.4.0 in /usr/local/lib/python3.10/dist-packages (from happytransformer) (4.29.2) Collecting datasets>=1.6.0 (from happytransformer)

`  `Downloading datasets-2.12.0-py3-none-any.whl (474 kB)

`     `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 474.6/474.6 kB 41.6 MB/s eta 0:00:00

Collecting sentencepiece (from happytransformer)

`  `Downloading sentencepiece-0.1.99-cp310-cp310-manylinux\_2\_17\_x86\_64.manylinux2014\_x86\_64.whl (1.3 MB)

`     `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 74.2 MB/s eta 0:00:00

Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from happytransformer) (3.20.3)

Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (1.22 Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (9 Collecting dill<0.3.7,>=0.3.0 (from datasets>=1.6.0->happytransformer)

`  `Downloading dill-0.3.6-py3-none-any.whl (110 kB)

`     `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 110.5/110.5 kB 15.1 MB/s eta 0:00:00

Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (1.5.3) Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) 

Collecting xxhash (from datasets>=1.6.0->happytransformer)

`  `Downloading xxhash-3.2.0-cp310-cp310-manylinux\_2\_17\_x86\_64.manylinux2014\_x86\_64.whl (212 kB)      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 212.5/212.5 kB 25.7 MB/s eta 0:00:00

Collecting multiprocess (from datasets>=1.6.0->happytransformer)

`  `Downloading multiprocess-0.70.14-py310-none-any.whl (134 kB)

`     `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.3/134.3 kB 18.9 MB/s eta 0:00:00

Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransfo Collecting aiohttp (from datasets>=1.6.0->happytransformer)

`  `Downloading aiohttp-3.8.4-cp310-cp310-manylinux\_2\_17\_x86\_64.manylinux2014\_x86\_64.whl (1.0 MB)

`     `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 69.4 MB/s eta 0:00:00

Requirement already satisfied: huggingface-hub<1.0.0,>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happy Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (23.1) Collecting responses<0.19 (from datasets>=1.6.0->happytransformer)

`  `Downloading responses-0.18.0-py3-none-any.whl (38 kB)

Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (6.0) Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (3.12.0) Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (4.5 Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (1.11.1) Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (3.1) Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (3.1.2) Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (2.0.0) Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.0->happytransformer) ( Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.0->happytransformer) (16 Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.4.0->happytransform Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.4.0 Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=1.6.0->happytransfo Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=1.6.0

Collecting multidict<7.0,>=4.5 (from aiohttp->datasets>=1.6.0->happytransformer)

`  `Downloading multidict-6.0.4-cp310-cp310-manylinux\_2\_17\_x86\_64.manylinux2014\_x86\_64.whl (114 kB)      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 114.5/114.5 kB 16.4 MB/s eta 0:00:00

Collecting async-timeout<5.0,>=4.0.0a3 (from aiohttp->datasets>=1.6.0->happytransformer)

`  `Downloading async\_timeout-4.0.2-py3-none-any.whl (5.8 kB)![](Aspose.Words.82b2d210-9f1c-489f-a688-dfcba58c9976.003.png)

Collecting yarl<2.0,>=1.0 (from aiohttp->datasets>=1.6.0->happytransformer)

`  `Downloading yarl-1.9.2-cp310-cp310-manylinux\_2\_17\_x86\_64.manylinux2014\_x86\_64.whl (268 kB)      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 268.8/268.8 kB 32.1 MB/s eta 0:00:00

from happytransformer import HappyTextToText, TTSettings

happy\_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction") args = TTSettings(num\_beams=5, min\_length=1)

Downloading (…)lve/main/config.json: 100% 1.42k/1.42k [00:00<00:00, 46.2kB/s] Downloading pytorch\_model.bin: 100% 892M/892M [00:09<00:00, 57.8MB/s] Downloading (…)okenizer\_config.json: 100% 1.92k/1.92k [00:00<00:00, 54.4kB/s] Downloading spiece.model: 100% 792k/792k [00:00<00:00, 26.6MB/s] Downloading (…)/main/tokenizer.json: 100% 1.39M/1.39M [00:00<00:00, 3.58MB/s] Downloading (…)cial\_tokens\_map.json: 100% 1.79k/1.79k [00:00<00:00, 45.2kB/s]

result = happy\_tt.generate\_text("grammar: This sentences has has bads grammar", args=args) print(result.text)

This sentence has bad grammar.

def compiler(ip):

`  `neg, neu, pos = roberta\_senanalyis(ip)['negative'], roberta\_senanalyis(ip)['neutral'], roberta\_senanalyis(ip)['positive']   maxm = max(neg, neu, pos)

`  `fixed = happy\_tt.generate\_text("grammar: {}".format(ip),args=args).text

`  `print("Corrected sentence: ", fixed)

`  `print("Mood: ", end='')

`  `if maxm == neg:

`    `print('😒')

`  `elif maxm == neu:

`    `print('😐')

`  `else:

`    `print('😀')

![](Aspose.Words.82b2d210-9f1c-489f-a688-dfcba58c9976.004.png) Testing for mood & grammatical errors in input -

compiler('I mean we all sorts messed') # Negative

Corrected sentence:  I mean, we all sort of messed up. 
Mood: 😒

compiler('What was Goin the clstroom was bad') # Negative

Corrected sentence:  What was going in the classroom was bad. 
Mood: 😒

compiler('That was such a goood times') # Positive

Corrected sentence:  That was such a great time. 
Mood: 😀

compiler('I will go check') # Neutral

Corrected sentence:  I will go check it out. 
Mood: 😐


