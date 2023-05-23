# Transformers for Sentiment Analysis and Grammatical-error Correction
Program to analyze sentiment &amp; correct grammatical errors in sample text. Using pre-trained, transformer-based RoBERTa (Robustly Optimized BERT approach) model for sentiment analysis. Additionally, using Hugging Face's Happy Transformer for grammatical error correction &amp; also testing Hugging Face pipelines.

#### Installing Libraries & Packages -

```
!pip install -q transformers
```
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.1/7.1 MB 71.8 MB/s eta 0:00:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 224.5/224.5 kB 24.1 MB/s eta 0:00:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 57.7 MB/s eta 0:00:
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
```
#### Below I use the roBERTa-base model trained on ~58M tweets and netuned for sentiment analysis with the TweetEval benchmark. This model

#### is suitable for English.

## Twitter-roBERTa-base for Sentiment Analysis

```
Downloading (...)lve/main/config.json: 100% 747/747 [00:00<00:00, 20.9kB/s]
```
```
Downloading (...)olve/main/vocab.json: 100% 899k/899k [00:00<00:00, 1.16MB/s]
```
```
Downloading (...)olve/main/merges.txt: 100% 456k/456k [00:00<00:00, 748kB/s]
```
```
Downloading (...)cial_tokens_map.json: 100% 150/150 [00:00<00:00, 2.07kB/s]
```
```
Downloading pytorch_model.bin: 100% 499M/499M [00:02<00:00, 210MB/s]
```
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

### Testing the model -

sample = "This was incredibly bad"
def roberta_senanalyis(inp):
    encoded_text = tokenizer(inp, return_tensors='pt')
    output = model(**encoded_text) #**
    scores = softmax(output[ 0 ][ 0 ].detach().numpy())
    scores_dict = {
    'negative' : scores[ 0 ],
    'neutral' : scores[ 1 ],
    'positive' : scores[ 2 ]
    }
    return scores_dict

neg, neu, pos = roberta_senanalyis(sample)['negative'], roberta_senanalyis(sample)['neutral'], roberta_senanalyis(sample)['positive']
maxm = max(neg, neu, pos)
print(neg, neu, pos)

```
0.9769726 0.020274766 0.
```
#### It can be seen above that the model gives a negative sentiment for the sample input "This was incredibly bad".

#### The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from

#### the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment

#### Analysis, Feature Extraction and Question Answering.

### Testing Hugging Face's Sentiment Analysis Pipeline -

sentiment_analysis = pipeline("sentiment-analysis")


```
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingfa
Using a pipeline without specifying a model name and revision in production is not recommended.
```
```
Xformers is not installed correctly. If you want to use memorry_efficient_attention to accelerate training use the followin
pip install xformers.
```
```
Downloading (...)lve/main/config.json: 100% 629/629 [00:00<00:00, 16.0kB/s]
```
```
Downloading pytorch_model.bin: 100% 268M/268M [00:01<00:00, 182MB/s]
```
```
Downloading (...)okenizer_config.json: 100% 48.0/48.0 [00:00<00:00, 3.28kB/s]
```
```
Downloading (...)solve/main/vocab.txt: 100% 232k/232k [00:00<00:00, 587kB/s]
```
sentiment_analysis("I kind of liked that song.") # Positive sentiment

```
[{'label': 'POSITIVE', 'score': 0.999756395816803}]
```
sentiment_analysis("beautiful mess") # Not so perfect (?)

```
[{'label': 'NEGATIVE', 'score': 0.826908528804779}]
```
## Grammatical Error Correction using Happytransformer from Hugging Face -

!pip install happytransformer

```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting happytransformer
Downloading happytransformer-2.4.1-py3-none-any.whl (45 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.5/45.5 kB 6.3 MB/s eta 0:00:
Requirement already satisfied: torch>=1.0 in /usr/local/lib/python3.10/dist-packages (from happytransformer) (2.0.1+cu118)
Requirement already satisfied: tqdm>=4.43 in /usr/local/lib/python3.10/dist-packages (from happytransformer) (4.65.0)
Requirement already satisfied: transformers>=4.4.0 in /usr/local/lib/python3.10/dist-packages (from happytransformer) (4.29.2)
Collecting datasets>=1.6.0 (from happytransformer)
Downloading datasets-2.12.0-py3-none-any.whl (474 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 474.6/474.6 kB 41.6 MB/s eta 0:00:
Collecting sentencepiece (from happytransformer)
Downloading sentencepiece-0.1.99-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 74.2 MB/s eta 0:00:
Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from happytransformer) (3.20.3)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (1.
Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (
Collecting dill<0.3.7,>=0.3.0 (from datasets>=1.6.0->happytransformer)
Downloading dill-0.3.6-py3-none-any.whl (110 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 110.5/110.5 kB 15.1 MB/s eta 0:00:
Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (1.5.3)
Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer)
Collecting xxhash (from datasets>=1.6.0->happytransformer)
Downloading xxhash-3.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 212.5/212.5 kB 25.7 MB/s eta 0:00:
Collecting multiprocess (from datasets>=1.6.0->happytransformer)
Downloading multiprocess-0.70.14-py310-none-any.whl (134 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.3/134.3 kB 18.9 MB/s eta 0:00:
Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransfo
Collecting aiohttp (from datasets>=1.6.0->happytransformer)
Downloading aiohttp-3.8.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.0 MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 69.4 MB/s eta 0:00:
Requirement already satisfied: huggingface-hub<1.0.0,>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happy
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (23.1)
Collecting responses<0.19 (from datasets>=1.6.0->happytransformer)
Downloading responses-0.18.0-py3-none-any.whl (38 kB)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets>=1.6.0->happytransformer) (6.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (3.12.0)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (4.
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (1.11.1)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (3.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (3.1.2)
Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.0->happytransformer) (2.0.0)
Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.0->happytransformer) (
Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.0->happytransformer) (
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.4.0->happytransform
Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.4.
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=1.6.0->happytransfo
Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=1.6.
Collecting multidict<7.0,>=4.5 (from aiohttp->datasets>=1.6.0->happytransformer)
Downloading multidict-6.0.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (114 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 114.5/114.5 kB 16.4 MB/s eta 0:00:
Collecting async-timeout<5.0,>=4.0.0a3 (from aiohttp->datasets>=1.6.0->happytransformer)
```

```
Downloading async_timeout-4.0.2-py3-none-any.whl (5.8 kB)
Collecting yarl<2.0,>=1.0 (from aiohttp->datasets>=1.6.0->happytransformer)
Downloading yarl-1.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (268 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 268.8/268.8 kB 32.1 MB/s eta 0:00:
```
from happytransformer import HappyTextToText, TTSettings

```
Downloading (...)lve/main/config.json: 100% 1.42k/1.42k [00:00<00:00, 46.2kB/s]
```
```
Downloading pytorch_model.bin: 100% 892M/892M [00:09<00:00, 57.8MB/s]
```
```
Downloading (...)okenizer_config.json: 100% 1.92k/1.92k [00:00<00:00, 54.4kB/s]
```
```
Downloading spiece.model: 100% 792k/792k [00:00<00:00, 26.6MB/s]
```
```
Downloading (...)/main/tokenizer.json: 100% 1.39M/1.39M [00:00<00:00, 3.58MB/s]
```
```
Downloading (...)cial_tokens_map.json: 100% 1.79k/1.79k [00:00<00:00, 45.2kB/s]
```
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

args = TTSettings(num_beams= 5 , min_length= 1 )

result = happy_tt.generate_text("grammar: This sentences has has bads grammar", args=args)

print(result.text)

```
This sentence has bad grammar.
```
def compiler(ip):
    neg, neu, pos = roberta_senanalyis(ip)['negative'], roberta_senanalyis(ip)['neutral'], roberta_senanalyis(ip)['positive']
    maxm = max(neg, neu, pos)
    fixed = happy_tt.generate_text("grammar: {}".format(ip),args=args).text
    print("Corrected sentence: ", fixed)
    print("Mood: ", end='')
    if maxm == neg:
        print('😒')
    elif maxm == neu:
        print('😐')
    else:
        print('😀')

# Testing for mood & grammatical errors in input -

```
compiler('I mean we all sorts messed') # Negative
```
```
Corrected sentence: I mean, we all sort of messed up.
Mood: 😒
```
compiler('What was Goin the clstroom was bad') # Negative

```
Corrected sentence: What was going in the classroom was bad.
Mood: 😒
```
compiler('That was such a goood times') # Positive

```
Corrected sentence: That was such a great time.
Mood: 😀
```
compiler('I will go check') # Neutral

```
Corrected sentence: I will go check it out.
Mood: 😐
```
