---
layout: default
title: "Predicting the CEFR Level of English Texts"
date: 2021-03-14
excerpt: This time I wanted to see if I could predict the CEFR level of a given text. This kind of system is a useful tool for teachers or self-studying students as it helps them find reading material of an appropriate difficulty level.
---

# Attempting to Predict the CEFR Level of English Texts

To try out the final model, check out the [Streamlit app](http://cefr-predictor.ml/). The code is available on [Github](https://github.com/AMontgomerie/CEFR-English-Level-Predictor).

I previously wrote [a blog about automatic reading comprehension question generation](https://amontgomerie.github.io/2020/07/30/question-generator.html). This one is somewhat related, in that it’s another project about English reading comprehension. This time I wanted to see if I could predict the [CEFR level](https://en.wikipedia.org/wiki/Common_European_Framework_of_Reference_for_Languages) of a given text. This kind of system is a useful tool for teachers or self-studying students as it helps them find reading material of an appropriate difficulty level.

There are several tools like this that already exist, so this is mostly just an exercise in trying to reproduce their behaviour.  For example [Duolingo CEFR checker](https://cefr.duolingo.com/) which predicts CEFR at a word level, and then gives an overall score, and [Text Inspector](https://textinspector.com/), which predicts an overall score based on a number of metrics. There are also a number of metrics which aim to estimate the difficulty level of a text, like [the Flesch-Kincaid readability test](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests), [the Gunning Fog index](https://en.wikipedia.org/wiki/Gunning_fog_index), and [the Coleman-Liau index](https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index).

## Dataset
The majority of CEFR levelled reading texts are not freely available, but some free samples can be found. I started by collecting all the freely available labelled sample texts that I could get hold of. The resulting dataset was fairly small, so to increase the size of the dataset, I used the existing CEFR levelling tools to label additional data.

The final dataset contains 1500 example texts split over the 6 CEFR levels. The texts are a mixture of dialogues, stories, articles, and other formats. The dataset can be found [here](https://github.com/AMontgomerie/CEFR-English-Level-Predictor/tree/main/data).

The dataset was then split into 80% training and 20% test.

## Text Complexity Metrics as a Baseline
The [textstat](https://pypi.org/project/textstat/) libraries contains a variety of functions for calculating text readability and complexity metrics, including all the previously mentioned ones. To set a baseline performance, each metric was computed for every example in the test set, and the results were scaled and rounded to fit in the range of labels for classification. Of these metrics, scaled Smog Index performed the best with 41% accuracy on the test set. Most seemed to have some predictive power with regards to CEFR levels, except for Flesch Reading Ease which got less than 13% (below the accuracy of a system which generates a random number between 0 and 5).

| Text Complexity Metric       | Accuracy |
|------------------------------|----------|
| Smog Index                   | 41.8%    | 
| Dale Chall Readability Score | 37.8%    | 
| Automated Readability Index  | 35.1%    | 
| Text Standard                | 34.8%    | 
| Flesch Kincaid Grade         | 34.1%    | 
| Linsear Write Formula        | 33.8%    | 
| Gunning Fog                  | 32.8%    |
| Coleman Liau Index           | 31.8%    | 
| Difficult Words              | 27.1%    |
| Baseline Random              | 16.7%    |
| Flesch Reading Ease          | 12.7%    |

## Feature Engineering

In order to try fitting some classifiers, I needed to generate some features. Since the text complexity metrics individually displayed some level of predictive power, I decided to use them. In addition, I generated some features such as the mean parse tree depth and the mean number of each part-of-speech tag using [spaCy](https://spacy.io/usage/linguistic-features/). Features using the mean were preferred over absolute counts to prevent the level predictions from being directly tied to text length. Higher level texts tend to be longer, but the length of a text by itself is not a good indicator of the text’s difficulty. A short text containing complex sentences filled with obscure terminology is more difficult to read than a longer text of simple short sentences. 

## Training

I tried training SVC, Decision Tree, Random Forest, and XGBoost. At almost 71% accuracy on the test set, XGBoost slightly outperformed the others.

I also also fine-tuned a couple of transformer models, which I initially assumed would be stronger at this kind of language understanding classification task. Pretrained [BERT-base](https://huggingface.co/bert-base-cased) and [DeBERTa-base](https://huggingface.co/microsoft/deberta-base) were fine-tuned for sequence classification. The raw text was tokenised, encoded, and used as inputs into the models. But even after experimenting with various hyperparameters, neither transformer managed to outperform XGBoost. These transformer-based solutions are also significantly more resource intensive than XGBoost, and slower at inference without a GPU.

The training code and model artifacts for the sklearn classifiers can be found [here](https://github.com/AMontgomerie/CEFR-English-Level-Predictor). A Colab notebook for finetuning BERT on the same data can be found [here](https://colab.research.google.com/drive/1rUQkjmr0fwJB_xDhafVXxveyBWex83Dz?usp=sharing). The table below shows the best test set accuracy I was able to get with each model.

| Model                     | Accuracy |
|---------------------------|----------|
| XGBoost                   | 70.9%    |
| DeBERTa-base (pretrained)   | 70.2%    |
| BERT-base-cased (pretrained)| 68.6%    |
| Random Forest             | 68.2%    |
| Logistic Regression       | 67.9%    |
| SVC                       | 67.9%    |

Given these results I went with XGBoost as my final model.

## The Problem of Vague Boundaries

A maximum of 71% accuracy on this 6 class problem isn’t a particularly impressive result. One possible limiting factor is that the data was collected from various sources without a set of consistent rules for labelling. 

Another likely reason is that the criteria for levelling texts are fairly vague, so the boundaries between each class are not clearly defined. [The criteria](https://rm.coe.int/CoERMPublicCommonSearchServices/DisplayDCTMContent?documentId=090000168045bb52) seem to be a set of “can-do” statements for each level, such as “can understand texts that consist mainly of high frequency everyday or job-related language” (B1). It’s not clear exactly which vocabulary is included in “high frequency everyday or job-related language”, or how much of text must consist of this to be considered “mainly”.

#### Confusion Matrix

| label | A1 | A2 | B1 | B2 | C1 | C2 |
|-------|----|----|----|----|----|----|
| A1    | 52 | 5  | 1  | 0  | 0  | 0  |
| A2    | 13 | 40 | 1  | 1  | 0  | 0  |
| B1    | 0  | 5  | 23 | 12 | 1  | 0  |
| B2    | 0  | 2  | 9  | 32 | 13 | 1  |
| C1    | 0  | 0  | 1  | 9  | 34 | 4  |
| C2    | 0  | 0  | 0  | 0  | 9  | 31 |

The confusion matrix above confirms that the majority of the model's incorrect predictions are one-off misclassifications. The model frequently confuses A1 and A2 for example, but rarely confuses a label with anything other than its immediate neighbour. This seems to confirm the idea that the boundaries are not easy to distinguish. For top-2 accuracy the model scored 95%.

## Using Probabilities to Distinguish Between Labels

In cases where a text seems to lie somewhere between B1 and B2, we can call the text B1+ to indicate that it’s somewhere in the middle. Text Inspector seems to take the same approach in these cases. However,  the + labels are not present in the dataset which I collected, so to avoid having to completely relabel the data, the model’s predicted probabilities for each label are used. 

Predictions where the maximum probability is below a certain threshold (0.7) are counted as instances of the model being uncertain. In these cases the predicted label is the average of the max and the second strongest prediction. For example, in the case that the model predicts somewhere between 0 (A1) and 1 (A2), the returned value will now be 0.5 (A1+). This indicates that the text could belong in either A1 or A2, and might be appropriate for advanced A1 level readers, or A2 level readers.

## Improvements

I think the most significant way to improve this model would be to collect a new dataset with a more precise set of rules for labelling. This could be done by only taking all labelled texts from one source. For example [the British Council site](https://learnenglish.britishcouncil.org/skills/reading/) has a set of texts which presumably follow a consistent set of rules for labelling. However the number of texts is fairly small, and I don’t know of any publicly available source of labelled texts which would be large enough for the task.
