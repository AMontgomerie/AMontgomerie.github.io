---
layout: default
title: "Token Classification With Sub-word Tokenizers for Bulgarian"
date: 2020-09-02
excerpt: The amount of tools for Bulgarian language learners seems pretty limited, so I thought I’d try building my own. I wanted to know what the individual words in the sentences I Google-translating were doing, so I decided to train a part-of-speech (POS) tagger. While I was at it I also trained a model for named-entity recognition (NER).

---

<h1>{{ page.title }}</h1>

All the code for this project can be found at: https://github.com/AMontgomerie/bulgarian-nlp

I can’t really speak Bulgarian, but I’d like to be able to. Sometimes when I receive an instant message in Bulgarian that I can’t understand, I’m forced to just copy-paste it into Google Translate,  which helps me find the overall meaning of the sentence most of the time, but doesn’t provide much in the way of lexical or grammatical information which I could learn something from.

The amount of tools for Bulgarian language learners seems pretty limited, so I thought I’d try building my own. I wanted to know what the individual words in the sentences I was Google-translating were doing, so I decided to train a part-of-speech (POS) tagger. While I was at it I also trained a model for named-entity recognition (NER).

[I have some experience](https://amontgomerie.github.io/2020/07/30/question-generator.html) fine-tuning models using the Huggingface Transformers library, but hadn’t done much training of transformer networks from scratch. There didn’t seem to be any pretrained checkpoints available for Bulgarian, so I was forced to pretrain my own model, before fine-tuning it on my chosen downstream tasks.

## Tokenization

POS tagging and NER are both token classification tasks in that they both require the model to make predictions about the roles of individual words in a sentence. Here we are using “word” and “token” interchangeably, which is not necessarily always the case as text can be tokenized in many different ways.

Tokenization is the process of breaking an input text into a series of meaningful chunks. These chunks, or tokens, can then be encoded and used for a variety of tasks. But along which lines should we split the text? An obvious answer is to split on a word level. We can simply split the text by whitespace and encode each word as a separate token. This allows us to preserve the meaning of each word, but will create problems whenever we encounter words that are not in our vocabulary.

Another strategy is to tokenize on a character level. This solves the problem of encountering out-of-vocabulary words, because we can construct any word (within the character-set of the languages we're using) from its component characters. However, by reducing words to series of characters, we seem to be discarding the meaning that languages contain at a word level.

### Subword Tokenization

A third strategy, and one that has become standard in a lot of modern NLP architectures, is subword tokenization. This is a kind of middle ground between word-level and character-level tokenization, where we split words into chunks based on common patterns. For example, lots of negative words start with the prefix *dis-*, such as *disorganised* or *dishonest*. We can split this prefix from the rest of the word and use it as a token. This helps us preserve the meaning of part of the word while still allowing us to build new unseen words using in-vocabulary components. Even if our system has never encountered the word *disagreeable* during its training, it can still represent it using the tokens *dis*, *##agree*, and *##able* (the ## here indicates that the previous token is part of the same word).

Two of the most common subword tokenization methods are WordPiece and Byte-Pair Encoding (BPE). WordPiece builds tokens based on the combinations of characters which increase likelihood on the training data the most. In contrast, BPE tokens are based on the most frequent byte strings in the data. For this project, BPE tokenization was used.

## Token Classification

POS tagging involves predicting which part of speech a word represents. For example *big* is an adjective and *dinosaur* is a noun. NER is the task of picking named entities from a text. Here *entity* means a string which represents a person, place, product, or other type of named thing. “Adam Montgomerie” is a named entity, but “potato” is not.

Datasets for POS tagging and NER are usually labelled at a word level, which means that, when using a word-level tokenizer, there is a one-to-one correspondence between input tokens and labels, which makes calculating training loss and test accuracy easy. But if we are using a subword tokenizer, each word will be potentially split into multiple tokens. How do we resolve this mismatch between inputs and labels?

### Token Mapping

A popular, but slightly unintuitive, approach is to ignore all but one token from each word when generating predictions. The original BERT paper uses this strategy, choosing the first token from each word. Let's use *disagreeable* as an example again: we split the word into *dis*, *##agree*, and *##able*, then just generate predictions based on *dis*.

[This implementation of a POS tagger using BERT](https://github.com/soutsios/pos-tagger-bert) suggests that choosing the last token from each word yields superior results. This would mean choosing *##able* as the token to generate predictions from. Intuitively this makes sense: in English, important morphological information which hints at the part of speech of a word is often contained at the end of that word. For example *manage* is a verb, *manager* is a noun, and *managerial* is an adjective. We need to inspect the ends of these words to determine which part of speech they correspond to.

This also holds in Bulgarian: учи (learn) is a verb, and училище (school) is a noun. There are several other words with various parts of speech but the same учи or уче prefix.

To implement this, we can use the `offset_mapping` from [Huggingface's tokenizers](https://huggingface.co/transformers/main_classes/tokenizer.html). If we set `return_offsets_mapping` to true, the tokenizer will also return a list of tuples indicating the span of each token.
```python
# list ix:  0123456789012345678
sentence = 'Кучето ми е гладно.'
encoded_sentence = tokenizer(
    sentence, 
    add_special_tokens=True,
    return_offsets_mapping=True,
)
print(encoded_sentence['offset_mapping'])
```
gives us the following output:
```
[(0, 0), (0, 2), (2, 6), (7, 9), (10, 11), (12, 15), (15, 18), (18, 19), (0, 0)]
```
Each tuple in the list represents one of the tokens that the input sequence was split into. The first value of the tuple indicates the start of the tokens span in the original sentence, and the second value indicates the end of the span. Because we set `add_special_tokens=True` we also got special start-of-sequence and end-of-sequence tags. We can tell which tokens are special tokens by checking the width of their span. The first token goes from zero to zero, because it is a start of sentence token and wasn't included in the input sentence at all!

We can also see which tokens are the starts and ends of words by checking if their spans overlap. (0, 2) and (2, 6) overlap at 2, so they must be part of the same word. This means that *Кучето* was split into two tokens. Depending on our label matching strategy, we can either take the first or the second one of these tokens. In contrast, (7, 9) and (10, 11) don't overlap and therefore represent separate words; *ми* and  *е* respectively. For single token words the label matching strategy is irrelevant.

### Implementation

In the training data for both POS and NER there is a single label per word, but when we tokenize our input sentence we end up with a sequence that is longer than the list of labels we have. To resolve this we can pad the label list to be the same length as the tokenized `input_ids` sequence. We can set all tokens that we aren't going to map to label to -100 so that they will be ignored for calculating loss. From [the transformers documentation](https://huggingface.co/transformers/model_doc/roberta.html):
> Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size].

Then we can match either all of the first or last tokens from each word to that word's label. [Here's a Huggingface guide](https://huggingface.co/transformers/custom_datasets.html#token-classification-with-w-nut-emerging-entities) which includes using `offset_mapping` to map tokens to labels. Alternatively, see the `encode_tags_first()` and `encode_tags_last()` methods in [my POS tagging fine-tuning notebook](https://github.com/AMontgomerie/bulgarian-nlp/blob/master/training/pos_finetuning.ipynb). For inference, we won't have any labels to compare with, but we still need to determine which of the input tokens to classify and which to ignore. This can also be done using `offset_mapping`:

```python
def get_relevant_labels(map):
    relevant_labels = np.zeros(len(map), dtype=int)

    for i in range(1, len(map) - 1):
        if is_last_token(map, i) and not ignore_mapping(map[i]):
            relevant_labels[i] = 1
                
    return relevant_labels

def is_last_token(map, i):
    return map[i][1] != map[i+1][0]

def ignore_mapping(mapping):
    return mapping[0] == mapping[1]
```
This function takes an `offset_mapping` generated by a tokenizer and checks each token to see if it's the last token in a word. It then returns a list of values of the same length as the `input_ids` list in range [0, 1] where 1 means that the token at this position should be used for prediction and 0 means that it should be ignored. Each token is compared to the next one to see if there's an overlap. We can skip the first and last tokens in the sequence because they are always SOS and EOS and will be ignored anyway. `is_last_token` checks if a specified token is at the end of a word and `ignore_mapping` just checks if the start and end of the span are the same; if so the token is a special token and can be ignored.

## Training

### Architecture

Following [this tutorial on pretraining transformer models](https://huggingface.co/blog/how-to-train) I used RoBERTa which is a model that was originally introduced in [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). It’s essentially BERT, but with some changes to improve performance. For pretraining, this means that the next sentence prediction objective that was used in the original BERT has been removed. The masked-language modeling objective has also been modified so that masked tokens are generated dynamically during training, rather than being generated all at once during pretraining. This is known as dynamic masking.

I initially pretrained a model using RoBERTA-base which has 12 layers, a hidden size of 768, and 12 attention heads. However, I also tried training a smaller version with only 6 layers and found that performance didn’t suffer at all, so I went with that for the final version.

### Training Set up

For pretraining data, I used data from [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/bulgarian) and [OSCAR](https://oscar-corpus.com/). The model was trained for about 1.5 million steps (CONFIRM?) on the pretraining data with a batch size of 8 using the masked language modeling objective. A Colab notebook containing the pretraining routine can be found [here](https://github.com/AMontgomerie/bulgarian-nlp/blob/master/training/pretraining.ipynb).

For fine-tuning as a part-of-speech tagger, the Bulgarian dataset from [Universal Dependencies](https://universaldependencies.org/) was used. I was able to easily parse the CONLL-U data using [this parser](https://github.com/EmilStenstrom/conllu) and then extract the POS tags. I used [this](https://github.com/usmiva/bg-ner) dataset for named-entity recognition. The data seems to be taken from the [BSNLP 2019 shared task](http://bsnlp.cs.helsinki.fi/shared_task.html). For both tasks, fine-tuning was performed over 5 epochs on the relevant dataset with a learning rate of 1e-4. The relevant Colab notebooks are available for both [POS tagging](https://github.com/AMontgomerie/bulgarian-nlp/blob/master/training/pos_finetuning.ipynb) and [NER](https://github.com/AMontgomerie/bulgarian-nlp/blob/master/training/ner_finetuning.ipynb).

## Results

Below is a comparison of model accuracy with various configurations on the POS tagging and NER test sets. The *Token Mapping* column shows whether the labels were mapped to the first or last token of each word in the input sequence.

### Part-Of-Speech Tagging:

**RoBERTa-small**
| Model                        | Token Mapping | Accuracy |
|------------------------------|---------------|----------|
| roberta-small-pretrained     | first         | 97.75%   |
| roberta-small-pretrained     | last          | 98.10%   |
| roberta-small-no-pretraining | first         | 92.45%   |
| roberta-small-no-pretraining | last          | 93.13%   |

**RoBERTa-base**
| Model                       | Token Mapping | Accuracy |
|-----------------------------|---------------|----------|
| roberta-base-pretrained     | first         | 97.40%   |
| roberta-base-pretrained     | last          | 97.65%   |
| roberta-base-no-pretraining | first         | 91.67%   |
| roberta-base-no-pretraining | last          | 92.93%   |

### Named-Entity Recognition

**RoBERTa-small**
| Model                        | Token Mapping | Accuracy |
|------------------------------|---------------|----------|
| roberta-small-pretrained     | first         | 98.52%   |
| roberta-small-pretrained     | last          | 98.52%   |
| roberta-small-no-pretraining | first         | 95.75%   |
| roberta-small-no-pretraining | last          | 95.69%   |

**RoBERTa-base**
| Model                       | Token Mapping | Accuracy |
|-----------------------------|---------------|----------|
| roberta-base-pretrained     | first         | 98.61%   |
| roberta-base-pretrained     | last          | 98.56%   |
| roberta-base-no-pretraining | first         | 95.66%   |
| roberta-base-no-pretraining | last          | 95.62%   |

### Observations

Interestingly, mapping the label to the last token of each input word produced very slightly better results in all POS tagging tests. However, it didn’t provide any benefit for NER, and even performed slightly worse in some cases. This makes sense since, while key morphological information is often contained at the end of a word (which helps us with POS tagging), there is no specific part of a name which identifies it as such: hence the identical performance of training on the first token and last token for NER.

Unsurprisingly, pretrained models outperform randomly initialised models. I also found that the randomly initialised models didn’t benefit from training for more epochs, so the benefit of pretraining here was more than just faster convergence on downstream tasks.

More surprisingly, base models didn’t outperform small models, despite having twice as many layers. They also appeared to converge more slowly during training.

A mitigating factor for explaining the lack of increased performance of the larger roberta-base model is the small size of the fine-tuning datasets. Perhaps they would have benefitted from larger datasets. Unfortunately I wasn’t able to find any more data, but perhaps could have used synonym replacement as a data augmentation strategy in the case of POS tagging. In theory, we could replace words with their synonyms as long as the synonym is the same part-of-speech as the original word. Other augmentation strategies like back-translation probably wouldn't work as the new sentence wouldn't be guaranteed to have the same correct labels as the original.
