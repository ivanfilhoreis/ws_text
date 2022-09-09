# Weak Supervision - Text

Title: Sequential Short-Text Classification from Multiple Textual Representations with Weak Supervision

# Method

The work investigates a short text labeling function of the commodity market using time series data.

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/method.png" width="600px" alt="table2"/>
</p>

## Labeling Function


<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/labelingFunction.png" width="600px" alt="table2"/>
</p>

## Textual representation

In addition, this study presents a vector text representation model based on bag-of-word that adopts a measure of distance between Terms and Documents from pre-trained BERT models, called TD-BERT. The proposed approach to obtain a new textual representation, which considers the semantic features. First, we extract the collection of documents $D = [d_1, d_2, ..., d_k]$ containing $k$ documents and a set of $b$ terms from this collection $T = [w_1, w_2, ..., w_b] $. This process is similar to the one used in Bag-of-Word. However, we consider the sentence transformers of BERT's pre-trained models to obtain the cosine distance of each term in each document.

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/TD_Bert.png" width="500px" alt="table2"/>
</p>


