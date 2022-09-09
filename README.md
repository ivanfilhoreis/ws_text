# Weak Supervision - Text

Title: Sequential Short-Text Classification from Multiple Textual Representations with Weak Supervision

# Method

The work investigates a short text labeling function of the commodity market using time series data.

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/method.png" width="600px" alt="table2"/>
</p>

## Labeling Function

A price series $S$ of size $m$ is defined as an ordered sequence of observations, \textit{i.e.}, $S = (s_1, s_2, ..., s_m)$, where $s_t$ represents an observation $s$ at time $t$. The texts documents $D$ is also an ordered sequence $D = (d_1, d_2, ..., d_k)$, where $d_t$ is a text $d$ at time $t$, and size $n$. So, we attribute via time alignment a label (-1, 0 or 1) to texts using the following equation:

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/function.png" width="300px" alt="table2"/>
</p>

the text $d_t$ receives a label according to the level and trend patterns of the time series $S$. The constant $lag$ corresponds to the seasonal period of the time series in number of observations. To exemplify, Figure illustrates the result of Equation applied to a synthetic time series with $lag = 5$. This function aims to capture the time series' stable, increasing, and decreasing behaviors to assign labels to short texts arranged chronologically in time.

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/labelingFunction.png" width="600px" alt="table2"/>
</p>

Table presents examples of labeled headlines with the labeling function.

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/examples.png" width="500px" alt="table2"/>
</p>

The work applies BoW-based representations, pre-trained Neural Language Models, and the proposed TD-BERT model for vector representation of the texts.

## TD-BERT

The work presents a vector text representation model based on bag-of-word that adopts a measure of distance between Terms and Documents from pre-trained BERT models, called TD-BERT. The proposed approach to obtain a new textual representation, which considers the semantic features. First, we extract the collection of documents $D = [d_1, d_2, ..., d_k]$ containing $k$ documents and a set of $b$ terms from this collection $T = [w_1, w_2, ..., w_b] $. This process is similar to the one used in Bag-of-Word. However, we consider the sentence transformers of BERT's pre-trained models to obtain the cosine distance of each term in each document.

The textual representation $D$ with transformers Sentence is defined as $DS = ([B_1], [B_2], ... [B_k])$, where each $B$ is a BERT vector of $h$ positions representing a document $d$ at time $t$. The representation of Terms with the transformers Sentence is defined as $TS = ([W_1], [W_2], ... [W_b])$, where $W_j$ is a BERT vector of $h$ positions that represents a term $w_j$. The set of documents is represented as a document-term matrix constituted by cossine distance $c$ from each vector $k$ composed of $b$ dimensions, illustrated in Figure

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/TD_Bert.png" width="500px" alt="table2"/>
</p>

The matrix values correspond to the cosine distance of each term in each document, i.e., $c(B_k, W_b)$ corresponds to the distance between vectors $W_j$ and $B_i$.

# Experimental Setup

We used five traditional classification algorithms: Multilayer Perceptron (MLP), Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Gaussian Naive Bayes (GNB), and Multinomial Naive Bayes (MNB).

The time series split evaluation strategy was proposed to consider temporal dependence of the textual data, i.e., we train past news to evaluate in a future scenario. Thus, seven splits were used for eight evaluations.

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/ts_split.png" width="500px" alt="table2"/>
</p>

## Results

Table presents a valuation scenario Positive Binary to Corn and Soybean, which we called CPB and SPB.

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/cpb_spb.png" width="500px" alt="table2"/>
</p>

Table below presents more two valuation scenario. We called this Negative Binary classification of Corn (CNB) and soybean (SNB) approach. 

<p align="center">
  <img src="https://github.com/ivanfilhoreis/ws_text/blob/main/img/cnb_snb.png" width="500px" alt="table2"/>
</p>

# Citation

To cite in your work, please use the following bibtex reference:

```
@INPROCEEDINGS{225493,
    AUTHOR="Ivan Reis Filho and Luiz H. D. Martins and Antonio Parmezan and Ricardo Marcacini and Solange Rezende",
    TITLE="Sequential Short-Text Classification from Multiple Textual Representations with Weak Supervision",
    BOOKTITLE="BRACIS 2022 () ",
    ADDRESS="",
    DAYS="28-1",
    MONTH="nov",
    YEAR="2022"
}
```
