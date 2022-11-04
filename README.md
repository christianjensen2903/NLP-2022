Preprocessing and Dataset Analysis {#sec:preprocess}
----------------------------------

To familiarize ourselves with the dataset and prepare it for
classification, we firstly preprocess the data by tokenizing at word
level, stemming and removing stop-words for all languages. The
implementation of the preprocessing vary across the languages in
accordance to their respective morphologies. We used NLTK
[@bird2009natural] for English and Finnish and the Fugashi package
[@mccann-2020-fugashi] for Japanese.\
We also noticed an imbalance in the class distribution throughout the
data. We therefore chose to balance the number of negative and positive
samples for each language. This helps mitigate bias in the classifier
and makes accuracy a meaning-full measure of performance.\
\
We begin by investigating the ways in which the sentences from our
dataset begin and end. Unsurprisingly, one observes from table
[\[tab:table1\]](#tab:table1){reference-type="ref"
reference="tab:table1"} that the most observed initial words are common
question words. Some of the last words for English may look surprising
in their apparent randomness, for example 'zombie'. However the total
frequency of the last english words ranking 2-5 is approximately 0.1%.
This means that, as expected, question mark is the final token in almost
all sentences. The story is almost identical for Finnish, i.e. similar
question words rank top 5 (altough in a slightly different order) and
the question mark makes up almost all the last tokens of the questions.\
Among the top 5 last words of Japanese, one can find question mark,
'when' (

UTF8minいつ

), 'where' (

UTF8minどこ

) and 'what' (

UTF8min何

). However Japanese is not as predictable in its positioning of question
words. Inspecting the second to last token, we also observed question
words like 'who' (

UTF8min誰

), 'what' (

UTF8min何

) and the the common question particle 'ka' (

UTF8minか

) ranked among the most frequent tokens. The first tokens of Japanese
are mostly nouns like America or Japan. This seems reasonable knowing
that Japanese is a SOV (Subject-Object-Verb) language.

Binary Question Classification {#sec:bqc}
------------------------------

To get a good baseline for further investigation into more complex
models, we create a simple Binary Question Classfier (BQC) predicting
whether a question is answerable given the context. The model was
trained on concatenated bag of words (BOW) representations of the
context and the question. Moreover we added a count of the number of
overlapping words as an extra feature. For all the features above we
used cleaned texts. One could however argue that by dropping question
words like 'How' or 'Where' we lose critical information about the type
of question and therefore whether the context includes the answer. To
circumvent this, we used the observation from our data exploration that
almost all the questions begin with question words, and also added a
one-hot vector encoding the first word.\
\
We chose to test four different model architectures for the BQC:
Logistic Regression, RandomForest, Multi-layer Perceptron classifier
(MLP) and XGBoost [@Chen:2016:XST:2939672.2939785]. We used sklearn's
default parameters for all models. The only exceptions are we changed
the MLP model architecture to 2 layers of 50 fully connected neurons,
and gave a max depth of 10 to the RandomForest to combat overfitting.
Given more access to compute, we would have performed an extensive grid
search to obtain more optimized model parameters and test a number of
other baseline models such as SVM, KNN, etc. The baseline accuracies for
the BQC accross languages can be seen in table
[\[table:BOWLogistic\]](#table:BOWLogistic){reference-type="ref"
reference="table:BOWLogistic"}. As we can see the classifiers generally
perform well on the training set, but a decent bit lower on the
validation set. This could indidicate some degree of overfitting. To
combat this several different strategies could be implemented like
regularization, early stopping etc.\
\
Looking at the validation performance, it seems that classifying Finnish
questions was the easiest task by quite a margin. We hypothesize that
this may have been a result of the uneven amount of data for the three
langauges. The dataset included about twice as much data for Finnish
than English and Japanese. Overall we can conclude that an accuracy of
about 75% can be achieved fairly easily with a simple baseline model.\
\

Representation Learning {#sec:repr}
=======================

The word2vec software of Tomas Mikolov et al. has provided
state-of-the-art word embeddings for the last decade
[@Mikolov:1; @Mikolov:2]. These embeddings can contain more complex
information about the words like semantic and syntactic similarity. We
could hope that this could improve the performance of our model - for
example by more easily recognizing synonyms or closely related words
used in both question and context.\
\
We started by training a word2vec model over both context and question
from the training data. The word2vec implementation is handled by the
Gensim package. We chose a vector dimensionality of 100. The original
paper uses sizes of 20-300, and find that, as a general rule, higher
dimensionality only yields better results when the training corpus is
similarly increased in size [@Mikolov:1]. Since our corpus is quite
small, we found a dimensionality of 100 to be appropriate. For all other
model parameters we used the Gensim package default. This meant training
a CBOW model using negative sampling for 5 epochs with an initial
learning rate of 0.025 (with time-based decay).\
In order to feed the word representations to our models, we need to fix
the number of input features - even when document length varies. We
achieved this by averaging over the individual word embeddings of the
document/sentence for both question and context. For out-of-vocabulary
words we returned a zero vector. The two document representations are
concatenated to the input features of the baseline model. To the input
we also added three similarity measures: Euclidian distance, cosine
similarity and BERTScore [@bertscore]. See appendix
[6.4](#sec:similarity){reference-type="ref" reference="sec:similarity"}
for more elaboration on the metrics.\
\
The performance of our extended BQC can be seen in table
[\[table:ContinuousBOWLogistic\]](#table:ContinuousBOWLogistic){reference-type="ref"
reference="table:ContinuousBOWLogistic"}. Accuracy is improved on the
training set but worsened on the validation set. The only exception is
logistic regression which gained from the added features. We figure that
the drop in performance stems from the models overfitting to the
continuous representation, which was not expressive enough. Some of the
context documents contain thousands of words, and it is simply
impossible to compress all that information into a meaning-full
representation of length 100. Word2Vec also has the drawback that word
representations are not context dependent. E.g. in the sentence: \"The
*cell* mate took samples of blood *cells* and took a call on his *cell*
phone.\" the word *cell* has 3 different meanings, but would only be
represented by a single embedding. Later we will see models which are
capable of capturing this contextual information.\
\
We also test a model using only the continuous vector representations.
The result can be seen in table
[\[table:ContinuousLogistic\]](#table:ContinuousLogistic){reference-type="ref"
reference="table:ContinuousLogistic"}. For this configuration validation
performance is much lower than the BOW baseline. For some model types
the gap is as large as 10%. We believe this is largely a result of the
challenges just mentioned, but furthermore the absence of the feature
quantifying BOW overlap. In section
[3](#sec:interp){reference-type="ref" reference="sec:interp"} we will
show that overlap was one of the most crucial features for classifying
samples as answerable.\
That being said, the classifiers based exclusively on continuous
features empirically verified our supposition that the three similarity
measures where effective in capturing answerablility. Adding the metrics
managed to improved performance by 2-3 percentage points on average. The
inclusion or exclusion of these features where however not as
influential for the other types of BQC's.

Language modelling {#sec:three}
==================

We chose to use GPT2 for language modeling as it has shown
state-of-the-art results across many NLP disciplines [@gpt2] . GPT2 is
an auto-regressive transformer-based language model built from decoder
blocks. We finetune the pretrained models (one for each language) on the
pairs of question and context from our dataset. Our choice of pretrained
huggingface models were *gpt2* (english), *Finnish-NLP/gpt2-finnish*
(finnish), *rinna/japanese-gpt2-small* (japanese). All these models
scaled down versions of the original OpenAI model with 1.5B parameters.
The training data was provided in the format
`"Question: "+question+"\nContext: "+context`. Traditionally one would
use a separation token like **\[SEP\]** to signify a divide in the
input. We choose to explicitly write the question/context relationship,
hypothesizing that the model this way would have an easier time learning
the structure of the data. We reason that similar examples like QA's
have most likely been included in the pre-training data.\
We fine-tune each model for 1 epoch on the associated data-set. Ideally
one would run multiple epochs, until either training loss stagnates or
validation performance decreases. The models were trained with 200
warmup steps, a weight decay of 0.01, with the Adam optimizer and a
learning rate of $0.00005$.\
To deal with GPT2's max input size of 1024 tokens, we chose the
head-only truncation strategy. One could potentially improve performance
by looking into more advanced strategies for handling long texts. After
all, head-only truncation can be limiting in circumstances where the
answer lies at the end of a large context document. There are other
methods for handling long texts, one possibility being selecting
sections of text under some length constraint based on feature
importance [@Fiok_2021]. Or one could segment the input into smaller
chunks with some overlap. After obtaining the LM outputs, one could
propagate these through a single recurrent layer, or another
transformer. Following this by a softmax activation one obtains the
final classification decision [@longtext].\
\
In appendix [6.1](#sec:samples){reference-type="ref"
reference="sec:samples"} we have added text samples from our fine-tuned
models. These samples are generated by giving the prompt `"Question: "`
concatenated with one of the top 3 most common starting words found in
section [0.1](#sec:preprocess){reference-type="ref"
reference="sec:preprocess"}. We have generated 5 variations per starting
word using top-k sampling with $k=50$ [@topk] and a max length of 50.\
It has seemingly learned to perfection that there always follows
`"\nContext: "` after `"Question: "` across languages. The questions
posed are quite sound, but some of them are vague like \"What is the
largest currency?\" or \"What is the name of a new kind of animal?\".
That being said, the question and context are always semantically
related, although most of the time the model generates context on some
mildly related tangent. It might for example, ask how something started
and give a context on how it ended. This could be a consequence of only
sampling 50 tokens, i.e. the answer might have come later, but it is
most likely a side effect of training on equal amounts of unanswerable
and answerable questions. Overall the text generated seems logical and
coherent.\
\
In order to test the models performance we utilize the most widely-used
language model evaluation metric named perplexity [@Chen2008]. It is
defined as such: $$\begin{aligned}
    \operatorname{PP}_T\left(p_M\right)=\frac{1}{\left(\prod_{i=1}^t p_M\left(w_i \mid w_1 \cdots w_{i-1}\right)\right)^{\frac{1}{t}}}\end{aligned}$$
Where $T=\left\{w_1, \ldots, w_t\right\}$ is the test set and $p_M$ is a
language model. For an intuitive explanation, say a model has a
perplexity of 20. This means the model is as confused on the test data
as if it had to choose uniformly and independently among 20
possibilities for each word. From the mathematical definition, and the
intuitive explanation, it is clear that the lower bound on this value is
1. It is however important to note, that lower perplexity does not
always equal more human-like text [@kuribayashi-etal-2021-lower].\
\
In table [1](#table:perp){reference-type="ref" reference="table:perp"}
you can see perplexity values calculated over the validation data. They
seem very reasonable. As a baseline comparison, a uniform language model
(which obviously has learned nothing), would have a perplexity equal to
the size of the vocabulary. These results are much to be preferred, and
we are not very surprised given the coherence of the generated text. It
seems the English model did slightly better than the Finnish and
Japanese. However, It is difficult to fairly compare perplexity values
across languages as vocabulary and token size are not constant. One
thing is certain, it is not the amount of pre-training data that
explains the variance. We initially suspected this to be the cause, as
there is a very skewed distribution of global training data favoring
English. But it turns out the Finnish model was trained on 84GB compared
to an English corpus of 40GB.\
The original paper showed slightly better results in the ppl. range 8-18
[@gpt2]. We would expect ppl. to decrease as the model trained for more
epochs, and approach similar values.

::: {#table:perp}
                  **Validation ppl.**
  -------------- ---------------------
   **English**           19.7
   **Finnish**           21.4
   **Japanese**          21.3

  : Perplexity of finetuned GPT2 on training and validation data.
:::

To improve our baseline BQC model, we also try to utilize the learned
representation from our language models in the decision process. There
are a number of ways to include the hidden representation. This could be
the first hidden layer, last hidden layer, sum of all hidden layers,
concatenations of the last 4 hidden layers and more
[@devlin-etal-2019-bert]. We choose to extract the last hidden layer of
the finetuned GPT2, as this performs well in the literature.\
The performance achieved by replacing the input to the BQC with the
hidden representation of length 768 can be seen in table
[2](#table:gpt2model){reference-type="ref" reference="table:gpt2model"}.
Since XGBoost was the preferred model type in all the previous BQC
tasks, we focused exclusively on XGBoost for this configuration. It is
to be noted, that usually for classification tasks like these, one would
use a linear layer on top of the pooled LM output and finally a softmax.
We used XGBoost to allow for a more direct and fair comparison with the
previous BQC tasks.

::: {#table:gpt2model}
                  **Train**   **Validation**
  -------------- ----------- ----------------
   **English**      79.8%         77.6%
   **Finnish**      80.1%         78.5%
   **Japanese**     83.7%         77.8%

  : Accuracy for BQC using GPT2 hidden layer and XGBoost.
:::

Performance is decent, but not competitive with the baseline BOW model.
It is however much better than the previous continuous BQC from section
[1](#sec:repr){reference-type="ref" reference="sec:repr"}. We believe
this is a result of: 1. A larger embedding (768 vs. 100) 2. The ability
to capture contextual information. 3. Much larger and more complex
network. 4. Larger training corpus with both training and fine-tuning 5.
The hidden-layer does not experience the same problem of being clouded
by potentially thousands of characters of context all averaged together.
6. Less over-fitting. The training and validation scores are very close.

Error Analysis and Interpretability {#sec:interp}
===================================

We will now conduct a detailed error analysis for two of the implemented
models. To ensure a fair comparison, both models analysed will have been
trained on the BOW representations extended with word2vec embeddings
(see table
[\[table:ContinuousBOWLogistic\]](#table:ContinuousBOWLogistic){reference-type="ref"
reference="table:ContinuousBOWLogistic"}). We will dissect the English
logistic regression model which achieved a validation accuracy of 77.1 %
and the Finnish random forest model which achieved 64.7%. These models
are highlighted, for their clear gap in performance of 12.4 percentage
points, and the strong tools for explainability both model types offer.\
In order to better explain the reasoning behind the classifications, we
will plot feature importance for the two models. In appendix
[6.5](#sec:feature_plots){reference-type="ref"
reference="sec:feature_plots"} you will see the 20 most significant
feature weights for LR. For the RF we visualize the 20 most crucial
impurity-based feature importances. From these, it is clear that the
engineered features have been a success. We find that overlap is the
most highly weighted positive feature for the LR. One can also notice
euclidean distance as the 10th most important feature in the LR and
cosine similarity is the 18th most important feature for the RF.\
It is also interesting that the word \"stallion\" being present in the
context is highly indicative of non-answerability for the LR. This is a
marker of potential overfitting, since there should be nothing
intrinsically unanswerable about an uncastrated male horse. The training
score of $99.8\%$ verifies this, and shows that there is definitely room
for improving the LR model.

::: {#table:confusion}
  -- ------- ------- ------- --
                             
              **p**   **n**  
       **p**                 
       **n**                 
  -- ------- ------- ------- --

  : Normalised confusion matrix for logistic regression and random
  forest
:::

Another interesting observation, is that the RF model attributes much
more importance to the continuous representation of the context, with 17
out top 20 features being from the average pooled context. This could
partially explain the difference in performance, since we have witnessed
that models trained purely on word2vec embeddings achieved low
generalization for reasons discussed in section
[1](#sec:repr){reference-type="ref" reference="sec:repr"}.\
\
To get more insight on the similarities and common mistakes of the
models, we present a confusion matrix in table
[3](#table:confusion){reference-type="ref" reference="table:confusion"}.
One notices that the RF actually outperforms the LR by a small margin in
more accurately avoiding false-negatives and correctly classifying
true-negatives. That being said, we see a large discrepancy when it
comes to predicting the positive class. The RF has trouble identifying
true-positives and an over-tendency to predict false-positives compared
to the LR.\
We believe this could be a result of the RF's inability to learn
BOW-overlap as a useful feature. This is consistent with LR weighting
overlap much higher, and clearly being more proficient in recognizing
true-positives.\
\
Now we will test the robustness of the better BQC system with
adversarial sequences in order to gage strengths and weaknesses. In
appendix [6.2](#sec:advers){reference-type="ref" reference="sec:advers"}
you will find three adversarial questions. For the first instance we
construct a coherent question and context. The context however
elaborates on a related tangent and does not address the question. We
expect the model to be fooled by this instance, as the sentences have
decent overlap, we have included no words which are weighted heavily
toward the negative class, and the euclidean distance is likely small,
since the two sentences are semantically related.

In the second, now answerable sample, we try to exploit the overfitting
of the model. We have made sure to add 'stallion' to the context,
'american' to the question, and make overlap as small as possible. All
of the above are correlated highly with the negative class.\
The last question has a nonsensical context which simply repeats words
from the question. This examples show the danger of relying too much on
overlap to assess answer-ability.\
\
As seen in table [\[table:advin\]](#table:advin){reference-type="ref"
reference="table:advin"} we manage successfully to fool the model on
every adversarial instance. This goes to show that our logistic
regression model has a few weaknesses. One is that the model seems to be
over-fitted on certain non-related tokens. More various remidies such as
more training data or regularization could rectify this. And secondly,
relying too much on overlap makes the model vulnerable to nonsensical
contexts.

Sequence Labelling {#sec:seqlab}
==================

Often one is not only interested in whether a question is answerable or
not but what the actual answer is. To do this the task is often model as
a span selection task. This setting is widely used as it is
straightforward and effective approach for single answer questions.
However, this approach has some limitations. Most notably that the
context only can contain one answer despite other parts of the text also
being valid answer. We therefore chose to model the answer span with the
IOB-tagging system as this allows for multiple parts of the text to
answer the question.\
To perform token-classification tasks such as NER- or POS-tagging a
BiLSTM-CRF model is commonly used. We therefore implemented such a model
composed of an embedding layer, two stacked Bi-LSTM layers, a linear
layer and finally a CRF layer. For the embedding layer we chose to use
the fasttext library [@grave2018learning] as it has pretrained vectors
for many different languages. We selected a Bi-LSTM dimensionality of
128 and employed dropout for regularization ($p=0.01$).\
We used an Adam optimizer and the Cyclic LR scheduler to schedule
learning rates to speed up the training. We used a learning rate of
$2\cdot 10^{-5}$. We trained this model for 10 epochs with a batch size
of 16. The results can be seen in table
[4](#table:bilstm-crf){reference-type="ref"
reference="table:bilstm-crf"}. Here we see that it seems to learn
something when looking at the performance on the train score, but when
looking at the validation score for both English and Japanese it fails
to predict the right sequence. However when looking at Finnish the
validation score seems to be on par with the training score. This is
accordance with the results we saw earlier where the BQC had a easier
time figuring out whether the Finnish contexts contained the answer.

::: {#table:bilstm-crf}
                  **Train F1**   **Validation F1**
  -------------- -------------- -------------------
   **English**       0.329             0.053
   **Finnish**       0.361             0.326
   **Japanese**      0.330             0.104

  : Training results for the BiLSTM-CRF
:::

One problem with the current implementation of the Bi-LSTM-CRF is that
to find the optimal sequence of tags the Viterbi-algorithm is used. This
can be computationally expensive and therefore is very time consuming.
We therefore investigate the effects of beam-search, which is an
alternate method for finding a probable optimal sequence of tags.
However when performing beam search with the current model the
probability distribution for the next step does not depend on the
previous step's choice. A common solution to this is to employ beam
search with an encoder-decoder architecture:\

![Illustration of the encoder-decoder architecture used for the
BiLSTM-CRF with beam-search](encoder-decoder.png){width="\\textwidth"}

The encoder consist of an embedding layer and two BiLSTMs layers with a
dimension of 128 as before. Likewise the decoder also consist of an
embedding layer and two BiLSTM layers. For training the same parameters
was used as the previous BiLSTM model. With this trained model we try to
predict the optimal sequence of tags with 1, 2 and 3 beams. The results
for each language can be seen at table
[5](#table:beam-search){reference-type="ref"
reference="table:beam-search"}. And the times to decode can be seen at
table [6](#table:beam-search-time){reference-type="ref"
reference="table:beam-search-time"}.

::: {#table:beam-search}
   **Language / Beams**   **1**   **2**   **3**
  ---------------------- ------- ------- -------
       **English**        0.316   0.172   0.329
       **Finnish**        0.009   0.015   0.234
       **Japanese**       0.021   0.091   0.007

  : Validation results for different amounts of beams
:::

::: {#table:beam-search-time}
   **Language / Beams**   **1**   **2**   **3**
  ---------------------- ------- ------- -------
       **English**         3.5    8.25     17
       **Finnish**          4       9     17.75
       **Japanese**        5.5    13.25    26

  : Decode times (in minutes rounded to nearest 15s) for different
  amounts of beams
:::

As we can see the time to decode the optimal sequence decreases notably
when the number of beams decreases. In theory when the number of beams
increases the performance should also increase however when we look at
our results this only seem to be true for Finnish. One thing is to
remember is that the decoder tries to find the optimal sequence given
the predicted probabilities. But like we saw with our previous
Bi-LSTM-CRF and the current one the performance is not perfect.
Therefore the probability mapping is not perfect which can be the reason
for these results. One more thing to note is that doing exhaustive
search contra greedy search is most important for problems with
long-distance dependencies between output decisions. In this case
whether it is part of the answer depends more on the surrounding tokens
that if some earlier text was tagged as an answer. We therefore don't
expect beam-search to have a big impact on the result which is what we
somewhat notice with English and Japanese.\
One interesting thing we however notice is that this sequence labeller
seem to be able learn English considerably better than the last and here
even outperforms Finnish.\
A reason for the poor performance could be that the LSTM doesn't capture
the long range dependency between the question and the different parts
of the context well enough. We therefore also chose to fine-tune a
pretrained transformer language model to do the tagging. In detail the
model used was the BERT-model [@devlin-etal-2019-bert] since this model
performs competitively with state-of-the-art methods on both NER-tagging
and Q&A. On top of the BERT model a linear layer is placed. We chose to
omit the CRF layer like the original BERT paper. Since the BERT model
splits words into word pieces we only use the representation of the
first sub-token as the input to the token-level classifier. The
pretrained models that was used was for all languages a BERT base model
with 110M trainable parameters pretrained on text from its respective
language. For this task we used the same training parameters as for the
BiLSTM-CRF for easy comparison in addition to 200 warm up steps and a
weight decay of 0.01. This model was also trained for 10 epochs with a
learning rate of $2\cdot 10^{5}$. The training results can be seen in
table [7](#table:seq-BERT){reference-type="ref"
reference="table:seq-BERT"}

::: {#table:seq-BERT}
                  **Train F1**   **Validation F1**
  -------------- -------------- -------------------
   **English**       0.642             0.497
   **Finnish**       0.865             0.711
   **Japanese**      0.668             0.562

  : Training results for the sequence tagger using BERT
:::

This model performs significantly better but still not perfect. We
therefore chose to do a qualitative inspection of some of the predicted
answer spans to see if they makes sense and where it goes wrong. A few
examples of predicted answers versus the correct one for different
question can be seen in Appendix [6.3](#app:seqlab){reference-type="ref"
reference="app:seqlab"}. What we generally notice is that it is a bit
biased towards 'O' which makes somewhat sense due to the huge presence
in the data. This could maybe be mitigated by oversampling questions
with an actual answer. We also notice it generally has a hard time with
answers only consisting of a date or a few words, which maybe can be
explained by the earlier observation that the presence of the word both
in the answer and in the question makes it more likely to be answerable.
Moreover we notice that there might be some ambiguity in the data. For
example one question ask what martial arts Marines learn. Here the
correct answer is the Marine Corps Martial Arts Program. However our
sequence labeller tagged the following answer \"combine existing and new
hand-to-hand and close quarters combat techniques with morale and
team-building functions and instruction in the Warrior Ethos\", which
also makes sense as an answer to the question.\
In general we found that it is indeed possible also to model Q&A as a
token-classification task. This may even fix the problem when the
context contains multiple correct answer as we saw with the ambiguity of
one of the questions. Our model was however long from perfect. It should
however be noted that the model isn't the largest or finetuned on much
data. Another problem is that we haven't done any fine tuning at all
which possibly further could improve the model.

Multilingual QA
===============

Answerable TyDiQA BQC
---------------------

In order to extend the Answerable TyDiQA binary question from section
[2](#sec:three){reference-type="ref" reference="sec:three"}, we change
the language model from a language-specific pretrained GPT2 to BLOOM
[@bloom]. BLOOM is an autoregressive language model, trained to continue
text from a prompt on more than 46 languages, including English, Finnish
and Japanese. We trained the smallest version of BLOOM (560M parameters)
with 200 warm up steps, a weight decay of $0.01$ and the Adam optimizer
with a learning rate of $0.00005$. Due to hardware constraints
(specifically RAM limitations) we were only able to run the model with a
batch-size of 1, which in turn made time-constraints an issue. The
results below are therefore based on training and validation of only 300
balanced samples each. It is to be stated clearly, that is of course not
optimal. One would normally train for multiple epochs, and until no
significant improvements in loss are to be found (or no improvement in
performance on the validation set).\
From our earlier BQC models, we have seen that XGBoost is very
performant on the classification tasks. But since XGboost is quite
memory consuming, we resorted to using Logistic regression as the
predictor for this task.

::: {#tab:multi-bloom}
   **Fine-tune/Eval**   **English**   **Finnish**   **Japanese**
  -------------------- ------------- ------------- --------------
      **English**        **0.675**       0.645          0.63
      **Finnish**          0.64        **0.705**       0.585
      **Japanese**         0.42          0.555       **0.655**

  : Zero-shot cross-lingual accuracy for the multilingual BQC
:::

From table [8](#tab:multi-bloom){reference-type="ref"
reference="tab:multi-bloom"} we see that the zero-shot matrix
non-theless looks very much like how you would expect. We have
highlighted the best performing configuration for each row, and the
diagonal is clearly to be preferred.\
It seems that Japanese is the training language which results in the
least amount of generalization. After fine-tuning, it actually performed
worse than simply guessing randomly on the English data-set. And again,
we observe Finnish to be by far the easiest language for determining
answerability. This time the phenomenon certainly cannot be explained by
a larger training set. It may be that the morphology of Finnish is
somehow easier to grasp for the models we investigate. We also notice
that fine-tuning on English achieved the best on average across the
languages. However, the results should be taken with a grain of salt.
Given the low count of training and validation samples, all values are
associated with a high degree of uncertainty.\
To conclude, there are many ways in which performance on this task could
be improved. Training on more data, larger language model (BLOOM can get
as large as 7B parameters), more sophisticated final classifier (a
larger neural network, XGBoost etc.), more deliberate choice of
hyperparameters and so on. It is well known that deep learning models
needs a great deal of data to achieve good performance, but the fact
that one already can get around 65% accuracy on only 300 samples is
truly a testament to the power of pretrained models.\

IOB tagging system
------------------

For the answer extraction we will use the model utilising a pre-trained
transformer architecture since this model was superior in the previous
analysis. We will use the same architecture and the same configuration
as before. However for this task we will use mBERT as our pretrained
model. The results of the zero-shot cross-lingual evaluation can be seen
in table [9](#tab:multiseqlab){reference-type="ref"
reference="tab:multiseqlab"}.

::: {#tab:multiseqlab}
   **Fine-tune/Eval**   **English**   **Finnish**   **Japanese**
  -------------------- ------------- ------------- --------------
      **English**        **0.447**       0.437         0.426
      **Finnish**          0.515       **0.677**       0.443
      **Japanese**         0.470         0.503       **0.572**

  : Zero-shot cross-lingual evaluation for the sequence labeller from
  section [4](#sec:seqlab){reference-type="ref" reference="sec:seqlab"}
:::

We have again highlighted the best configuration for each row. We notice
it performs the best on the language it was pretrained on which is in
accordance with our expectations. We also notice that the performance is
somewhat the same inpendent on what language it was pretrained on. The
performance is however a bit lower than when using a language specific
pretrained model which is to be expected. The performance when
finetuning on Japanese is however slightly better. We hypothesise that
this is because the Japanese texts often also contains a bit of English

Appendix
========

  ----------------------------- ------- ------------ --------- ----------------- ----------------- -------------
                                                                                                   
   (r)2-3(l)4-5(l)6-7 **Rank**   First      Last       First         Last              First           Last
                1                When        ?        Milloin          ?            UTF8min日本          ?
                2                What      zombie      Mikä     tulitaistelussa      UTF8min『      UTF8minいつ
                3                 How    metabolite    Missä      tohtoriksi+     UTF8minアメリカ    UTF8minた
                4                 Who       \\\\       Kuka        syntynyt         UTF8min世界     UTF8minどこ
                5                Where      BCE        Mitä        pinta-ala         UTF8min第       UTF8min何
  ----------------------------- ------- ------------ --------- ----------------- ----------------- -------------

  --------------------------------------- ------- ------- ------- ------- ------- ------- ------- -------
                                                                                                  
   (r)2-3(l)4-5(l)6-7(l)8-9 **Language**     T       V       T       V       T       V       T       V
                  English                  99.8%   76.8%   94.9%   79.1%   88.8%   77.5%   82.1%   75.5%
                  Finnish                  99.6%   80.2%   89.3%   82.2%   93.8%   79.2%   78.5%   74.8%
                 Japanese                  99.8%   73.2%   95.3%   79.6%   93.9%   73.4%   78.5%   70.8%
  --------------------------------------- ------- ------- ------- ------- ------- ------- ------- -------

  --------------------------------------- ------- ------- ------- ------- ------- ------- ------- -------
                                                                                                  
   (r)2-3(l)4-5(l)6-7(l)8-9 **Language**     T       V       T       V       T       V       T       V
                  English                  99.8%   77.1%   99.8%   78.7%   92.9%   76.7%   81.6%   74.2%
                  Finnish                  99.3%   80.5%   96.7%   81.6%   96.4%   77.8%   74.2%   64.7%
                 Japanese                  99.7%   73.4%   99.7%   78.0%   94.6%   74.3%   78.9%   70.2%
  --------------------------------------- ------- ------- ------- ------- ------- ------- ------- -------

  --------------------------------------- ------- ------- ------- ------- -------- ------- ------- -------
                                                                                                   
   (r)2-3(l)4-5(l)6-7(l)8-9 **Language**     T       V       T       V       T        V       T       V
                  English                  71.2%   71.0%   100%    72.6%   73.3%    72.2%   91.8%   71.9%
                  Finnish                  69.3%   70.1%   98.6%   72.4%   72.4%    72.2%   87.6%   72.7%
                 Japanese                  68.0%   65.8%   100%    70.1%   72.8 %   68.7%   89.9%   69.2%
  --------------------------------------- ------- ------- ------- ------- -------- ------- ------- -------

Sampled Sentences {#sec:samples}
-----------------

### English

``` {breaklines=""}
0: Question: When was the International Hydrographic Organization formed?
Context: From 1869 to 1904 the Organization was composed of 17 international teams, consisting of representatives from 13 countries. These included: France, Russia, Brazil, the former Yugoslavia (formerly
1: Question: When did the War of the Spanish Succession begin?
Context: War of the Spanish Succession (Spanish: Caligula de las Indias) was a Spanish civil war between the Spanish and French forces that ended with the assassination of
2: Question: When was the original Tomb Raider published?
Context: Tomb Raider is a third-person action role-playing game developed by Eidos Interactive and published by Square Enix Studios, Inc. for Microsoft Windows, PlayStation 3, PlayStation 4, Xbox
3: Question: When was the Last House on the Left released?
Context: Last House on the Left (also known as the Last House on the Right and the Last House on the Left) is an American social movement founded in 1971 by the Socialist Party
4: Question: When are the NFL draft eligible?
Context: The 2017 NFL Draft began on April 29, 2017, and ended on March 22, 2018, with all 18 teams choosing a winner. Once the picks were determined, all teams then chose an
0: Question: What is the largest currency?
Context: The English minting system, also known as the Royal Mint,[lower-alpha 1] followed the tradition of creating the most precious single-dollar note, known as the "Queen's Crown".[
1: Question: What language do they speak in Denmark?
Context: Danish is known for its distinctive nasal, choral, and laryngeal vocalizations and for its distinctive pronunciation. It is also known for its unique architecture and distinctive style of architecture
2: Question: What did George Orwell write to the Observer public school system?
Context: On 3 November 1984, the Observer's public school system closed, as a decision to do so has been described as "unconscionable". The Observer has
3: Question: What is the name of a new kind of animal?
Context: Species of animals have been described since the earliest known listings. As animal names are based on their behavior, and animals that move based on their environment, the name which is
4: Question: What is the most abundant mineral in Hawaii?
Context: Hawaii has an area of 4,700.7 million acres (24,000km2), of which 800,000,000 acres (28,000km2) is agricultural
0: Question: How loud is lightning?
Context: The current has a wavelength of about 380KHz (542 metres/s). Although the luminous intensity was thought to fluctuate over the years, the actual luminous luminous intensity was computed
1: Question: How many soldiers are in the U.S. Army?
Context: The enlistment of enlisted men and women by the United States armed forces is authorized by law at all levels of command.[1] The enlistment of an enlisted man
2: Question: How long was the battle of Ypres in France?
Context: After its defeat, the Austrians formed a coalition led by the Prussians against them. The Prussians won. When Austrian France fell on 19 August 1914
3: Question: How old is Luke Cage?
Context: After the death of Cage's half-brother Luke Cage (Sean Connery), Luke undergoes a series of transformations to become his true self.[2] When he appears at the end of the
4: Question: How many years did it take for Glee to air?
Context: The series premiered on CBS on December 19, 1965. It starred Leslie Knope and Leonard Nimoy as characters assigned to the Los Angeles Police Department (LPD)
```

### Finnish

``` {breaklines=""}
0: Question: Milloin ensimmäinen Avaruusasema laukaisi Tähtitiede -lehden?
Context: 
 
Vuoden 1953 avaruuslentosuunnitelmassa esiteltiin ensimmäinen avaruusasema. Avaruusasema nimettiin sen jälkeen uudeksi avaruusaseman nimellä
1: Question: Milloin ensimmäinen elokuva tehtiin?
Context: 

Vuonna 1989 brittiläinen tuottaja Denny Hooker palkittiin Women of the Universe -palkinnolla.[2] Lisäksi kaksi muuta tunnustusta annettiin muun muassa George Luc
2: Question: Milloin Charles de Gaullen syntyi?
Context: Charles de Gaullen syntyi Ranskassa 1872. Hän joutui vuonna 1872 naimisiin saksalaisen, ranskalaisen ja italialaisen kreivittären kanssa. Avioliitosta tuli
3: Question: Milloin Nintendo on perustettu?
Context: Nintendo ei ole enää itsenäinen yhtiö, vaan se on itsenäinen brittiläinen tietokone- ja Internet-palveluntarjoaja. Sen perustivat William Friedman ja Arthur B. B. Goth
4: Question: Milloin Ludvig XIII syntyi?
Context: 
Valentian tasavallan hallitus aloitti valtakuntansa hajottamisen vuonna 1886. Vallankumous alkoi jo vuonna 1896. Vallankumous johti vallankumoukseen ja se
0: Question: Mikä oli Leonid Brešnijevin puoliso?
Context: Ignatius, jota pidettiin Venäjän aatelisena, nimitti usein [nijiveiksi.  [1
1: Question: Mikä on Suomen pisin joki?
Context: Suomen pisin joki on Laatokan koillisosa. Se on 5,6 kilometriä pitkä ja 12,7 kilometriä leveä. Joki laskee Päijänteelle, josta se kulkee Ha
2: Question: Mikä oli Suomen yleisin kalalaji?
Context: Suomen yleisin kalalaji, ahven, ovat pieniä särkiä ja muita järviä. Järvellä esiintyy ainakin kymmenen lajia, joista seitsemän lajia on uhanalaisia.[
3: Question: Mikä on Sknen korkein vuori?
Context: Skánessa on kahdeksan tasavaltaan kuuluvaa vuorijonoa.[1] Se on pisin yhtäjaksoisesti kohoava joki. Vuorenhuippu
4: Question: Mikä oli Tsernobylin ydinvoimalaonnettomuus?
Context: 
Tsernobylin ydinvoimala suljettiin vuonna 2014.[1] Tsernobyl on radioaktiivisen hiilen onnettomuus.[2] 
0: Question: Missä sijaitsee Suomen vanhin jäähalli?
Context: Vuoden 1949 talviolympialaisten mitalitoiveissa oli kaksi hopeista ja kaksi pronssista palkintoa. Seuraavana vuonna olympialaisissa pelattiin pronssiottelu, jossa pronssimitalistijoukkueiden
1: Question: Missä sijaitsee Runeberginkadun raatihuone?
Context: Raatihuone (lyhenne Valse dše des grands) on kaupunki Mannerheimintiellä. Se on myös Suomen vanhin kaupunginmuseo. Kulttuurikeskuksen
2: Question: Missä sijaitsee Kiinan muuri?
Context: Luokka:Suuri Venäjä
Luokka:Maailman sota
Luokka:Kansainvälisen politiikan tapahtumat
Luokka:Seulonnan keskeiset artikkelit
Luokka:Kiinalaiset
Luokka:Kiinan
3: Question: Missä maakunnassa sijaitsee Suomen suurin uimahalli?
Context: Suomessa on kaksi uimahallia sekä yli 6500 ulkopaikkaa: Telakkavesi (660 ha) ja Vääksyn uimahalli (202 ha
4: Question: Missä Suomen kuntapolitiikka kehittyi?
Context: Suomen 1900-luvulla alkanut kuntapolitiikka oli voimakasta ja voimakasta. Tällöin alkoi nousta esiin merkittäviä kysymyksiä kuntien talouden luonteesta ja kehityksestä. Merkittävimmät uudistukset tapahtuivat 1900-luvun alussa,
```

### Japanese

**Note:** The original Japanese generated text does have newlines, but
the latex package rendering the characters does not recognize this.\

UTF8min0: question: 日本における、自動車の最大排気量は何? context: 1.
が日本の自動車メーカーに、2が軽自動車の各社に適用される。そして現在では、日本自動車協会は排気量と車名・車種で
1: question: 日本自衛隊はいつ誕生した context:
戦後自衛隊創設以降、国主導の下で防衛組織の整備と防衛機能の充実が図られており、自衛隊組織が「自衛隊」として独立する事態にはなっていない\[38\]
2: question: 日本国鉄はいつ創立した? context:
国鉄は、日本最大の私鉄で、日本の鉄道事業者では第2位の規模を持つ日本唯一の鉄道事業者である。1895年(明治28年)に、東京・日本橋
3: question: 日本の一人当たりgdpは? context: gdp (english)
は、アメリカ合衆国の数値。世界の国民総生産の内訳を取りまとめた世界経済予測を基に算出され、各国のgdpを比較した際、
4: question: 日本サッカーはいつ始まった? context:
サッカー競技は、日本の国民競技で歴史ある競技の一つ\[1\]\[2\]\[3\]\[4\]。日本文化の発祥地ともいわれる。
0: question: 『銀河英雄伝説 新風」の舞台となった場所はどこ context:
銀河英雄伝説の中心人物として活躍した「英雄」と「伝説の英雄」との因縁に思いをいだき、仲間として彼らと共に銀河
1: question: 『赤マルジャンプ』の視聴率は? context:
2011年9月5日〜9月18日に、nhk総合テレビで再放送された。 2: question:
『バットマン』はいつ書かれた? context:
1993年12月2日の『映画『バットマン』に主演することになったレイ・チャールズは、後に『バットマン:
バタイヤー』で主演することを約束する 3: question: 『キャプテンハーロック
』の監督はだれ context:
『キャプテンハーロック』の脚本を担当したのは映画監督のスティーヴ・クロッパーである。脚本についてクロッパーは「この映画の制作に重要な影響を与えた」と
4: question: 『ポケットモンスター サン・ムーン 』のストーリーは?
context: 『ポケットモンスター ウルトラサン』・『ポケットモンスター
ブラック・ホワイト』に登場したポケモンのボスと闘います。 0: question:
アメリカ連邦準備理事会(frb)はいつ結成した context:
2007年11月11日の声明では、連邦準備理事会法に基づいて、連邦公開市場委員会(fomc)の議長を務めるジョージ・c
1: question: アメリカ南北戦争での南軍の勝利数は? context:
1861年1月2日に、北軍のジェイムズ・ヘンリーが南軍側に加わっていたレキシントンの戦いで、フランクリン・d.c・マクヘンリーが南軍のジェイムズ
2: question: アメリカの国旗は何色ですか? context:
アメリカ合衆国の国旗(アメリカがっしゅうこくのこっき)とはアメリカ合衆国の実色であるアメリカ合衆国黄色を使用する連邦の国旗であり、アメリカ合衆国のシンボルである。
3: question: アメリカテネシー州メンフィスはどこにある? context:
アメリカ最大のインディアン居留地メンフィスは現在8の部族が住む、3つのインディアン居留地が連結しており、この地区はメンフィス州の州都メンフィス県に属している\[6\]。メンフィスは
4: question: アメリカワシントン州出身のバスケットボール選手は誰?
context:
bjリーグの秋田ノーザンハピネッツに4年目にして入団し、4年間プレーした。秋田がhcとなった2011-12シーズンは7勝1敗、同シーズン終了後には

Adverserial questions {#sec:advers}
---------------------

###  {#sec:q1}

**Q:** When was Queen Elizabeth II born?\
**A:** Queen Elizabeth II was the queen of England from 1952 until she
died on the 8th of September 2022.\
**Type:** Not answerable.\
[**Source**](https://en.wikipedia.org/wiki/Elizabeth_II).

###  {#sec:q2}

**Q:** What is an uncastrated male horse called in American English?\
**A:** A horse of masculine gender which has not been castrated can be
referred to as a stallion in American English\
**Type:** Answerable.\
[**Source**](https://www.merriam-webster.com/dictionary/stallion).

###  {#sec:q3}

**Q:** How can fast does the earth spin around its own axis?\
**A:** Earth earth earth earth spin spin spin spin spin around around
around around\
**Type:** Not answerable.

Examples of sequence labeller {#app:seqlab}
-----------------------------

Question: When was github created?\
Answer: February 2008\
Predicted answers: \[\]\
Question: Who was the first leader of West Germany?\
Answer: Theodor Heuss\
Predicted answers: \['Theodor Heuss'\]\
Question: What martial arts do Marines learn?\
Answer: Marine Corps Martial Arts Program\
Predicted answers: \['combine existing and new hand-to-hand and close
quarters combat techniques with morale and team-building functions and
instruction in the Warrior Ethos', '\[', 'which began in 2001 , trains
Marines', 'of'\]\
Question: What is the world's largest horse show?\
Answer: Devon Horse Show\
Predicted answers: \['Since 1896 , the Devon Horse', 'is'\]\
Question: When did the first episode of Big Brother Australia air?\
Answer:\
Predicted answers: \[\]\
Question: When was Nike founded?\
Answer: January 25, 1964\
Predicted answers: \[\]\
Question: When did Final Fantasy Type-0 come out?\
Answer:\
Predicted answers: \[\]\
Question: What is the oldest city in Myanmar?\
Answer: Beikthano\
Predicted answers: \[\]\
Question: What is the strongest recorded wind?\
Answer: was during the passage of Tropical Cyclone Olivia on 10 April
1996: an automatic weather station on Barrow Island, Australia,
registered a maximum wind gust of 408km/h\
Predicted answers: \[\]\
Question: Who was president in 1817?\
Answer:\
Predicted answers: \[\]\

Similarity Measures {#sec:similarity}
-------------------

$$\begin{gathered}
    \textbf{Euclidean distance}\\ \lVert q_{\text{avg}} - c_{\text{avg}} \rVert\\
    \textbf{Cosine similarity}\\ \frac{q_{\text{avg}}\cdot c_{\text{avg}}}{\lVert q_{\text{avg}}\rVert\lVert c_{\text{avg}}\rVert}\\
    \textbf{BERTScore}\\
    R_{\mathrm{BERT}}=\frac{1}{|q|} \sum_{q_i \in q} \max _{c_j \in c} q_i^{\top} c_j\\ P_{\mathrm{BERT}}=\frac{1}{|c|} \sum_{c_j \in c} \max _{q_i \in q} q_i^{\top} c_j\\ F_{\mathrm{BERT}}=2 \frac{P_{\mathrm{BERT}} \cdot R_{\mathrm{BERT}}}{P_{\mathrm{BERT}}+R_{\mathrm{BERT}}}\end{gathered}$$
Where $q_{\text{avg}}$ and $c_{\text{avg}}$ are the average question and
context representations, and $q_i$ and $c_j$ represent the individual
word embeddings that the documents are composed of. In BERTScore it is
also presumed that $q_i$, $c_j$ have been normalized to unit length. We
hypothesized that these similarity measures could increase performance
in a similar manor as to the simple overlap feature from section
[0.2](#sec:bqc){reference-type="ref" reference="sec:bqc"}. The first
measure, euclidean distance, does provide some indication of whether the
two embeddings are close in the representational space. But it has the
drawback, that occurrence counts impact vector magnitudes, meaning that
semantic similarity is better captured in the angle between embeddings.
This is exactly what the second measure, cosine similarity, achieves,
and why it is the default indicator of like-ness for word embeddings.
The last similarity measure is BERTScore [@bertscore]. BERTScore is
traditionally used as an automatic evaluation metric for text
generation. BERTScore works by computing a similarity score for each
token in the candidate sentence with each token in the reference
sentence. However, instead of exact matches, it relies on token
similarity using contextual embeddings.\
Even though we are not doing text generation, we can nonetheless utilize
this same method with word2vec embeddings rather than BERT to obtain a
more nuanced similarity metric.\
\

Feature importance plots {#sec:feature_plots}
------------------------

  ---------------------------------------------------
   ![image](LR_feature_importance.png){width="25cm"}
  ---------------------------------------------------

  -----------------------------------------------------------
   ![image](RF_feature_importance_finnish.png){width="25cm"}
  -----------------------------------------------------------
