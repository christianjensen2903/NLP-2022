\title{
NLP 2022/2023: Group Project
}

\author{
Christian Mølholt Jensen
}

$\operatorname{vdj} 579$ Axel Højmark

x.bm2 65 Christoffer Ringgaard

CW 1852

\subsection{Preprocessing and Dataset Analysis}

To familiarize ourselves with the dataset and prepare it for classification, we firstly preprocess the data by tokenizing at word level, stemming and removing stop-words for all languages. The implementation of the preprocessing vary across the languages in accordance to their respective morphologies. We used NLTK (Bird et al., 2009) for English and Finnish and the Fugashi package (McCann, 2020) for Japanese.

We also noticed an imbalance in the class distribution throughout the data. We therefore chose to balance the number of negative and positive samples for each language. This helps mitigate bias in the classifier and makes accuracy a meaning-full measure of performance.

We begin by investigating the ways in which the sentences from our dataset begin and end. Unsurprisingly, one observes from table 11 that the most observed initial words are common question words. Some of the last words for English may look surprising in their apparent randomness, for example 'zombie'. However the total frequency of the last english words ranking 2-5 is approximately $0.1 \%$. This means that, as expected, question mark is the final token in almost all sentences. The story is almost identical for Finnish, i.e. similar question words rank top 5 (altough in a slightly different order) and the question mark makes up almost all the last tokens of the questions.

Among the top 5 last words of Japanese, one can find question mark, 'when' (いつ), 'where' (どこ) and 'what' (何). However Japanese is not as predictable in its positioning of question words. Inspecting the second to last token, we also observed question words like 'who' (誰), 'what' (何) and the the common question particle ' $\mathrm{ka}$ ' (か) ranked among the most frequent tokens. The first tokens of Japanese are mostly nouns like America or Japan. This seems reasonable knowing that Japanese is a SOV (Subject-Object-Verb) language.

\subsection{Binary Question Classification}

To get a good baseline for further investigation into more complex models, we create a simple Binary Question Classfier (BQC) predicting whether a question is answerable given the context. The model was trained on concatenated bag of words (BOW) representations of the context and the question. Moreover we added a count of the number of overlapping words as an extra feature. For all the features above we used cleaned texts. One could however argue that by dropping question words like 'How' or 'Where' we lose critical information about the type of question and therefore whether the context includes the answer. To circumvent this, we used the observation from our data exploration that almost all the questions begin with question words, and also added a one-hot vector encoding the first word.

We chose to test four different model architectures for the BQC: Logistic Regression, RandomForest, Multi-layer Perceptron classifier (MLP) and XGBoost (Chen and Guestrin, 2016). We used sklearn's default parameters for all models. The only exceptions are we changed the MLP model architecture to 2 layers of 50 fully connected neurons, and gave a max depth of 10 to the RandomForest to combat overfitting. Given more access to compute, we would have performed an extensive grid search to obtain more optimized model parameters and test a number of other baseline models such as SVM, KNN, etc. The baseline accuracies for the BQC accross languages can be seen in table 12 . As we can see the classifiers generally perform well on the training set, but a decent bit lower on the validation set. This could indidicate some degree of overfitting. To combat this several different strategies could be implemented like regularization, early stopping etc.

Looking at the validation performance, it seems that classifying Finnish questions was the easiest task by quite a margin. We hypothesize that this may have been a result of the uneven amount of data for the three langauges. The dataset included about twice as much data for Finnish than English and Japanese. Overall we can conclude that an accuracy of about $75 \%$ can be achieved fairly easily with a simple baseline model.

\section{Representation Learning}

The word2vec software of Tomas Mikolov et al. has provided state-of-the-art word embeddings for the last decade (Mikolov et al., 2013a,b). These embeddings can contain more complex information about the words like semantic and syntactic similarity. We could hope that this could improve the performance of our model - for example by more easily recognizing synonyms or closely related words used in both question and context.

We started by training a word2vec model over both context and question from the training data. The word2vec implementation is handled by the Gensim package. We chose a vector dimensionality of 100 . The original paper uses sizes of 20-300, and find that, as a general rule, higher dimensionality only yields better results when the training corpus is similarly increased in size (Mikolov et al., 2013a). Since our corpus is quite small, we found a dimensionality of 100 to be appropriate. For all other model parameters we used the Gensim package default. This meant training a CBOW model using negative sampling for 5 epochs with an initial learning rate of $0.025$ (with time-based decay).

In order to feed the word representations to our models, we need to fix the number of input features - even when document length varies. We achieved this by averaging over the individual word embeddings of the document/sentence for both question and context. For out-of-vocabulary words we returned a zero vector. The two document representations are concatenated to the input features of the baseline model. To the input we also added three similarity measures: Euclidian distance, cosine similarity and BERTScore (Zhang et al., 2019). See appendix $7.4$ for more elaboration on the metrics.

The performance of our extended BQC can be seen in table 13. Accuracy is improved on the training set but worsened on the validation set. The only exception is logistic regression which gained from the added features. We figure that the drop in performance stems from the models overfitting to the continuous representation, which was not expressive enough. Some of the context documents contain thousands of words, and it is simply impossible to compress all that information into a meaning-full representation of length 100 . Word2Vec also has the drawback that word representations are not context dependent. E.g. in the sentence: "The cell mate took samples of blood cells and took a call on his cell phone." the word cell has 3 different meanings, but would only be represented by a single embedding. Later we will see models which are capable of capturing this contextual information.

We also test a model using only the continuous vector representations. The result can be seen in table 14. For this configuration validation performance is much lower than the BOW baseline. For some model types the gap is as large as $10 \%$. We believe this is largely a result of the challenges just mentioned, but furthermore the absence of the feature quantifying BOW overlap. In section 4 we will show that overlap was one of the most crucial features for classifying samples as answerable.

That being said, the classifiers based exclusively on continuous features empirically verified our supposition that the three similarity measures where effective in capturing answerablility. Adding the metrics managed to improved performance by 2-3 percentage points on average. The inclusion or exclusion of these features where however not as influential for the other types of BQC's.

\section{Language modelling}

We chose to use GPT2 for language modeling as it has shown state-of-the-art results across many NLP disciplines (Radford et al., 2018) . GPT2 is an auto-regressive transformer-based language model built from decoder blocks. We finetune the pretrained models (one for each language) on the pairs of question and context from our dataset. Our choice of pretrained huggingface models were gpt2 (english), Finnish-NLP/gpt2-finnish (finnish), rinnaljapanese-gpt2-small (japanese). All these models scaled down versions of the original OpenAI model with $1.5 \mathrm{~B}$ parameters. The training data was provided in the format "Question: "+question+" \ncontext: "+ context. Traditionally one would use a separation token like [SEP] to signify a divide in the input. We choose to explicitly write the question/context relationship, hypothesizing that the model this way would have an easier time learning the structure of the data. We reason that similar examples like QA's have most likely been included in the pre-training data.

We fine-tune each model for 1 epoch on the associated data-set. Ideally one would run multiple epochs, until either training loss stagnates or validation performance decreases. The models were trained with 200 warmup steps, a weight decay of $0.01$, with the Adam optimizer and a learning rate of $0.00005$.

To deal with GPT2's max input size of 1024 tokens, we chose the head-only truncation strategy. One could potentially improve performance by looking into more advanced strategies for handling long texts. After all, head-only truncation can be limiting in circumstances where the answer lies at the end of a large context document. There are other methods for handling long texts, one possibility being selecting sections of text under some length constraint based on feature importance (Fiok et al., 2021). Or one could segment the input into smaller chunks with some overlap. After obtaining the LM outputs, one could propagate these through a single recurrent layer, or another transformer. Following this by a softmax activation one obtains the final classification decision (Pappagari et al., 2019).

In appendix $7.1 \mathrm{we}$ have added text samples from our fine-tuned models. These samples are generated by giving the prompt concatenated with one of the top 3 most common starting words found in section 1.1. We have generated 5 variations per starting word using top-k sampling with $k=50$ (Fan et al., 2018) and a max length of 50 .

It has seemingly learned to perfection that there always follows "\ncontext: " after "Question : " across languages. The questions posed are quite sound, but some of them are vague like "What is the largest currency?" or "What is the name of a new kind of animal?". That being said, the question and context are always semantically related, although most of the time the model generates context on some mildly related tangent. It might for example, ask how something started and give a context on how it ended. This could be a consequence of only sampling 50 tokens, i.e. the answer might have come later, but it is most likely a side effect of training on equal amounts of unanswerable and answerable questions. Overall the text generated seems logical and coherent.

In order to test the models performance we utilize the most widely-used language model evaluation metric named perplexity (Chen et al., 2008). It is defined as such:

$$
\operatorname{PP}_{T}\left(p_{M}\right)=\frac{1}{\left(\prod_{i=1}^{t} p_{M}\left(w_{i} \mid w_{1} \cdots w_{i-1}\right)\right)^{\frac{1}{t}}}
$$

Where $T=\left\{w_{1}, \ldots, w_{t}\right\}$ is the test set and $p_{M}$ is a language model. For an intuitive explanation, say a model has a perplexity of 20 . This means the model is as confused on the test data as if it had to choose uniformly and independently among 20 possibilities for each word. From the mathematical definition, and the intuitive explanation, it is clear that the lower bound on this value is 1 . It is however important to note, that lower perplexity does not always equal more human-like text (Kuribayashi et al., 2021).

In table 1 you can see perplexity values calculated over the validation data. They seem very reasonable. As a baseline comparison, a uniform language model (which obviously has learned nothing), would have a perplexity equal to the size of the vocabulary. These results are much to be preferred, and we are not very surprised given the coherence of the generated text. It seems the English model did slightly better than the Finnish and Japanese. However, It is difficult to fairly compare perplexity values across languages as vocabulary and token size are not constant. One thing is certain, it is not the amount of pre-training data that explains the variance. We initially suspected this to be the cause, as there is a very skewed distribution of global training data favoring English. But it turns out the Finnish model was trained on 84GB compared to an English corpus of $40 \mathrm{~GB}$. 

\begin{tabular}{c|c} 
& Validation ppl. \\
\hline English & $19.7$ \\
Finnish & $21.4$ \\
Japanese & $21.3$
\end{tabular}

Table 1: Perplexity of finetuned GPT2 on training and validation data.

The original paper showed slightly better results in the ppl. range 8-18 (Radford et al., 2018). We would expect ppl. to decrease as the model trained for more epochs, and approach similar values.

To improve our baseline BQC model, we also try to utilize the learned representation from our language models in the decision process. There are a number of ways to include the hidden representation. This could be the first hidden layer, last hidden layer, sum of all hidden layers, concatenations of the last 4 hidden layers and more (Devlin et al., 2019). We choose to extract the last hidden layer of the finetuned GPT2, as this performs well in the literature.

The performance achieved by replacing the input to the BQC with the hidden representation of length 768 can be seen in table 2. Since XGBoost was the preferred model type in all the previous $\mathrm{BQC}$ tasks, we focused exclusively on XGBoost for this configuration. It is to be noted, that usually for classification tasks like these, one would use a linear layer on top of the pooled LM output and finally a softmax. We used XGBoost to allow for a more direct and fair comparison with the previous BQC tasks.

\begin{tabular}{c|cc} 
& Train & Validation \\
\hline English & $79.8 \%$ & $77.6 \%$ \\
Finnish & $80.1 \%$ & $78.5 \%$ \\
Japanese & $83.7 \%$ & $77.8 \%$
\end{tabular}

Table 2: Accuracy for BQC using GPT2 hidden layer and XGBoost.

Performance is decent, but not competitive with the baseline BOW model. It is however much better than the previous continuous $\mathrm{BQC}$ from section 2. We believe this is a result of: 1 . A larger embedding (768 vs. 100) 2 . The ability to capture contextual information. 3. Much larger and more complex network. 4. Larger training corpus with both training and fine-tuning 5. The hidden-layer does not experience the same problem of being clouded by potentially thousands of characters of

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-04.jpg?height=520&width=551&top_left_y=231&top_left_x=1141)

Table 3: Normalised confusion matrix for logistic regression and random forest

context all averaged together. 6. Less over-fitting. The training and validation scores are very close.

\section{Error Analysis and Interpretability}

We will now conduct a detailed error analysis for two of the implemented models. To ensure a fair comparison, both models analysed will have been trained on the BOW representations extended with word2vec embeddings (see table 13). We will dissect the English logistic regression model which achieved a validation accuracy of $77.1 \%$ and the Finnish random forest model which achieved $64.7 \%$. These models are highlighted, for their clear gap in performance of $12.4$ percentage points, and the strong tools for explainability both model types offer.

In order to better explain the reasoning behind the classifications, we will plot feature importance for the two models. In appendix $7.5$ you will see the 20 most significant feature weights for LR. For the RF we visualize the 20 most crucial impurity-based feature importances. From these, it is clear that the engineered features have been a success. We find that overlap is the most highly weighted positive feature for the LR. One can also notice euclidean distance as the 10th most important feature in the LR and cosine similarity is the 18th most important feature for the RF.

It is also interesting that the word "stallion" being present in the context is highly indicative of non-answerability for the LR. This is a marker of potential overfitting, since there should be nothing intrinsically unanswerable about an uncastrated male horse. The training score of $99.8 \%$ verifies this, and shows that there is definitely room for improving the LR model.

Another interesting observation, is that the RF 

\begin{tabular}{c|cc} 
Instance & Answerable & LR \\
\hline $7.2 .1$ & False & True \\
$7.2 .2$ & True & False \\
$7.2 .3$ & False & True
\end{tabular}

Table 4: Results from adverserial instances

model attributes much more importance to the continuous representation of the context, with 17 out top 20 features being from the average pooled context. This could partially explain the difference in performance, since we have witnessed that models trained purely on word $2 \mathrm{vec}$ embeddings achieved low generalization for reasons discussed in section 2.

To get more insight on the similarities and common mistakes of the models, we present a confusion matrix in table 3 . One notices that the RF actually outperforms the LR by a small margin in more accurately avoiding false-negatives and correctly classifying true-negatives. That being said, we see a large discrepancy when it comes to predicting the positive class. The RF has trouble identifying true-positives and an over-tendency to predict false-positives compared to the LR.

We believe this could be a result of the RF's inability to learn BOW-overlap as a useful feature. This is consistent with LR weighting overlap much higher, and clearly being more proficient in recognizing true-positives.

Now we will test the robustness of the better BQC system with adversarial sequences in order to gage strengths and weaknesses. In appendix $7.2$ you will find three adversarial questions. For the first instance we construct a coherent question and context. The context however elaborates on a related tangent and does not address the question. We expect the model to be fooled by this instance, as the sentences have decent overlap, we have included no words which are weighted heavily toward the negative class, and the euclidean distance is likely small, since the two sentences are semantically related.

In the second, now answerable sample, we try to exploit the overfitting of the model. We have made sure to add 'stallion' to the context, 'american' to the question, and make overlap as small as possible. All of the above are correlated highly with the negative class.

The last question has a nonsensical context which simply repeats words from the question. This examples show the danger of relying too much on overlap to assess answer-ability.

As seen in table 4 we manage successfully to fool the model on every adversarial instance. This goes to show that our logistic regression model has a few weaknesses. One is that the model seems to be over-fitted on certain non-related tokens. More various remidies such as more training data or regularization could rectify this. And secondly, relying too much on overlap makes the model vulnerable to nonsensical contexts.

\section{Sequence Labelling}

Often one is not only interested in whether a question is answerable or not but what the actual answer is. To do this the task is often model as a span selection task. This setting is widely used as it is straightforward and effective approach for single answer questions. However, this approach has some limitations. Most notably that the context only can contain one answer despite other parts of the text also being valid answer. We therefore chose to model the answer span with the IOB-tagging system as this allows for multiple parts of the text to answer the question.

To perform token-classification tasks such as NERor POS-tagging a BiLSTM-CRF model is commonly used. We therefore implemented such a model composed of an embedding layer, two stacked Bi-LSTM layers, a linear layer and finally a CRF layer. For the embedding layer we chose to use the fasttext library (Grave et al., 2018) as it has pretrained vectors for many different languages. We selected a Bi-LSTM dimensionality of 128 and employed dropout for regularization ( $p=0.01$ ). We used an Adam optimizer and the Cyclic LR scheduler to schedule learning rates to speed up the training. We used a learning rate of $2 \cdot 10^{-5}$. We trained this model for 10 epochs with a batch size of 16 . The results can be seen in table 5. Here we see that it seems to learn something when looking at the performance on the train score, but when looking at the validation score for both English and Japanese it fails to predict the right sequence. However when looking at Finnish the validation score seems to be on par with the training score. This is accordance with the results we saw earlier where the $\mathrm{BQC}$ had a easier time figuring out whether the Finnish contexts contained the answer. 

\begin{tabular}{c|cc} 
& Train F1 & Validation F1 \\
\hline English & $0.329$ & $0.053$ \\
Finnish & $0.361$ & $0.326$ \\
Japanese & $0.330$ & $0.104$
\end{tabular}

Table 5: Training results for the BiLSTM-CRF

\begin{tabular}{c|ccc} 
Language / Beams & $\mathbf{1}$ & $\mathbf{2}$ & $\mathbf{3}$ \\
\hline English & $0.316$ & $0.172$ & $0.329$ \\
Finnish & $0.009$ & $0.015$ & $0.234$ \\
Japanese & $0.021$ & $0.091$ & $0.007$
\end{tabular}

Table 6: Validation results for different amounts of beams

One problem with the current implementation of the Bi-LSTM-CRF is that to find the optimal sequence of tags the Viterbi-algorithm is used. This can be computationally expensive and therefore is very time consuming. We therefore investigate the effects of beam-search, which is an alternate method for finding a probable optimal sequence of tags. However when performing beam search with the current model the probability distribution for the next step does not depend on the previous step's choice. A common solution to this is to employ beam search with an encoder-decoder architecture:

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-06.jpg?height=448&width=771&top_left_y=1558&top_left_x=243)

Figure 1: Illustration of the encoder-decoder architecture used for the BiLSTM-CRF with beam-search

The encoder consist of an embedding layer and two BiLSTMs layers with a dimension of 128 as before. Likewise the decoder also consist of an embedding layer and two BiLSTM layers. For training the same parameters was used as the previous BiLSTM model. With this trained model we try to predict the optimal sequence of tags with 1,2 and 3 beams. The results for each language can be seen at table 6 . And the times to decode can be seen at table 7.

As we can see the time to decode the optimal

\begin{tabular}{c|ccc} 
Language / Beams & $\mathbf{1}$ & $\mathbf{2}$ & $\mathbf{3}$ \\
\hline English & $3.5$ & $8.25$ & 17 \\
Finnish & 4 & 9 & $17.75$ \\
Japanese & $5.5$ & $13.25$ & 26
\end{tabular}

Table 7: Decode times (in minutes rounded to nearest $15 \mathrm{~s}$ ) for different amounts of beams

sequence decreases notably when the number of beams decreases. In theory when the number of beams increases the performance should also increase however when we look at our results this only seem to be true for Finnish. One thing is to remember is that the decoder tries to find the optimal sequence given the predicted probabilities. But like we saw with our previous Bi-LSTM-CRF and the current one the performance is not perfect. Therefore the probability mapping is not perfect which can be the reason for these results. One more thing to note is that doing exhaustive search contra greedy search is most important for problems with long-distance dependencies between output decisions. In this case whether it is part of the answer depends more on the surrounding tokens that if some earlier text was tagged as an answer. We therefore don't expect beam-search to have a big impact on the result which is what we somewhat notice with English and Japanese.

One interesting thing we however notice is that this sequence labeller seem to be able learn English considerably better than the last and here even outperforms Finnish.

A reason for the poor performance could be that the LSTM doesn't capture the long range dependency between the question and the different parts of the context well enough. We therefore also chose to fine-tune a pretrained transformer language model to do the tagging. In detail the model used was the BERT-model (Devlin et al., 2019) since this model performs competitively with state-of-theart methods on both NER-tagging and Q\&A. On top of the BERT model a linear layer is placed. We chose to omit the CRF layer like the original BERT paper. Since the BERT model splits words into word pieces we only use the representation of the first sub-token as the input to the token-level classifier. The pretrained models that was used was for all languages a BERT base model with $110 \mathrm{M}$ trainable parameters pretrained on text from its respective language. For this task we used the 

\begin{tabular}{c|cc} 
& Train F1 & Validation F1 \\
\hline English & $0.642$ & $0.497$ \\
Finnish & $0.865$ & $0.711$ \\
Japanese & $0.668$ & $0.562$
\end{tabular}

Table 8: Training results for the sequence tagger using BERT

same training parameters as for the BiLSTM-CRF for easy comparison in addition to 200 warm up steps and a weight decay of $0.01$. This model was also trained for 10 epochs with a learning rate of $2 \cdot 10^{5}$. The training results can be seen in table 8 This model performs significantly better but still not perfect. We therefore chose to do a qualitative inspection of some of the predicted answer spans to see if they makes sense and where it goes wrong. A few examples of predicted answers versus the correct one for different question can be seen in Appendix 7.3. What we generally notice is that it is a bit biased towards 'O' which makes somewhat sense due to the huge presence in the data. This could maybe be mitigated by oversampling questions with an actual answer. We also notice it generally has a hard time with answers only consisting of a date or a few words, which maybe can be explained by the earlier observation that the presence of the word both in the answer and in the question makes it more likely to be answerable. Moreover we notice that there might be some ambiguity in the data. For example one question ask what martial arts Marines learn. Here the correct answer is the Marine Corps Martial Arts Program. However our sequence labeller tagged the following answer "combine existing and new hand-to-hand and close quarters combat techniques with morale and team-building functions and instruction in the Warrior Ethos", which also makes sense as an answer to the question.

In general we found that it is indeed possible also to model Q\&A as a token-classification task. This may even fix the problem when the context contains multiple correct answer as we saw with the ambiguity of one of the questions. Our model was however long from perfect. It should however be noted that the model isn't the largest or finetuned on much data. Another problem is that we haven't done any fine tuning at all which possibly further could improve the model.

\begin{tabular}{c|ccc} 
Fine-tune/Eval & English & Finnish & Japanese \\
\hline English & $\mathbf{0 . 6 7 5}$ & $0.645$ & $0.63$ \\
Finnish & $0.64$ & $\mathbf{0 . 7 0 5}$ & $0.585$ \\
Japanese & $0.42$ & $0.555$ & $\mathbf{0 . 6 5 5}$
\end{tabular}

Table 9: Zero-shot cross-lingual accuracy for the multilingual $\mathrm{BQC}$

\section{Multilingual QA}

\subsection{Answerable TyDiQA BQC}

In order to extend the Answerable TyDiQA binary question from section 3 , we change the language model from a language-specific pretrained GPT2 to BLOOM (Margaret Mitchell and co., 2022). BLOOM is an autoregressive language model, trained to continue text from a prompt on more than 46 languages, including English, Finnish and Japanese. We trained the smallest version of BLOOM (560M parameters) with 200 warm up steps, a weight decay of $0.01$ and the Adam optimizer with a learning rate of $0.00005$. Due to hardware constraints (specifically RAM limitations) we were only able to run the model with a batch-size of 1 , which in turn made time-constraints an issue. The results below are therefore based on training and validation of only 300 balanced samples each. It is to be stated clearly, that is of course not optimal. One would normally train for multiple epochs, and until no significant improvements in loss are to be found (or no improvement in performance on the validation set).

From our earlier BQC models, we have seen that XGBoost is very performant on the classification tasks. But since XGboost is quite memory consuming, we resorted to using Logistic regression as the predictor for this task.

From table 9 we see that the zero-shot matrix non-theless looks very much like how you would expect. We have highlighted the best performing configuration for each row, and the diagonal is clearly to be preferred.

It seems that Japanese is the training language which results in the least amount of generalization. After fine-tuning, it actually performed worse than simply guessing randomly on the English data-set. And again, we observe Finnish to be by far the easiest language for determining answerability. This time the phenomenon certainly cannot be explained by a larger training set. It may be that the morphology of Finnish is somehow easier to grasp for the models we investigate. We also notice 

\begin{tabular}{c|ccc} 
Fine-tune/Eval & English & Finnish & Japanese \\
\hline English & $\mathbf{0 . 4 4 7}$ & $0.437$ & $0.426$ \\
Finnish & $0.515$ & $\mathbf{0 . 6 7 7}$ & $0.443$ \\
Japanese & $0.470$ & $0.503$ & $\mathbf{0 . 5 7 2}$
\end{tabular}

Table 10: Zero-shot cross-lingual evaluation for the sequence labeller from section 5

that fine-tuning on English achieved the best on average across the languages. However, the results should be taken with a grain of salt. Given the low count of training and validation samples, all values are associated with a high degree of uncertainty. To conclude, there are many ways in which performance on this task could be improved. Training on more data, larger language model (BLOOM can get as large as 7B parameters), more sophisticated final classifier (a larger neural network, XGBoost etc.), more deliberate choice of hyperparameters and so on. It is well known that deep learning models needs a great deal of data to achieve good performance, but the fact that one already can get around $65 \%$ accuracy on only 300 samples is truly a testament to the power of pretrained models.

\subsection{IOB tagging system}

For the answer extraction we will use the model utilising a pre-trained transformer architecture since this model was superior in the previous analysis. We will use the same architecture and the same configuration as before. However for this task we will use mBERT as our pretrained model. The results of the zero-shot cross-lingual evaluation can be seen in table 10 .

We have again highlighted the best configuration for each row. We notice it performs the best on the language it was pretrained on which is in accordance with our expectations. We also notice that the performance is somewhat the same inpendent on what language it was pretrained on. The performance is however a bit lower than when using a language specific pretrained model which is to be expected. The performance when finetuning on Japanese is however slightly better. We hypothesise that this is because the Japanese texts often also contains a bit of English

\section{References}

Steven Bird, Ewan Klein, and Edward Loper. 2009. Natural language processing with Python: analyzing text with the natural language toolkit. " O'Reilly Media, Inc.".

Stanley F Chen, Douglas Beeferman, and Roni Rosenfeld. 2008. Evaluation Metrics For Language Models.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pages 785-794, New York, NY, USA. ACM.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation.

Krzysztof Fiok, Waldemar Karwowski, Edgar GutierrezFranco, Mohammad Reza Davahli, Maciej Wilamowski, Tareq Ahram, Awad Al-Juaid, and Jozef Zurada. 2021. Text guide: Improving the quality of long text classification by a text selection method based on feature importance. IEEE Access, 9:105439-105450.

Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin, and Tomas Mikolov. 2018. Learning word vectors for 157 languages. In Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018).

Tatsuki Kuribayashi, Yohei Oseki, Takumi Ito, Ryo Yoshida, Masayuki Asahara, and Kentaro Inui. 2021. Lower perplexity is not always human-like. In Proceedings of the 59 th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5203-5217, Online. Association for Computational Linguistics.

Yacine Jernite Margaret Mitchell, Giada Pistilli and co. 2022. Bigscience large open-science open-access multilingual language model.

Paul McCann. 2020. fugashi, a tool for tokenizing Japanese in python. In Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS), pages 44-51, Online. Association for Computational Linguistics.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. Efficient estimation of word representations in vector space. CoRR, $a b s / 1301.3781$.

Tomas Mikolov, Ilya Sutskever, Gregory S. Corrado Kai Chen, , and Jeffrey Dean. 2013b. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems.

Raghavendra Pappagari, Piotr Żelasko, Jesús Villalba, Yishay Carmiel, and Najim Dehak. 2019. Hierarchical transformers for long document classification.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2018. Language models are unsupervised multitask learners.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. 

\section{Appendix}

\subsection{Sampled Sentences}

\subsubsection{English}

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-10.jpg?height=2234&width=736&top_left_y=461&top_left_x=269)

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-10.jpg?height=1783&width=726&top_left_y=247&top_left_x=1088)

\subsubsection{Finnish}

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-10.jpg?height=573&width=737&top_left_y=2119&top_left_x=1087)



![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-11.jpg?height=417&width=1205&top_left_y=228&top_left_x=425)

Table 11: Top 5 most common first and last tokens for each language

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-11.jpg?height=319&width=1340&top_left_y=740&top_left_x=358)

Table 12: BOW baseline accuracies on training (T) and validation (V) data.

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-11.jpg?height=319&width=1340&top_left_y=1154&top_left_x=358)

Table 13: Accuracy for $\mathrm{BQC}$ extended with continuous representation.

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-11.jpg?height=317&width=1356&top_left_y=1566&top_left_x=356)

Table 14: Accuracy for models using only continuous vector representation
![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-11.jpg?height=642&width=1542&top_left_y=2022&top_left_x=272)

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-12.jpg?height=2297&width=725&top_left_y=250&top_left_x=270)

\subsubsection{Japanese}

Note: The original Japanese generated text does have newlines, but the latex package rendering the characters does not recognize this.

0: question: 日本における、自動車の最大排気

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-12.jpg?height=54&width=774&top_left_y=521&top_left_x=1052)
に、2が軽自動車の各社に摘用される。そL て現在では、日本自動車協会は排気量と車 名-車種で 1 : question: 日本自衛隊はいつ誕 生した context: 戦後自衛隊創設以降、国主導 の下で防衛組織の整備と防衛機能の充龶が龱 られており、自衛隊組織が「自衛隊」として 独立する事態にはなっていない[38] 2: question: 日本国鉄はいつ創立した? context: 国鉄 は、日本最大の私鉄で、日本の鉄道事業者で は第 2 位の規模を持つ日本唯一の鉄道事業者 である。1895年(明治28年)に、東京 - 日本橋 3: question: 日本の一人当た $\mathrm{gdp}$ は? context: gdp (english) は、アメリカ合衆国の数值。世 界の国民総生産の内訳を取りまとめた世界 経済予測を基に算出され、各国のgdpを比較 Lた際、4: question: 日本サッカーはいつ始 まった? context: サッカー競技は、日本の国 民競技で歴史あろ競技の一つ[1][2][3][4]。 日 本文化の発祥地ともいわれる。0: question: 『銀河英雄伝説 新風」の舞台となった場所 はどこ context: 銀河英雄佶説の中心人物と Lて活躍した「英雄」と「佶説の英雄」と の因縁に思いをいだき、仲間として彼らと 共に銀河 1: question: 『赤マルジ $マ ン フ ゚ 』$ の視聴率は? context: 2011年9月 5 日 $\sim 9$ 月 18 日 に、nhk総合テレビで再放送された。2: question: 『バットマン』はいつ書かれた? context: 1993年12月2日の 『映画『バットマン』 に主演することになったレ $゚$ に チャールズ は、後に『バットマン: バタイヤー』で主 演することを約束する 3: question: 『キャ プテンハーロック』の監督はだれ context: 『キャプテンハーロック』の脚本を担当し たのは映画監督のスティーヴ・クロッパー である。脚本についてクロッパーは「この映 画の制作に重要な影響を与えた」と 4: question: 『ポケットモンスター サン・ムーン』 のストーリーは? context: 『ポケットモン スターウルトラサン』.『ポケットモンス ターブラック・ホフイト』に凳場したポケ モンのボスと闒レます。 0: question: アメリ $\mathrm{~ 力 連 邦 準 備 理 事 会 ( f r b ) は い つ 結 成 し た ~ c o n t e}$ 2007年11月11日の声明では、連邦準備理事会 法に基づぃて、連邦公開市場委員会(fomc)の

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-12.jpg?height=57&width=774&top_left_y=2636&top_left_x=1052)

1) カ南北戦争での南軍の勝利数は? context: 1861年1月2日に、北軍のジェイ4ズ・ヘン 1ーが南軍側に加わっていたLキシントンの 戦いで、フラングリン・d.c・マクヘン リー

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-13.jpg?height=54&width=769&top_left_y=430&top_left_x=238)

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-13.jpg?height=60&width=777&top_left_y=473&top_left_x=231)
の国旗(アメリカがっLゅうこくのこっき)と

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-13.jpg?height=54&width=780&top_left_y=573&top_left_x=227)
衆国黄色を使用する連邦の国旗であり、ア メリカ合衆国のシンボルでかる。 3: question:

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-13.jpg?height=57&width=768&top_left_y=711&top_left_x=244)
る? context: アメリカ最大のイン ディアン居 留地メンフィスは現在8の部族が住む、3つの インディアン居留地が連結しており、この 地区はメンフィス州の州都メンフィス貝に

![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-13.jpg?height=57&width=780&top_left_y=945&top_left_x=227)
×リカワシントン州出身のバスケットボー ル 選手は誰? context: bj リーグの秋田ーザ ンハピネッツに4年目にして入団し、4年間プ $L-L た 。$ 秋田がhcとなった2011-12シーズ ンは7勝1敗、同シーズン終了後には

\subsection{Adverserial questions}

\subsection{1}

Q: When was Queen Elizabeth II born?

A: Queen Elizabeth II was the queen of England from 1952 until she died on the 8th of September 2022.

Type: Not answerable.

Source.

\subsection{2}

Q: What is an uncastrated male horse called in American English?

A: A horse of masculine gender which has not been castrated can be referred to as a stallion in American English

Type: Answerable.

Source.

\subsection{3}

Q: How can fast does the earth spin around its own axis?

A: Earth earth earth earth spin spin spin spin spin around around around around

Type: Not answerable.

\subsection{Examples of sequence labeller}

Question: When was github created?

Answer: February 2008

Predicted answers: [] Question: Who was the first leader of West Germany?

Answer: Theodor Heuss

Predicted answers: ['Theodor Heuss']

Question: What martial arts do Marines learn? Answer: Marine Corps Martial Arts Program Predicted answers: ['combine existing and new hand-to-hand and close quarters combat techniques with morale and team-building functions and instruction in the Warrior Ethos', '[', 'which began in 2001, trains Marines', 'of']

Question: What is the world's largest horse show?

Answer: Devon Horse Show

Predicted answers: ['Since 1896 , the Devon Horse', 'is']

Question: When did the first episode of Big Brother Australia air?

Answer:

Predicted answers: []

Question: When was Nike founded?

Answer: January 25, 1964

Predicted answers: [] out?

Question: When did Final Fantasy Type- 0 come

Answer:

Predicted answers: []

Question: What is the oldest city in Myanmar? Answer: Beikthano

Predicted answers: []

Question: What is the strongest recorded wind? Answer: was during the passage of Tropical Cyclone Olivia on 10 April 1996: an automatic weather station on Barrow Island, Australia, registered a maximum wind gust of $408 \mathrm{~km} / \mathrm{h}$ Predicted answers: []

Question: Who was president in $1817 ?$

Answer:

Predicted answers: [] 

\subsection{Similarity Measures}

\section{Euclidean distance}

$$
\left\|q_{\text {avg }}-c_{\text {avg }}\right\|
$$

\section{Cosine similarity}

$$
\frac{q_{\mathrm{avg}} \cdot c_{\mathrm{avg}}}{\left\|q_{\mathrm{avg}}\right\|\left\|c_{\mathrm{avg}}\right\|}
$$

\section{BERTScore}

$$
\begin{gathered}
R_{\mathrm{BERT}}=\frac{1}{|q|} \sum_{q_{i} \in q} \max _{c_{j} \in c} q_{i}^{\top} c_{j} \\
P_{\mathrm{BERT}}=\frac{1}{|c|} \sum_{c_{j} \in c} \max _{q_{i} \in q} q_{i}^{\top} c_{j} \\
F_{\mathrm{BERT}}=2 \frac{P_{\mathrm{BERT}} \cdot R_{\mathrm{BERT}}}{P_{\mathrm{BERT}}+R_{\mathrm{BERT}}}
\end{gathered}
$$

Where $q_{\text {avg }}$ and $c_{\text {avg }}$ are the average question and context representations, and $q_{i}$ and $c_{j}$ represent the individual word embeddings that the documents are composed of. In BERTScore it is also presumed that $q_{i}, c_{j}$ have been normalized to unit length. We hypothesized that these similarity measures could increase performance in a similar manor as to the simple overlap feature from section 1.2. The first measure, euclidean distance, does provide some indication of whether the two embeddings are close in the representational space. But it has the drawback, that occurrence counts impact vector magnitudes, meaning that semantic similarity is better captured in the angle between embeddings. This is exactly what the second measure, cosine similarity, achieves, and why it is the default indicator of like-ness for word embeddings. The last similarity measure is BERTScore (Zhang et al., 2019). BERTScore is traditionally used as an automatic evaluation metric for text generation. BERTScore works by computing a similarity score for each token in the candidate sentence with each token in the reference sentence. However, instead of exact matches, it relies on token similarity using contextual embeddings.

Even though we are not doing text generation, we can nonetheless utilize this same method with word2vec embeddings rather than BERT to obtain a more nuanced similarity metric.

\subsection{Feature importance plots}



![](https://cdn.mathpix.com/cropped/2022_11_04_0407e104e8e4530a9e83g-16.jpg?height=1999&width=1128&top_left_y=505&top_left_x=384)

Table 16: Random forest feature importance
