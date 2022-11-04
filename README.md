NLP 2022/2023: Group Project

Christian Mølholt Jensen Axel Højmark Christoffer Ringgaard

vdj579 xbm265 cwl852

1. Preprocessing and Dataset Analysis

To familiarize ourselves with the dataset and prepare it for classification, we firstly preprocessthe data by tokenizing at word level, stemming and removing stop-words for all languages. The implementation of the preprocessing vary across the languages in accordance to their respective morphologies. We used NLTK ([Bird et al.,](#_page7_x65.89_y749.11) [2009) ](#_page7_x65.89_y749.11)for English and Finnish and the Fugashi package [(McCann, 2020)](#_page7_x301.16_y621.81) for Japanese.

We also noticed an imbalance in the class distribution throughout the data. We therefore chose to balance the number of negative and positive samples for each language. This helps mitigate bias in the classifierand makes accuracy a meaning-full measure of performance.

We begin by investigating the ways in which the sentences from our dataset begin and end. Unsurprisingly, one observes from table[ 11](#_page10_x70.87_y64.09) that the most observed initial words are common question words. Some of the last words for English may look surprising in their apparent randomness, for example ’zombie’. However the total frequency of the last english words ranking 2-5 is approximately 0.1%. This means that, as expected, question mark is the finaltoken in almost all sentences. The story is almost identical for Finnish, i.e. similar question words rank top 5 (altough in a slightly different order) and the question mark makes up almost all the last tokens of the questions.

Among the top 5 last words of Japanese, one can find question mark, ’when’ (いつ ), ’where’ (どこ ) and ’what’ (何 ). However Japanese is not as predictable in its positioning of question words. Inspecting the second to last token, we also observed question words like ’who’ (誰), ’what’ (何) and the the common question particle ’ka’ (か ) ranked among the most frequent tokens. The first tokens of Japanese are mostly nouns like America or Japan. This seems reasonable knowing

that Japanese is a SOV (Subject-Object-Verb) language.

2. Binary Question Classification

To get a good baseline for further investigation into more complex models, we create a simple Binary Question Classfier (BQC) predicting whether a question is answerable given the context. The model was trained on concatenated bag of words (BOW) representations of the context and the question. Moreover we added a count of the number of overlapping words as an extra feature. For all the features above we used cleaned texts. One could however argue that by dropping question words like ’How’ or ’Where’ we lose critical information about the type of question and therefore whether the context includes the answer. To circumvent this, we used the observation from our data exploration that almost all the questions begin with question words, and alsoaddedaone-hotvectorencodingthefirstword.

We chose to test four different model archi- tectures for the BQC: Logistic Regression, RandomForest, Multi-layer Perceptron classifier (MLP) and XGBoost ([Chen and Guestrin,](#_page7_x301.16_y143.37) [2016). ](#_page7_x301.16_y143.37)We used sklearn’s default parameters for all models. The only exceptions are we changed the MLP model architecture to 2 layers of 50 fully connected neurons, and gave a max depth of 10 to the RandomForest to combat overfitting. Given more access to compute, we would have performed an extensive grid search to obtain more optimized model parameters and test a number of other baseline models such as SVM, KNN, etc. The baseline accuracies for the BQC accross languages can be seen in table [12.](#_page10_x70.87_y210.22) As we can see the classifiers generally perform well on the training set, but a decent bit lower on the validation set. This could indidicate some degree of overfitting. To combat this several different strategies could be

implementedlikeregularization, earlystoppingetc.

Looking at the validation performance, it seems that classifying Finnish questions was the easiest task by quite a margin. We hypothesize that this may have been a result of the uneven amount of data for the three langauges. The dataset included about twice as much data for Finnish than English and Japanese. Overall we can conclude that an accuracy of about 75% can be achieved fairly easily with a simple baseline model.

2 Representation Learning

The word2vec software of Tomas Mikolov et al. has provided state-of-the-art word embeddings for the last decade ([Mikolov et al.,](#_page7_x301.16_y685.46) [2013a,](#_page7_x301.16_y685.46)[b). ](#_page7_x301.16_y727.19)These embeddings can contain more complex information about the words like semantic and syntactic similarity. We could hope that this could improve the performance of our model - for example by more easily recognizing synonyms or closely related words used in both question and context.

We started by training a word2vec model over both context and question from the training data. The word2vec implementation is handled by the Gensim package. We chose a vector dimensionality of 100. The original paper uses sizes of 20-300, and find that, as a general rule, higher dimensionality only yields better results when the training corpus is similarly increased in size ([Mikolov et al.,](#_page7_x301.16_y685.46) [2013a).](#_page7_x301.16_y685.46) Since our corpus is quite small, we found a dimensionality of 100 to be appropriate. For all other model parameters we used the Gensim package default. This meant training a CBOW model using negative sampling for 5 epochs with an initial learning rate of 0.025 (with time-based decay).

In order to feed the word representations to our models, we need to fix the number of inputfeatures - even when document length varies. We achieved this by averaging over the individual word embeddings of the document/sentence for both question and context. For out-of-vocabulary words we returned a zero vector. The two document representations are concatenated to the input features of the baseline model. To the input we also added three similarity measures: Euclidian

distance, cosine similarity and BERTScore [(Zhang et al., 2019](#_page8_x65.89_y185.44)). See appendix[ 7.4](#_page13_x70.87_y70.87) for more elaboration on the metrics.

The performance of our extended BQC can be seen in table [13.](#_page10_x70.87_y329.26) Accuracy is improved on the training set but worsened on the validation set. The only exception is logistic regression which gained from the added features. We figure that the drop in performance stems from the models overfittingto the continuous representation, which was not expressive enough. Some of the context documents contain thousands of words, and it is simply impossible to compress all that information into a meaning-full representation of length 100. Word2Vec also has the drawback that word representations are not context dependent. E.g. in the sentence: "The cell mate took samples of blood cells and took a call on his cell phone." the word cell has 3 different meanings, but would only be represented by a single embedding. Later we will see models which are capable of capturing this contextual information.

We also test a model using only the continu- ous vector representations. The result can be seen in table [14.](#_page10_x70.87_y448.29) For this configurationvalidation performance is much lower than the BOW baseline. For some model types the gap is as large as 10%. We believe this is largely a result of the challenges just mentioned, but furthermore the absence of the feature quantifying BOW overlap. In section[ 4 ](#_page3_x306.14_y304.46)we will show that overlap was one of the most crucial features for classifying samples as answerable. That being said, the classifiers based exclusively on continuous features empirically verified our supposition that the three similarity measures where effective in capturing answerablility. Adding the metrics managed to improved performance by 2-3 percentage points on average. The inclusion or exclusion of these features where however not as influentialfor the other types of BQC’s.

3 Language modelling

We chose to use GPT2 for language modeling as it has shown state-of-the-art results across many NLP disciplines ([Radford et al.,](#_page8_x65.89_y143.59) [2018)](#_page8_x65.89_y143.59) . GPT2 is an auto-regressive transformer-based language model built from decoder blocks. We finetunethe pretrained models (one for each language) on the pairs of question and context from our dataset.

Our choice of pretrained huggingface models were gpt2 (english), Finnish-NLP/gpt2-finnish (finnish), rinna/japanese-gpt2-small (japanese). All these models scaled down versions of the original OpenAI model with 1.5B parameters. The training data was provided in the format "Question: "+question+"\nContext: "+

context. Traditionally one would use a separation token like [SEP] to signify a divide in the input. We choose to explicitly write the question/context relationship, hypothesizing that the model this way would have an easier time learning the structure of the data. We reason that similar examples like QA’s have most likely been included in the pre-training data.

We fine-tune each model for 1 epoch on the associated data-set. Ideally one would run multiple epochs, until either training loss stagnates or validation performance decreases. The models were trained with 200 warmup steps, a weight decay of 0.01, with the Adam optimizer and a learning rate of 0.00005.

To deal with GPT2’s max input size of 1024 tokens, we chose the head-only truncation strategy. One could potentially improve performance by looking into more advanced strategies for handling long texts. After all, head-only truncation can be limit- ing in circumstances where the answer lies at the end of a large context document. There are other methods for handling long texts, one possibility being selecting sections of text under some length constraint based on feature importance [(Fiok et al.](#_page7_x301.16_y345.28), [2021](#_page7_x301.16_y345.28)). Or one could segment the input into smaller chunks with some overlap. After obtaining the LM outputs, one could propagate these through a single recurrent layer, or another transformer. Fol- lowing this by a softmax activation one obtains the finalclassificationdecision[(Pappagarietal.](#_page8_x65.89_y101.75),[2019](#_page8_x65.89_y101.75)).

In appendix [7.1 ](#_page9_x70.87_y92.56)we have added text samples from our fine-tuned models. These samples are generated by giving the prompt "Question: " concatenated with one of the top 3 most common starting words found in section [1.1.](#_page0_x70.87_y212.60) We have generated 5 variations per starting word using top-k sampling with k = 50 [(Fan et al., 2018)](#_page7_x301.16_y314.51) and a max length of 50.

It has seemingly learned to perfection that there always follows "\nContext: " after "Question : " across languages. The questions posed are quite sound, but some of them are vague like

"What is the largest currency?" or "What is the name of a new kind of animal?". That being said, the question and context are always semantically related, although most of the time the model generates context on some mildly related tangent. It might for example, ask how something started and give a context on how it ended. This could be a consequence of only sampling 50 tokens, i.e. the answer might have come later, but it is most likely a side effect of training on equal amounts of unanswerable and answerable questions. Overall the text generated seems logical and coherent.

In order to test the models performance we utilize the most widely-used language model evaluation metric named perplexity ([Chen et al., 2008](#_page7_x301.16_y101.64)). It is definedas such:

1

PPT (pM ) =

t pM (wi | w1 ···wi−1) 1t i=1

Where T = {w1,...,wt} is the test set and pM is a language model. For an intuitive explanation, say a model has a perplexity of 20. This means the model is as confused on the test data as if it had to choose uniformly and independently among 20 possibilities for each word. From the mathematical definition,and the intuitive explanation, it is clear that the lower bound on this value is 1. It is however important to note, that lower perplexity does not always equal more human-like text [(Kuribayashi et al., 2021](#_page7_x301.16_y483.54)).

In table [1 ](#_page3_x70.87_y64.09)you can see perplexity values cal- culated over the validation data. They seem very reasonable. As a baseline comparison, a uniform language model (which obviously has learned nothing), would have a perplexity equal to the size of the vocabulary. These results are much to be preferred, and we are not very surprised given the coherence of the generated text. It seems the English model did slightly better than the Finnish and Japanese. However, It is difficult to fairly compare perplexity values across languages as vocabulary and token size are not constant. One thing is certain, it is not the amount of pre-training data that explains the variance. We initially suspected this to be the cause, as there is a very skewed distribution of global training data favoring English. But it turns out the Finnish model was trained on 84GB compared to an English corpus of 40GB.

Validation ppl.![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.001.png)English 19.7 Finnish 21.4

Japanese 21.3

Table 1: Perplexity of finetunedGPT2 on training and validation data.

The original paper showed slightly better results in the ppl. range 8-18 ([Radford et al.,](#_page8_x65.89_y143.59) [2018).](#_page8_x65.89_y143.59) We would expect ppl. to decrease as the model trained for more epochs, and approach similar values.

To improve our baseline BQC model, we also try to utilize the learned representation from our lan- guage models in the decision process. There are a number of ways to include the hidden represen- tation. This could be the first hidden layer, last hidden layer, sum of all hidden layers, concatena- tions of the last 4 hidden layers and more ([Devlin et al., 2019](#_page7_x301.16_y207.02)). We choose to extract the last hidden layer of the finetunedGPT2, as this performs well in the literature.

The performance achieved by replacing the input to the BQC with the hidden representation of length 768 can be seen in table [2.](#_page3_x70.87_y526.23) Since XGBoost was the preferred model type in all the previous BQC tasks, we focused exclusively on XGBoost for this configuration. It is to be noted, that usually for clas- sification tasks like these, one would use a linear layer on top of the pooled LM output and finally a softmax. We used XGBoost to allow for a more direct and fair comparison with the previous BQC tasks.

Train Validation English![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.002.png) 79.8% 77.6% Finnish 80.1% 78.5%

Japanese 83.7% 77.8%

Table 2: Accuracy for BQC using GPT2 hidden layer and XGBoost.

Performance is decent, but not competitive with the baseline BOW model. It is however much better than the previous continuous BQC from section [2.](#_page1_x70.87_y257.43) We believe this is a result of: 1. A larger em- bedding (768 vs. 100) 2. The ability to capture contextual information. 3. Much larger and more complex network. 4. Larger training corpus with both training and fine-tuning5. The hidden-layer does not experience the same problem of being clouded by potentially thousands of characters of

True Class

p n

p LR 0.37 LR 0.13 RF 0.23![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.003.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.004.png) RF 0.28

n LR 0.10 LR 0.40 RF 0.07![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.005.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.006.png) RF 0.42

PredictedClass

Table 3: Normalised confusion matrix for logistic re- gression and random forest

context all averaged together. 6. Less over-fitting. The training and validation scores are very close.

4 Error Analysis and Interpretability

We will now conduct a detailed error analysis for two of the implemented models. To ensure a fair comparison, both models analysed will have been trained on the BOW representations extended with word2vec embeddings (see table [13).](#_page10_x70.87_y329.26) We will dissect the English logistic regression model whichachievedavalidationaccuracyof77.1%and the Finnish random forest model which achieved 64.7%. These models are highlighted, for their clear gap in performance of 12.4 percentage points, and the strong tools for explainability both model types offer.

In order to better explain the reasoning behind the classifications,we will plot feature importance for the two models. In appendix [7.5 ](#_page13_x70.87_y722.88)you will see the 20 most significantfeature weights for LR. For the RF we visualize the 20 most crucial impurity-based feature importances. From these, it is clear that the engineered features have been a success. We find that overlap is the most highly weighted positive feature for the LR. One can also notice euclidean distance as the 10th most important feature in the LR and cosine similarity is the 18th most important feature for the RF.

It is also interesting that the word "stallion" be- ing present in the context is highly indicative of non-answerability for the LR. This is a marker of potential overfitting, since there should be noth- ing intrinsically unanswerable about an uncastrated male horse. The training score of 99.8% verifies this, and shows that there is definitely room for improving the LR model.

Another interesting observation, is that the RF

Instance Answerable LR![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.007.png)

1. False True
1. True False
1. False True

Table 4: Results from adverserial instances

model attributes much more importance to the continuous representation of the context, with 17 out top 20 features being from the average pooled context. This could partially explain the difference in performance, since we have witnessed that models trained purely on word2vec embeddings achieved low generalization for reasons discussed in section[ 2.](#_page1_x70.87_y257.43)

To get more insight on the similarities and common mistakes of the models, we present a confusion matrix in table [3.](#_page3_x306.14_y64.09) One notices that the RF actually outperforms the LR by a small margin in more accurately avoiding false-negatives and correctly classifying true-negatives. That being said, we see a large discrepancy when it comes to predicting the positive class. The RF has trouble identifying true-positives and an over-tendency to predict false-positives compared to the LR.

We believe this could be a result of the RF’s inability to learn BOW-overlap as a useful feature. This is consistent with LR weighting overlap much higher, and clearly being more proficientin recognizing true-positives.

Now we will test the robustness of the bet- ter BQC system with adversarial sequences in order to gage strengths and weaknesses. In appendix [7.2](#_page12_x70.87_y363.89) you will find three adversarial questions. For the first instance we construct a coherent question and context. The context however elaborates on a related tangent and does not address the question. We expect the model to be fooled by this instance, as the sentences have decent overlap, we have included no words which are weighted heavily toward the negative class, and the euclidean distance is likely small, since the two sentences are semantically related.

In the second, now answerable sample, we try to exploit the overfittingof the model. We have made sure to add ’stallion’ to the context, ’american’ to the question, and make overlap as small as possible. All of the above are correlated highly with the negative class.

The last question has a nonsensical context which

simply repeats words from the question. This examples show the danger of relying too much on overlap to assess answer-ability.

As seen in table [4 ](#_page4_x70.87_y64.09)we manage successfully to fool the model on every adversarial instance. This goes to show that our logistic regression model has a few weaknesses. One is that the model seems to be over-fitted on certain non-related tokens. More various remidies such as more training data or regularization could rectify this. And secondly, relying too much on overlap makes the model vulnerable to nonsensical contexts.

5 Sequence Labelling

Often one is not only interested in whether a ques- tion is answerable or not but what the actual an- swer is. To do this the task is often model as a span selection task. This setting is widely used as it is straightforward and effective approach for single answer questions. However, this approach has some limitations. Most notably that the context only can contain one answer despite other parts of thetextalsobeingvalidanswer. Wethereforechose to model the answer span with the IOB-tagging sys- tem as this allows for multiple parts of the text to answer the question.

To perform token-classificationtasks such as NER- or POS-tagging a BiLSTM-CRF model is com- monly used. We therefore implemented such a model composed of an embedding layer, two stacked Bi-LSTM layers, a linear layer and finally a CRF layer. For the embedding layer we chose to use the fasttext library ([Grave et al.,](#_page7_x301.16_y419.89) [2018)](#_page7_x301.16_y419.89) as it has pretrained vectors for many different languages. We selected a Bi-LSTM dimensionality of 128 and employed dropout for regularization (p = 0.01). We used an Adam optimizer and the Cyclic LR scheduler to schedule learning rates to speed up the training. We used a learning rate of 2 ·10−5. We trained this model for 10 epochs with a batch size of 16. The results can be seen in table [5.](#_page5_x70.87_y64.09) Here we see that it seems to learn something when looking at the performance on the train score, but when looking at the validation score for both English and Japanese it fails to predict the right sequence. How- ever when looking at Finnish the validation score seems to be on par with the training score. This is accordance with the results we saw earlier where the BQC had a easier time figuringout whether the Finnish contexts contained the answer.

Train F1 Validation F1![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.008.png)

English 0.329 0.053 Finnish 0.361 0.326 Japanese 0.330 0.104

Table 5: Training results for the BiLSTM-CRF

Language / Beams 1 2 3 English![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.009.png) 0.316 0.172 0.329 Finnish 0.009 0.015 0.234 Japanese 0.021 0.091 0.007

Table 6: Validation results for different amounts of beams

One problem with the current implementation of the Bi-LSTM-CRF is that to find the optimal sequenceoftagstheViterbi-algorithmisused. This can be computationally expensive and therefore is very time consuming. We therefore investigate the effects of beam-search, which is an alternate method for findinga probable optimal sequence of tags. However when performing beam search with the current model the probability distribution for the next step does not depend on the previous step’s choice. A common solution to this is to employ beam search with an encoder-decoder architecture:

![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.010.jpeg)

Figure 1: Illustration of the encoder-decoder architec- ture used for the BiLSTM-CRF with beam-search

The encoder consist of an embedding layer and two BiLSTMs layers with a dimension of 128 as before. Likewise the decoder also consist of an em- beddinglayerandtwoBiLSTMlayers. Fortraining the same parameters was used as the previous BiL- STM model. With this trained model we try to predict the optimal sequence of tags with 1, 2 and 3 beams. The results for each language can be seen at table[ 6.](#_page5_x70.87_y152.01) And the times to decode can be seen at table[ 7.](#_page5_x306.14_y64.09)

As we can see the time to decode the optimal

Language / Beams 1 2 3 English 3.5 8.25 17

Finnish 4 9 17.75 Japanese![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.011.png) 5.5 13.25 26

Table 7: Decode times (in minutes rounded to nearest 15s) for different amounts of beams

sequence decreases notably when the number of beams decreases. In theory when the number of beams increases the performance should also increase however when we look at our results this only seem to be true for Finnish. One thing is to remember is that the decoder tries to find the optimal sequence given the predicted probabilities. But like we saw with our previous Bi-LSTM-CRF and the current one the performance is not perfect. Therefore the probability mapping is not perfect which can be the reason for these results. One more thing to note is that doing exhaustive search contra greedy search is most important for problems with long-distance dependencies between output decisions. In this case whether it is part of the answer depends more on the surrounding tokens that if some earlier text was tagged as an answer. We therefore don’t expect beam-search to have a big impact on the result which is what we somewhat notice with English and Japanese.

One interesting thing we however notice is that this sequence labeller seem to be able learn English considerably better than the last and here even outperforms Finnish.

A reason for the poor performance could be that the LSTM doesn’t capture the long range dependency between the question and the different parts of the context well enough. We therefore also chose to fine-tunea pretrained transformer language model to do the tagging. In detail the model used was the BERT-model ([Devlin et al.,](#_page7_x301.16_y207.02) [2019)](#_page7_x301.16_y207.02) since this model performs competitively with state-of-the- art methods on both NER-tagging and Q&A. On top of the BERT model a linear layer is placed. We chose to omit the CRF layer like the original BERT paper. Since the BERT model splits words into word pieces we only use the representation of the first sub-token as the input to the token-level classifier. The pretrained models that was used was for all languages a BERT base model with 110M trainable parameters pretrained on text from its respective language. For this task we used the

Train F1 Validation F1![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.008.png)

English 0.642 0.497 Finnish 0.865 0.711 Japanese 0.668 0.562

Table 8: Training results for the sequence tagger using BERT

same training parameters as for the BiLSTM-CRF for easy comparison in addition to 200 warm up steps and a weight decay of 0.01. This model was also trained for 10 epochs with a learning rate of 2 ·105. The training results can be seen in table[ 8](#_page6_x70.87_y64.09)

This model performs significantly better but still not perfect. We therefore chose to do a qualitative inspection of some of the predicted answer spans to see if they makes sense and where it goes wrong. A few examples of predicted answers versus the correct one for different question can be seen in Appendix [7.3.](#_page12_x70.87_y700.97) What we generally notice is that it is a bit biased towards ’O’ which makes somewhat sense due to the huge presence in the data. This could maybe be mitigated by oversampling questions with an actual answer. We also notice it generally has a hard time with answers only consisting of a date or a few words, which maybe can be explained by the earlier observation that the presence of the word both in the answer and in the question makes it more likely to be answerable. Moreover we notice that there might be some ambiguity in the data. For example one question ask what martial arts Marines learn. Here the correct answer is the Marine Corps Martial Arts Program. However our sequence labeller tagged the following answer "combine existing and new hand-to-hand and close quarters combat techniques with morale and team-building functions and instruction in the Warrior Ethos", which also makes sense as an answer to the question.

In general we found that it is indeed possible also to model Q&A as a token-classificationtask. This may even fix the problem when the context con- tains multiple correct answer as we saw with the ambiguity of one of the questions. Our model was however long from perfect. It should however be noted that the model isn’t the largest or finetuned on much data. Another problem is that we haven’t done any finetuning at all which possibly further could improve the model.

Fine-tune/Eval English Finnish Japanese English 0.675 0.645 0.63

Finnish 0.64 0.705 0.585 Japanese![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.012.png) 0.42 0.555 0.655

Table 9: Zero-shot cross-lingual accuracy for the multi- lingual BQC

6 Multilingual QA

1. Answerable TyDiQA BQC

In order to extend the Answerable TyDiQA bi- nary question from section [3,](#_page1_x306.14_y652.72) we change the lan- guage model from a language-specificpretrained GPT2 to BLOOM ([Margaret Mitchell and co., 2022](#_page7_x301.16_y580.07)). BLOOM is an autoregressive language model, trained to continue text from a prompt on more than 46 languages, including English, Finnish and Japanese. We trained the smallest version of BLOOM (560M parameters) with 200 warm up steps, a weight decay of 0.01 and the Adam opti- mizer with a learning rate of0.00005. Due to hard- ware constraints (specificallyRAM limitations) we were only able to run the model with a batch-size of 1, which in turn made time-constraints an issue. The results below are therefore based on training and validation of only 300 balanced samples each. It is to be stated clearly, that is of course not opti- mal. One would normally train for multiple epochs, and until no significant improvements in loss are to be found (or no improvement in performance on the validation set).

From our earlier BQC models, we have seen that XGBoost is very performant on the classification tasks. But since XGboost is quite memory consum- ing, we resorted to using Logistic regression as the predictor for this task.

From table [9 ](#_page6_x306.14_y64.09)we see that the zero-shot matrix non-theless looks very much like how you would expect. We have highlighted the best performing configuration for each row, and the diagonal is clearly to be preferred.

It seems that Japanese is the training language which results in the least amount of generalization. After fine-tuning,it actually performed worse than simply guessing randomly on the English data-set. And again, we observe Finnish to be by far the easiest language for determining answerability. This time the phenomenon certainly cannot be explained by a larger training set. It may be that the morphology of Finnish is somehow easier to grasp for the models we investigate. We also notice


Fine-tune/Eval English Finnish Japanese with the natural language toolkit. " O’Reilly Media,![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.013.png)

English 0.447 0.437 0.426 Inc.".

Finnish 0.515 0.677 0.443 Stanley F Chen, Douglas Beeferman, and Roni Rosen- Japanese 0.470 0.503 0.572 feld. 2008. [Evaluation Metrics For Language Mod- els.](https://doi.org/10.1184/R1/6605324.v1)

Table 10: Zero-shot cross-lingual evaluation for the

sequence labeller from section[ 5](#_page4_x306.14_y257.54) Tianqi Chen and Carlos Guestrin. 2016. [XGBoost: A](https://doi.org/10.1145/2939672.2939785)

[scalable tree boosting system.](https://doi.org/10.1145/2939672.2939785) In Proceedings of the

22nd ACM SIGKDD International Conference on that fine-tuning on English achieved the best on Knowledge Discovery and Data Mining, KDD ’16, average across the languages. However, the results pages 785–794, New York, NY, USA. ACM.

should be taken with a grain of salt. Given the low Jacob Devlin, Ming-Wei Chang, Kenton Lee, and count of training and validation samples, all values Kristina Toutanova. 2019. [BERT: Pre-training of ](https://doi.org/10.18653/v1/N19-1423)are associated with a high degree of uncertainty. [deep bidirectional transformers for language under-](https://doi.org/10.18653/v1/N19-1423)

[standing.](https://doi.org/10.18653/v1/N19-1423) In Proceedings of the 2019 Conference of To conclude, there are many ways in which perfor- the North American Chapter of the Association for

mance on this task could be improved. Training on Computational Linguistics: Human Language Tech- more data, larger language model (BLOOM can nologies, Volume 1 (Long and Short Papers), pages get as large as 7B parameters), more sophisticated 4171–4186, Minneapolis, Minnesota. Association for finalclassifier(a larger neural network, XGBoost Computational Linguistics.

etc.), more deliberate choice of hyperparameters Angela Fan, Mike Lewis, and Yann Dauphin. 2018. and so on. It is well known that deep learning [Hierarchical neural story generation.](https://doi.org/10.48550/ARXIV.1805.04833)

models needs a great deal of data to achieve good Krzysztof Fiok, Waldemar Karwowski, Edgar Gutierrez- performance, but the fact that one already can Franco, Mohammad Reza Davahli, Maciej Wilam- get around 65% accuracy on only 300 samples is owski, Tareq Ahram, Awad Al-Juaid, and Jozef Zu-

truly a testament to the power of pretrained models. rada. 2021.[ Text guide: Improving the quality of long](https://doi.org/10.1109/access.2021.3099758)

[text classificationby a text selection method based on feature importance](https://doi.org/10.1109/access.2021.3099758). IEEE Access, 9:105439–105450.


2. IOB tagging system

For the answer extraction we will use the model utilising a pre-trained transformer architecture since this model was superior in the previous analy- sis. We will use the same architecture and the sameconfigurationas before. However for this task we will use mBERT as our pretrained model. The re- sults of the zero-shot cross-lingual evaluation can be seen in table[ 10.](#_page7_x70.87_y64.09)

We have again highlighted the best configuration for each row. We notice it performs the best on the language it was pretrained on which is in accor- dance with our expectations. We also notice that the performance is somewhat the same inpendent on what language it was pretrained on. The per- formance is however a bit lower than when using a language specific pretrained model which is to be expected. The performance when finetuningon Japanese is however slightly better. We hypothe- sise that this is because the Japanese texts often also contains a bit of English

References

Steven Bird, Ewan Klein, and Edward Loper. 2009.Nat-

ural language processing with Python: analyzing text

Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Ar-

mand Joulin, and Tomas Mikolov. 2018. Learning word vectors for 157 languages. In Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018).

Tatsuki Kuribayashi, Yohei Oseki, Takumi Ito, Ryo

Yoshida, Masayuki Asahara, and Kentaro Inui. 2021. [Lower perplexity is not always human-like.](https://doi.org/10.18653/v1/2021.acl-long.405) In Pro- ceedings of the 59th Annual Meeting of the Associa- tion for Computational Linguistics and the 11th Inter- national Joint Conference on Natural Language Pro- cessing (Volume 1: Long Papers), pages 5203–5217, Online. Association for Computational Linguistics.

Yacine Jernite Margaret Mitchell, Giada Pistilli and co.

\2022. [Bigscience large open-science open-access multilingual language model.](https://huggingface.co/bigscience/bloom)

Paul McCann. 2020. [fugashi, a tool for tokenizing](https://www.aclweb.org/anthology/2020.nlposs-1.7)

[Japanese in python.](https://www.aclweb.org/anthology/2020.nlposs-1.7) In Proceedings of Second Work- shop for NLP Open Source Software (NLP-OSS), pages 44–51, Online. Association for Computational Linguistics.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey

Dean. 2013a. Efficientestimation of word represen- tations in vector space. CoRR, abs/1301.3781.

Tomas Mikolov, Ilya Sutskever, Gregory S. Corrado

Kai Chen, , and Jeffrey Dean. 2013b. Distributed representations of words and phrases and their com- positionality. In Advances in Neural Information

Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems.

Raghavendra Pappagari, Piotr Zelask˙ o, Jesús Villalba,

Yishay Carmiel, and Najim Dehak. 2019.[ Hierarchi- cal transformers for long document classification.](https://doi.org/10.48550/ARXIV.1910.10781)

Alec Radford, Jeffrey Wu, Rewon Child, David Luan,

Dario Amodei, and Ilya Sutskever. 2018. [Language models are unsupervised multitask learners.](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.

Weinberger, and Yoav Artzi. 2019.[ Bertscore: Evalu- ating text generation with bert.](https://doi.org/10.48550/ARXIV.1904.09675)

7 Appendix

1. Sampled Sentences
1. English

0: Question: When was the

International Hydrographic Organization formed?

Context: From 1869 to 1904 the Organization was composed of 17

international teams, consisting of representatives from 13 countries

- These included: France, Russia, Brazil, the former Yugoslavia ( formerly

1: Question: When did the War of the

Spanish Succession begin?

Context: War of the Spanish Succession (Spanish: Caligula de las Indias)

was a Spanish civil war between the Spanish and French forces that

ended with the assassination of 2: Question: When was the original

Tomb Raider published?

Context: Tomb Raider is a third-person action role-playing game

developed by Eidos Interactive and

published by Square Enix Studios, Inc. for Microsoft Windows, PlayStation 3, PlayStation 4, Xbox

3: Question: When was the Last House

on the Left released?

Context: Last House on the Left (also

known as the Last House on the Right and the Last House on the Left) is an American social movement founded in 1971 by the Socialist Party

4: Question: When are the NFL draft

eligible?

Context: The 2017 NFL Draft began on April 29, 2017, and ended on March

22, 2018, with all 18 teams choosing a winner. Once the picks were determined, all teams then chose an

0: Question: What is the largest

currency?

Context: The English minting system, also known as the Royal Mint,[ lower-alpha 1] followed the         tradition of creating the most

precious single-dollar note, known as the "Queen’s Crown".[

1: Question: What language do they

speak in Denmark?

Context: Danish is known for its distinctive nasal, choral, and laryngeal vocalizations and for

its distinctive pronunciation. It is also known for its unique

architecture and distinctive style of architecture

2: Question: What did George Orwell

write to the Observer public school system?

Context: On 3 November 1984, the

Observer’s public school system closed, as a decision to do so has

been described as "unconscionable ". The Observer has

3: Question: What is the name of a new

kind of animal?

Context: Species of animals have been described since the earliest known

listings. As animal names are based on their behavior, and

animals that move based on their environment, the name which is

4: Question: What is the most abundant

mineral in Hawaii?

Context: Hawaii has an area of 4,700.7 million acres (24,000km2), of

which 800,000,000 acres (28,000km2 ) is agricultural

0: Question: How loud is lightning? Context: The current has a wavelength

of about 380KHz (542 metres/s). Although the luminous intensity was thought to fluctuate over the years, the actual luminous luminous intensity was computed

1: Question: How many soldiers are in

the U.S. Army?

Context: The enlistment of enlisted men and women by the United States

armed forces is authorized by law at all levels of command.[1] The enlistment of an enlisted man

2: Question: How long was the battle

of Ypres in France?

Context: After its defeat, the Austrians formed a coalition led

by the Prussians against them. The Prussians won. When Austrian France fell on 19 August 1914

3: Question: How old is Luke Cage? Context: After the death of Cage’s

half-brother Luke Cage (Sean Connery), Luke undergoes a series of transformations to become his true self.[2] When he appears at the end of the

4: Question: How many years did it

take for Glee to air?

Context: The series premiered on CBS

on December 19, 1965. It starred Leslie Knope and Leonard Nimoy as characters assigned to the Los Angeles Police Department (LPD)

2. Finnish

0: Question: Milloin ensimm inen

Avaruusasema laukaisi T htitiede -lehden?

Context:

Vuoden 1953 avaruuslentosuunnitelmassa esiteltiin ensimm inen

avaruusasema. Avaruusasema nimettiin sen j lkeen uudeksi avaruusaseman nimell

1: Question: Milloin ensimm inen

elokuva tehtiin?

Context:

Vuonna 1989 brittil inen tuottaja

Denny Hooker palkittiin Women of

English Finish Japanese Rank![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.014.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.015.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.016.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.017.png) First Last First Last First Last![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.018.png)

1  When ? Milloin ? 日本 ?
1  What zombie Mikä tulitaistelussa 『 いつ
1  How metabolite Missä tohtoriksi+ アメリカ た
1  Who \\ Kuka syntynyt 世 界 どこ
1  Where BCE Mitä pinta-ala 第 何![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.019.png)

Table 11: Top 5 most common firstand last tokens for each language![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.020.png)

LR XGBoost MLP RandomForest Language![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.021.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.022.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.023.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.024.png) T V T V T V T V



|English 99.8% 76.8% 94.9% 79.1% 88.8% 77.5% 82.1% Finnish 99.6% 80.2% 89.3% 82.2% 93.8% 79.2% 78.5% Japanese 99.8% 73.2% 95.3% 79.6% 93.9% 73.4% 78.5%|75.5% 74.8% 70.8%|
| -: | - |
|Table 12: BOW baseline accuracies on training (T) and validation (V) data.||
LR XGBoost MLP RandomForest Language![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.025.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.026.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.027.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.028.png) T V T V T V T V



|English 99.8% 77.1% 99.8% 78.7% 92.9% 76.7% 81.6% Finnish 99.3% 80.5% 96.7% 81.6% 96.4% 77.8% 74.2% Japanese 99.7% 73.4% 99.7% 78.0% 94.6% 74.3% 78.9%|74.2% 64.7% 70.2%|
| -: | - |
|Table 13: Accuracy for BQC extended with continuous representation.||
LR XGBoost MLP RandomForest Language![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.029.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.030.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.031.png)![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.032.png) T V T V T V T V![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.033.png)

English 71.2% 71.0% 100% 72.6% 73.3% 72.2% 91.8% 71.9% Finnish 69.3% 70.1% 98.6% 72.4% 72.4% 72.2% 87.6% 72.7% Japanese 68.0% 65.8% 100% 70.1% 72.8 % 68.7% 89.9% 69.2%![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.034.png)

Table 14: Accuracy for models using only continuous vector representation

the Universe -palkinnolla.[2]

Lis ksi kaksi muuta tunnustusta

annettiin muun muassa George Luc 2: Question: Milloin Charles de

Gaullen syntyi?

Context: Charles de Gaullen syntyi Ranskassa 1872. H n joutui vuonna

1872 naimisiin saksalaisen, ranskalaisen ja italialaisen kreivitt ren kanssa. Avioliitosta tuli

3: Question: Milloin Nintendo on

perustettu?

Context: Nintendo ei ole e n

itsen inen yhti , vaan se on itsen inen brittil inen tietokone- ja Internet- palveluntarjoaja. Sen perustivat

William Friedman ja Arthur B. B. Goth

4: Question: Milloin Ludvig XIII

syntyi?

Context:

Valentian tasavallan hallitus aloitti valtakuntansa hajottamisen vuonna

\1886. Vallankumous alkoi jo vuonna 1896. Vallankumous johti vallankumoukseen ja se

0: Question: Mik oli Leonid

Bre nijevin puoliso?

Context: Ignatius, jota pidettiin

Ven j n aatelisena, nimitti usein [nijiveiksi. [1

1: Question: Mik on Suomen pisin

joki?

Context: Suomen pisin joki on Laatokan

koillisosa. Se on 5,6 kilometri pitk ja 12,7 kilometri leve

- Joki laskee P ij nteelle , josta se kulkee Ha

2: Question: Mik oli Suomen yleisin

kalalaji?

Context: Suomen yleisin kalalaji,

ahven, ovat pieni s r k i ja muita j r v i . J rvell

esiintyy ainakin kymmenen lajia, joista seitsem n lajia on uhanalaisia.[

3: Question: Mik on Sknen korkein

vuori?

Context: Sk nessa on kahdeksan

tasavaltaan kuuluvaa vuorijonoa .[1] Se on pisin yht jaksoisesti kohoava joki. Vuorenhuippu

4: Question: Mik oli Tsernobylin

ydinvoimalaonnettomuus?

Context:

Tsernobylin ydinvoimala suljettiin

vuonna 2014.[1] Tsernobyl on radioaktiivisen hiilen onnettomuus .[2]

0: Question: Miss sijaitsee Suomen

vanhin j halli?

Context: Vuoden 1949

talviolympialaisten mitalitoiveissa oli kaksi hopeista

ja kaksi pronssista palkintoa. Seuraavana vuonna olympialaisissa pelattiin pronssiottelu, jossa pronssimitalistijoukkueiden

1: Question: Miss sijaitsee

Runeberginkadun raatihuone? Context: Raatihuone (lyhenne Valse

d e des grands) on kaupunki Mannerheimintiell . Se on my s Suomen vanhin kaupunginmuseo. Kulttuurikeskuksen

2: Question: Miss sijaitsee Kiinan

muuri?

Context: Luokka:Suuri V e n j Luokka:Maailman sota

Luokka:Kansainv lisen politiikan

tapahtumat

Luokka:Seulonnan keskeiset artikkelit Luokka:Kiinalaiset

Luokka:Kiinan

3: Question: Miss maakunnassa

sijaitsee Suomen suurin uimahalli? Context: Suomessa on kaksi uimahallia

sek yli 6500 ulkopaikkaa:

Telakkavesi (660 ha) ja V ksyn

uimahalli (202 ha

4: Question: Miss Suomen

kuntapolitiikka kehittyi?

Context: Suomen 1900-luvulla alkanut kuntapolitiikka oli voimakasta ja

voimakasta. T ll in alkoi nousta esiin merkitt vi kysymyksi kuntien talouden luonteesta ja kehityksest . Merkitt vimm t uudistukset tapahtuivat 1900-luvun alussa,

3. Japanese

Note: The original Japanese generated text does have newlines, but the latex package rendering the characters does not recognize this.

0: question: 日 本 における 、 自 動 車 の 最 大 排 気 量 は 何 ? context: 1. が 日 本 の 自 動 車 メ ー カ ー に 、 2が 軽 自 動 車 の 各 社 に 適 用 される 。 そし て 現在 では 、 日 本 自 動 車 協 会 は 排 気 量 と 車 名 ・ 車種 で 1: question: 日 本 自 衛 隊 はいつ 誕 生 した context: 戦 後 自 衛 隊創設 以 降 、 国 主 導 の 下 で 防 衛 組 織 の 整 備 と 防 衛 機 能 の 充実 が 図 られており 、 自 衛 隊組 織 が 「 自 衛 隊 」 として 独 立 する 事 態 にはなっていない [38] 2: ques- tion: 日 本 国 鉄 はいつ 創 立 した ? context: 国 鉄 は 、 日 本 最 大 の 私 鉄 で 、 日 本 の 鉄 道 事 業 者 で は 第 2位 の 規 模 を 持 つ 日 本唯 一 の 鉄 道 事 業 者 である 。 1895年 (明 治 28年 )に 、 東 京 ・ 日 本 橋 3: question: 日 本 の 一 人 当 たり gdpは ? context: gdp (english) は 、 アメリカ 合 衆 国 の 数 値 。 世 界 の 国 民 総 生 産 の 内 訳 を 取 りまとめた 世 界 経 済 予 測 を 基 に 算出 され 、 各 国 の gdpを 比 較 した 際 、 4: question: 日 本 サ ッ カ ー はいつ 始 まった ? context: サ ッ カ ー 競 技 は 、 日 本 の 国 民 競 技 で 歴 史 ある 競 技 の 一 つ [1][2][3][4]。 日 本 文 化 の 発 祥 地 ともいわれる 。 0: question:

- 銀 河英 雄 伝 説 新 風 」 の 舞 台 となった 場所 はどこ context: 銀 河英 雄 伝 説 の 中 心人 物 と して 活 躍 した 「 英 雄 」 と 「 伝 説 の 英 雄 」 と の 因縁 に 思 いをいだき 、 仲 間 として 彼 らと 共 に 銀 河 1: question: 『 赤 マルジャンプ 』 の 視 聴 率 は ? context: 2011年 9月 5日 〜 9月 18日 に 、 nhk総 合 テレビ で 再 放 送 された 。 2: ques- tion: 『 バ ッ トマン 』 はいつ 書 かれた ? con- text: 1993年 12月 2日 の 『 映 画 『 バ ッ トマン 』 に 主 演 することになった レイ ・ チャ ー ルズ は 、 後 に 『 バ ッ トマン : バタイヤ ー』 で 主 演 することを 約 束 する 3: question: 『 キャ プテンハ ー ロ ッ ク 』 の 監 督 はだれ context:
- キャプテンハ ー ロ ッ ク 』 の 脚 本 を 担 当 し たのは 映 画監 督 の スティ ー ヴ ・ クロ ッ パ ー である 。 脚 本 について クロ ッ パ ー は 「 この 映 画 の 制 作 に 重 要 な 影 響 を 与 えた 」 と 4: ques- tion: 『 ポケ ッ トモンスタ ー サン ・ ム ー ン 』 の スト ー リ ー は ? context: 『 ポケ ッ トモン スタ ー ウルトラサン 』・『 ポケ ッ トモンス タ ー ブラ ッ ク ・ ホワイト 』 に 登 場 した ポケ モン の ボス と 闘 います 。 0: question: アメリ カ 連 邦 準 備 理 事 会 (frb)はいつ 結 成 した context: 2007年 11月 11日 の 声 明 では 、 連 邦 準 備 理 事 会 法 に 基 づいて 、 連 邦 公 開 市 場 委員 会 (fomc)の 議 長 を 務 める ジョ ー ジ ・ c 1: question: アメ

リカ 南 北 戦争 での 南 軍 の 勝 利 数 は ? context: 1861年 1月 2日 に 、 北 軍 の ジェイムズ ・ ヘン リ ー が 南 軍 側 に 加 わっていた レキシントン の 戦 いで 、 フランクリン ・ d.c・ マクヘンリ ー が 南 軍 の ジェイムズ 2: question: アメリカ の 国 旗 は 何 色 ですか ? context: アメリカ 合 衆 国 の 国 旗 (アメリカ がっしゅうこくのこっき )と は アメリカ 合 衆 国 の 実 色 である アメリカ 合 衆 国 黄 色 を 使 用 する 連 邦 の 国 旗 であり 、 ア メリカ 合 衆 国 の シンボル である 。 3: question: アメリカテネシ ー 州 メンフィス はどこにあ

る ? context: アメリカ 最 大 の インディアン 居 留 地 メンフィス は 現在 8の 部 族 が 住 む 、 3つの インディアン 居 留 地 が 連 結 しており 、 この 地 区 は メンフィス 州 の 州 都 メンフィス 県 に 属 している [6]。 メンフィス は 4: question: ア メリカワシントン 州出 身 の バスケ ッ トボ ー ル 選 手 は 誰 ? context: bjリ ー グ の 秋 田 ノ ー ザ ンハピネ ッ ツ に 4年 目 にして 入 団 し 、 4年 間 プ レ ー した 。 秋 田 が hcとなった 2011-12シ ー ズ ン は 7勝 1敗 、 同 シ ー ズン 終 了 後 には

2. Adverserial questions 7.2.1

Q: When was Queen Elizabeth II born?

A: Queen Elizabeth II was the queen of England from 1952 until she died on the 8th of September 2022.

Type: Not answerable.

[Source.](https://en.wikipedia.org/wiki/Elizabeth_II)

7.2.2

Q: What is an uncastrated male horse called in American English?

A: A horse of masculine gender which has not been castrated can be referred to as a stallion in American English

Type: Answerable.

[Source.](https://www.merriam-webster.com/dictionary/stallion)

7.2.3

Q: How can fast does the earth spin around its own axis?

A: Earth earth earth earth spin spin spin spin spin around around around around

Type: Not answerable.

3. Examples of sequence labeller

Question: When was github created? Answer: February 2008

Predicted answers: []

Question: Who was the first leader of West Germany?

Answer: Theodor Heuss

Predicted answers: [’Theodor Heuss’]

Question: What martial arts do Marines learn? Answer: Marine Corps Martial Arts Program Predicted answers: [’combine existing and new hand-to-hand and close quarters combat techniques with morale and team-building functions and instruction in the Warrior Ethos’, ’[’, ’which began in 2001 , trains Marines’, ’of’]

Question: What is the world’s largest horse show?

Answer: Devon Horse Show

Predicted answers: [’Since 1896 , the Devon Horse’, ’is’]

Question: When did the first episode of Big Brother Australia air?

Answer:

Predicted answers: []

Question: When was Nike founded? Answer: January 25, 1964

Predicted answers: []

Question: When did Final Fantasy Type-0 come out?

Answer:

Predicted answers: []

Question: What is the oldest city in Myanmar? Answer: Beikthano

Predicted answers: []

Question: What is the strongest recorded wind? Answer: was during the passage of Tropical Cyclone Olivia on 10 April 1996: an automatic weather station on Barrow Island, Australia, registered a maximum wind gust of 408km/h Predicted answers: []

Question: Who was president in 1817? Answer:

Predicted answers: []

4. Similarity Measures

Euclidean distance

∥qavg − cavg∥

Cosine similarity

qavg ·cavg ∥qavg∥∥cavg∥ BERTScore

1

RBERT = max qi⊤cj

|q| q ∈q cj

∈c

i

PBERT = 1 max q⊤cj i

FBERT = 2|cP| BERcj ∈cT ·RBERT

qi∈q           PBERT + RBERT

Where qavg and cavg are the average question and context representations, andqi and cj represent the individualwordembeddingsthatthedocumentsare composed of. In BERTScore it is also presumed that qi, cj have been normalized to unit length. We hypothesized that these similarity measures could increase performance in a similar manor as to the simple overlap feature from section [1.2.](#_page0_x306.14_y251.15) The first measure, euclidean distance, does provide some indication of whether the two embeddings are close in the representational space. But it has the drawback, that occurrence counts impact vector magnitudes, meaning that semantic similarity is better captured in the angle between embeddings. This is exactly what the second measure, cosine similarity, achieves, and why it is the default indicator of like-ness for word embeddings. The last similarity measure is BERTScore [(Zhang et al.](#_page8_x65.89_y185.44), [2019](#_page8_x65.89_y185.44)). BERTScore is traditionally used as an automatic evaluation metric for text generation. BERTScore works by computing a similarity score for each token in the candidate sentence with each token in the reference sentence. However, instead of exact matches, it relies on token similarity using contextual embeddings.

Even though we are not doing text generation, we can nonetheless utilize this same method with word2vec embeddings rather than BERT to obtain a more nuanced similarity metric.

5. Feature importance plots

![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.035.png)

Table 15: Logistic regression highest weights

![](Aspose.Words.0befaa39-83f4-455a-ad5f-0980de7d956e.036.png)

Table 16: Random forest feature importance
