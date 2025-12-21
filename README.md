[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/k80emrmd)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=21813697)
## HW 6 Mining F25

**Class code repository: https://github.com/HofstraDoboli/TextMining**

**Starting code** 

`encoder_classification_example.ipynb`

`decoder_class.ipynb`

`bert_sentiment_class.ipynb`



**Classification problem**: https://www.kaggle.com/competitions/predict-movie-genres-from-plot-summaries/data 

The dataset already has a train and test sets. You need a validation set - 10% of the train set for BERT training (Step 2)


1. Implement the Naive Bayes classifier from scratch - not using the sklearn library. You need to do the text processing - be careful which terms/NERs you want to keep or discard. Then you need to compute the class or prior probabilities, and all conditional probabilities for the vocabulary words. Show the classification results for each class (accuracy, precision, recall, F1, and overall macro and micro averaged scores for the training and testing datasets
2. Train a BERT model to predict the movie genres. Show the same results for train, test, and validation datasets. 
3. Use a decoder model to predict the movie genres.  Show the same results for train, test, and validation datasets.

 ### Turn in ###

1. The jupyter notebook(s). Please organize and comment your code so that it is clear where the code and answer for each question are.
2. Add to Answers.md the description of what you did, how you chose the training/validation and testing data sets, and what text processing steps you took for the Naive bayes, etc. Show the results and compare and discuss the performance of the three methods. Please be thorough with your writing and comment your results. 
3. Write in Acknowledgements.md file any help you got, and if you used LLMs, all prompts and answers you used for your coding.
 



 
