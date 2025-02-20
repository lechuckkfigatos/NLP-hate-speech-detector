# NLP-hate-speech-detector by chuck
  

Step 1: dataloader.py :
        clean the data from "labeled_data.csv" and "labeled_data_2.csv", 
        returns 2 "cleaned_hate_speech_data.csv"s that only contain label and clean text.

Step 2: split.py :
        Split the 2 cleaned dataset into "train_data.csv"/"test_data.csv" with a 80/20 %

Step 3: vocabulary.py : 
        def "build_vocab" is used to build a vocabulary:
        Args:
        texts (list): list of tokenized sentences.
        Returns:
        vocab (dict): map from word to index.

Step 3: naivebayes_training.py : 
        Train the Multinomial Naive Bayes model:
        use the train_data.csv to calculate the logprior, loglikelihood, vocab with alpha= 1
        then save into "naive_bayes_model.pkl"

Step 4: main.py :
        Load the trained model "naive_bayes_model.pkl"
        used to evaluate the "test_data.csv" using the "predict" function from "prediction_function.py"

        