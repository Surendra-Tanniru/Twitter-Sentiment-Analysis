# Twitter-Sentiment-Analysis

Twitter Sentiment Analysis using Lexicon based and NLP based algorithms

In this git, we apply sentiment analysis and machine learning principles to identify the “public sentiment”. We use twitter data to predict sentiments. We then used Lexicon based approach and NLP based approach to get the predictions and to test our results, we used cross validation and accuracy of the test data.

Twitter is one of the platforms widely used by people to express their opinions and showcase sentiments on various occasions. Sentiment analysis is an approach to analyse data and retrieve sentiment that it embodies.
The tweet format is very small, which generates a whole new dimension of problems like the use of slang, abbreviations, etc. 

# Why Use Twitter?

There are many reasons why Twitter is used as a source for information associated with a disturbance including:

### Data from mixed sources:
Anyone can use Twitter and thus the sources of information can include media, individuals, official and others. A mixed source of information provides more well-rounded perspective of the impacts of the particular event and the actions being taken to deal with that event.

### Embedded content:
Twitter allows users to embed pictures, videos and more to capture various elements of a disturbance both visually and quantitatively.

### Instantaneous coverage:
Twitter allows users to communicate directly in real time. Thus, reports on what is going on during an event can happen as the incident unfolds.

Here we explain on the exploration,  pre-processing of data, transforming data into a proper input format and classify the user’s perspective via tweets into positive, negative and neutral  by building supervised learning models like Lexicon based and NLP based approach using Python.

The sentiment analysis is done on Covid Vaccinations and we tried to identify public sentiments on different types of vaccines available 

# DATASET


In this project, we used publicly available Twitter data.

We use Twitter API Tweepy to fetch the data using Hashtags keywords.We used below Keywords to get the data from twitter and then combined the tweets into a single dataset.
#covid+vaccination
#pfizer+vaccine
#Astrazeneca+vaccine
#Covaxin+vaccine
#sputnik+vaccine 

The data includes various components that you can use to extract information:

User Name: This is how each unique user is identified.

Time Stamp: When the tweet was sent.

Tweet Text: The body of the tweet - needs to be 140 characters of less!

Hashtags: Always proceeded by a # symbol. A hashtag is often describes a particular event or can be related to a particular topic.

Links: Links can be embedded within a tweet. Links are a way that users share information.

Embedded Media: tweets can contain pictures and videos. The most popular tweets often contain pictures.

Replies: When someone posts a tweet, another user can reply directly to that user - similar to a text message except the message is visible to the public.

Retweets: A retweet is when someone shares a tweet with their followers.

Favorites: You can “like” a tweet to keep a history of content that you like in your account.

Latitude/Longitude: about 1% of all tweets contain coordinate information.

Since we need only the Tweet text from entire data we extracted text from the entire tweet data
 
 
# Data Pre-processing

The data obtained from the above mentioned sources had to be pre-processed to make it suitable for reliable analysis. We pre-processed the data in the following manner-

1.	Removed the URLs and punctuations, symbols, emoticons etc using a regular expression to get the clean data
2.	Then move the data to a file and labelled the sentiments manually.

#	SENTIMENT ANALYSIS

  Sentiment analysis was an important part of our solution since the output of this module was used for learning our predictive model. While there has been a lot of research going on in classifying a piece of text as either positive or negative, there has been little work on multi-class classification. In this project, we use three classes, namely, Positive, Negative and Neutral. There are several approaches in identifying the sentiments but we decided to develop our own analysis code. The methodology we adopted in finding the public sentiment is as follows-

1. Tokenization

  Once the data is cleaned and manually labelled, we read the dataset and create tokens (words) by splitting the words from the sentence 

2. Normalization
  
  Normalization is helpful in reducing the number of unique tokens present in the text, removing the variations in a text and also cleaning the text by removing redundant information. Here to make the processing easy and we need almost everything from the tweet, we converted the data into lower words

3. Word Frequency Generation
  
  We get the number of times a word repeated in the total tokens and get the value of it. Later we added them into a dictionary with word as key and occurrence of the word as value.

4. Removal of Stop Words
  
  Stop words are a set of commonly used words in any language. For example, in English, “the”, “is” and “and”, would easily qualify as stop words. In NLP and text mining applications, stop words are used to eliminate unimportant words, allowing applications to focus on the important words instead.
We took NLTK and SPACY stop words library and also created our own stop words list and then we appended our list of stop words to NLTK library
We then removed all the stop words from the word dictionary we created. 

5. Lexicon dictionary Creation
  
  Lexicon is nothing but (a list of) all the words used in a particular language or subject, or a dictionary.
We used AFINN dictionary dataset to identify the Lexicons. It contains the scoring to each word/lexicon.

6. Sentiment Score Computation
	
  We created a function to identify the sentiment score of each word from the tweet and check if that is available in lexicon dictionary.
Based on the value available in Lexicon dictionary we calculate the word as positive, Negative and Neutral and predict the sentiments.
We use the tweet and normalize and tokenize the sentence and check the word available in lexicon dictionary and calculate the score.

Sentiment Score = Sum of lexicon values for each word in the sentence that are available in lexicon dictionary

#	Modelling and Prediction

  We have used Lexicon based analysis and NLP based analysis.

  Lexicon based analysis used the process of tokenizing, Normalizing, creating word frequency dictionary and identifying the sentiment score based on AFINN or VADER based lexicons(We used AFINN in our paper).
NLP based modelling used CountVectorizer and TF-IDF Vectorizer algorithms.

  Count Vectorizer is a way to convert a given set of strings into a frequency representation. Count Vectorizer will skip the stop words and takes the rest of the words and convert it into a vector/frequency representation. If the word contains in the text, the word will have a value of 1 else the value is 0 and so on. This way it will create a sparse matrix with values 0 and 1 based on the occurrence of the word in the sentence.

  TF-IDF means Term Frequency-Inverse Document Frequency. This is based on the frequency of a word in the corpus but it also provides a numerical representation of how important a word is for statistical analysis. Here instead of 1 and 0, the values lie between o and 1 and will calculate based on the number of times it occurs in the sentence. The lower occurrence will have higher value closer to 1 and the most occurred word will have less value closer to 0.

  With both Algorithms, we have used models like Multinomial Naïve Bayes, Support Vector Classifier, Random forest Classifier, Multilayer Perceptron classifier and calculated the test Accuracy and also True positive rate using the confusion matrix. Along with the said algorithms and models, we have also tried using the transformer and pipeline models using GridSearch CV with Naïve Bayes and Random forest classifiers.

#	Conclusions and Future Work	

  After testing and analyzing with different algorithms and models we came to a conclusion that From above, we saw that SVC with TF-IDF Vectorizer gives TP rate of 76% with an Accuracy of 51% followed by Random Forest Classifier using Pipeline model with 52% Accuracy and 74% TP rate.

Finally, it’s worth mentioning that our analysis only considered twitter tweets and only from english speaking people and doesn’t take into account many factors. So, this may or may not map the real public sentiment. It’s possible to obtain a higher correlation if the actual mood is studied. But in that case, there’s no direct correlation between the people who took vaccine and who use twitter more frequently, though there certainly is an indirect correlation – people getting vaccinated or choosing a particular vaccine brand aﬀected by the moods of people around them, i.e., the general public sentiment. 

Also, in this project we considered three sentiments like Positive, negative and Neutral. We can also do the same analysis with Positive and Non-positive sentiments making Neutral and Negative sentiments as Non-positive and do analysis. All these remain as areas of future research.
