import pandas as pd
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import pickle

# of the sentence.
def vader_scores(sentence):
    """
    Args: 
    sentence (str): sentence we will weight 
    """

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
     
    #only returns the compound value for ease of use
    compound_value =  sentiment_dict['compound']

    return compound_value

def tb_scores(sentence):
    """
    Args: 
    sentence (str): sentence we will weight 
    """
    #By default uses the Pattern Library
    sentencetb = TextBlob(sentence)
    compound_value =  sentencetb.sentiment.polarity

    return compound_value

def siebert_scores(sentence):
    """
    Args: 
    sentence (str): sentence we will weight 
    """
    if(len(sentence)>500):
        return 1
    sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english", device= "cuda")
    rating=sentiment_analysis(sentence)
    if (rating[0]['label']=="POSITIVE"):
        return 1
    else:
        return -1

def svm_scores(sentence):
    text = {'text': [sentence]}
    text = pd.DataFrame(text)
    loaded_sv = pickle.load(open(r'C:\Users\Seeratul\Documents\GitHub\BachelorThesis\code\fSVM_model.sav', 'rb'))
    return vectroremapper(loaded_sv.predict(preprocesser(text)))

def nb_scores(sentence):
    text= {'text': [sentence]}
    text = pd.DataFrame(text)
    loaded_nb = pickle.load(open(r'C:\Users\Seeratul\Documents\GitHub\BachelorThesis\code\fNaive_model.sav', 'rb'))
    return vectroremapper(loaded_nb.predict(preprocesser(text)))      

def sentimentmapper(df, scoring= "Vader"):
    """
    Args: 
    sentence (df): pandas dataframe to be analyzed
    scoring: Vader,Siebert,TextBlob,FelixsSVM,FelixsNB
    """ 
    # Mapps the sentiments expects a dataframe containing indexes, titles, year, and excerpt adds a score
    
    df = df.reset_index()
    #creates new indexes incase indexes are not 1-n
    if (scoring == "Vader"):
        dfs = df['title'].apply(vader_scores) 
    elif (scoring == "Siebert"):
        dfs = df['title'].apply(siebert_scores)
    elif (scoring == "TextBlob"):
        dfs = df['title'].apply(tb_scores) 
    elif (scoring == "FelixsSVM"):
        dfs = df['title'].apply(svm_scores)  
    elif (scoring == "FelixsNB"):
        dfs = df['title'].apply(nb_scores)   
    else:
        raise ValueError('You specified a non legal scoring method. (Vader,Siebert,TextBlob,FelixsSVM,FelixsNB)')
    #creates new dataframe only conatining sentiment scores
    dfs = dfs.rename('scores')
    #renames the scores as they previously inherited the name title
    L = pd.concat([df,dfs], axis= 1)
    #concatonates both dataframes turning them into a list
    dfn = pd.DataFrame(L, columns=["index", "year", "title", "scores"])
    #turns the list back into a dataframe
    return dfn

def preprocesser(Corpus):
    """
    Corpus: A dataframe containing a "text" segment 
    to be handed over to my NB or SVM model.
    """
    loadedvect = pickle.load(open(r'C:\Users\Seeratul\Documents\GitHub\BachelorThesis\code\vectorizer.sav', 'rb'))
    Corpusnew = Corpus
    Corpusnew['text'].dropna(inplace=True)
    # Step - 1 : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently                             
    Corpusnew['text'] = [entry.lower() for entry in Corpusnew['text']]
    # Step - 2 : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpusnew['text']= [word_tokenize(entry) for entry in Corpusnew['text']]
    # Step - 3 : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV 
    for index,entry in enumerate(Corpusnew['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpusnew.loc[index,'text_final'] = str(Final_words)
    return loadedvect.transform(Corpusnew["text_final"])

def vectroremapper(int):
    #necassery due to the weird neutral positive neg encoding of the encoder
    if int == 0:
        return -1
    elif int ==1:
        return 0
    else:
        return 1