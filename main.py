import pandas as pd
from Collector import sentimentmapper
from annualizer import annualizer, keywordsearch
import matplotlib.pyplot as plt 
import numpy as np 

if __name__ == "__main__" :
    # Below here are things you will probably need to change, just follow the comments.
    
    # dfoo is uspposed to load the original dataset you want to analyze
    dfoo = pd.read_csv(r'C:\Users\Seeratul\Documents\GitHub\BachelorThesis\data\nyt_data.csv') 
    # You will need to change the path here 
    # The data you will load requires the following features:
    # A "year" collum containing the year and only the year
    # A "title" collum containing the string that is to be analyzed


    # The keys are the words that marke the topic we are looking for, the code will look for 
    # strings that match the strings given below and count them as examples of the topic.
    # Unless you are just testing my implementation on the same dataset you will probably want 
    # to pick keys appropriate to your topic.
    keys = ' AI |Artificial Intelligence|robot| ai |Robot|artificial intelligence|AI '

    #Select the duration you analysis is supposed to span, will not work if  the years arent in the provied data.
    ybase = 1987
    ymax = 2017

    #select the sentiment mapper to use Vader and Textblob are reommended SiEBERT is also supported 
    #and FelixNB and FelixSVM are in the code but neither supported nor recommended nor good.
    #maptype = Vader or Siebert or TextBlob
    maptype= "Vader"

    #If you use Vader or Textblob this will select how many datatpoints are classified as neutral
    #greater margin, more neutral
    margin = 0.1

    ###############################################################################################################
    # 
    #                   The code below dosen´t need adjustment untill it gets to the plotting part
    #                   there will be a box like this there, untill then, no touching!
    #
    ###############################################################################################################



    dfo = dfoo[dfoo['year'] >= ybase ]
    dfo = dfo[dfo['year'] <= ymax ]
    #cuts the original data to the time frame of reference

    df = keywordsearch(dfo,keys)
    #df = pd.read_pickle(r'C:\Users\Seeratul\Documents\GitHub\BachelorThesis\code\SaveHeadline.pkl')
    #Load the data
    df = df[df['year'] >= ybase ]
    df = df[df['year'] <= ymax ]

    dfsm = sentimentmapper(df,maptype)
    #Maps sentiments to df creates dataframe sentimentmapped
    #Takes "Siebert" or "Vader" as arguments

    # Using range to create an array X 
    X = range(ybase,ymax,1)
  
    # Assign variables to the y axis part of the curve 
    neutral = annualizer(dfsm,dfo, ybase=ybase,ymax=ymax,)
    positive = annualizer(dfsm,dfo,1,ybase=ybase,ymax=ymax, margin= margin)
    negative = annualizer(dfsm,dfo,-1,ybase=ybase,ymax=ymax,margin= margin)

  
    # Plotting all the curves simultaneously 
    plt.plot(X, neutral, color='b', label='total') 
    plt.plot(X, positive, color='g', label='positive') 
    plt.plot(X, negative, color='y', label='negative') 

    ###############################################################################################################
    # 
    #                   Welcome to "The Plotting part" 
    #                    
    #
    ###############################################################################################################
  
    # Naming the x-axis, y-axis and the whole graph 
    # You will probably want to change the title.
    plt.xlabel("Year") 
    plt.ylabel("Percentage") 
    plt.title("Percentage of Headlines that discuss Ai" + maptype) 
  
    # Adding legend, which helps us recognize the curve according to it's color 
    plt.legend() 
  
    # To load the display window 
    plt.show() 

    #here we save the results to a csv if you wish you can change its name or remove this part if you just want the graph
    #and dont mind rerunning the programm if you want the results again.
    d = {'neutral': neutral, 'positive': positive, 'negative': negative} 
    dftot = pd.DataFrame(data=d)
    dftot.to_csv('datadump_name_17.csv')
    