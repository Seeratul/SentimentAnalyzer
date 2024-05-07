import pandas as pd

def annualizer(data,datapool,sentiment=0,ybase=1986,ymax=2017,margin=0.1):
    """
    Args: 
    data (pd): a panda containing a subset of data with a scores and a year atribute
    datapool(pd): a superset of data to which you want to relativize data
    sentiment(int): 0 means you want to count all, >0 means to only count positive,
                    <0 means to only count negative    
    ybase: start point in year attribute
    ymax: end point in year attribute 
    margin: a value between 1 and 0 to set how agressiveley the programm assigns neutal values"""
    #ybase = 1986
    year = 0
    yeararray = []
    if sentiment > 0:
        data = data.drop(data[data.scores < margin].index)
    if sentiment < 0:
        data = data.drop(data[data.scores >= -margin].index)
    while year + ybase < ymax:
        ycurrent = ybase + year
        if ((data[data['year']== ycurrent].shape[0]) != 0):
            size = (data[data['year']== ycurrent].shape[0])/(datapool[datapool['year']== ycurrent].shape[0])
        else:
            size = 0
        yeararray.append(size)
        year =year+1

    return(yeararray)

def keywordsearch(df,keys):
    """
    Args: 
    data (pd): a panda containing a subset of data with a title and a year atribute
    keys: the strings you are searching for, sensitive to spaces and capitalization given in this form " Lorem | Ipsum | etc " 
    """ 
    dfr = df[df['title'].str.contains(keys,na=False)] 


    return(dfr)