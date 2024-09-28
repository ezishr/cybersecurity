#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

domaindatadf = pd.read_csv('dga_data_small.csv')

    

print(domaindatadf.tail())


# In[ ]:


print(domaindatadf.shape)


# In[ ]:


def catToNumber(isdga):
    if isdga == 'legit':
        return 1
    elif isdga == 'dga':
        return 0
domaincopydf['isDGA_N'] = domaincopydf['isDGA'].apply(catToNumber)
print(domaincopydf.tail())
#Converting DGA col


# In[ ]:


print(domaincopydf["subclass"].value_counts())
#Checking data split based on subclass column


# In[ ]:


def catToNumber2(subclass):
    if subclass == 'alexa':
        return 0
    elif subclass == 'legit':
        return 1
    elif subclass == 'cryptolocker':
        return 2
    elif subclass == 'newgoz':
        return 3
    elif subclass == 'necurs':
        return 4
    elif subclass == 'nivdort':
        return 5
    elif subclass == 'gameoverdga':
        return 6
    elif subclass == 'goz':
        return 7
    elif subclass == 'bamital':
        return 8
domaincopydf['subclass_N'] = domaincopydf['subclass'].apply(catToNumber2)

print(domaincopydf)
#Subclass converted into num values


# In[ ]:


domain_cross_check = pd.crosstab(domaincopydf['subclass'],domaincopydf['isDGA_N'])
print(domain_cross_check)
#Checking subclass split against isdga training column


# In[ ]:


#domaincopydf['topleveldomain'] = domaincopydf['host'].str.split('.').str[1]
domaincopydf['topleveldomain'] = domaincopydf['host'].str.split('.', 1).str[1]
#Splitting up top level domain


# In[ ]:


print(domaincopydf)


# In[ ]:


dummies = pd.get_dummies(domaincopydf['subclass'])
#domaincopydf.subclass_N = pd.get_dummies(domaincopydf, columns=['subclass_N'])
domaincopydf = domaincopydf.join(dummies)
domaincopydf
#Hot encoding


# In[ ]:


def digitsPerWord(word):
    sum=0
    for l in word:
        if l.isdigit():
            sum=sum+1     
    return sum 


domaincopydf['digitCount'] = domaincopydf['domain'].apply(digitsPerWord)

print(domaincopydf)
#Function for a digit count of each domain row


# In[ ]:


pd.set_option('display.max_rows', None)
print(domaincopydf["topleveldomain"].value_counts())
#Value split check through top level domain


# In[ ]:


domain_cross_check2 = pd.crosstab(domaincopydf['topleveldomain'],domaincopydf['isDGA_N'])
print(domain_cross_check2)
#Crosscheck through DGA col


# In[ ]:


def majority_value(domain_cross_check):
    zero = domain_cross_check[0]
    one = domain_cross_check[1]
    if zero > one:
        return 0
    else:
        return 1

domain_cross_check2['majority01'] = domain_cross_check2.apply(lambda row: majority_value(row), axis=1)
domain_cross_check2 = domain_cross_check2.sort_values(by=['majority01'], ascending=False)
print(domain_cross_check2)
#Don't use this. Removing this from code in future.


# In[ ]:


domainfirsttestsdf = domaincopydf.copy(deep=True)
domainfirsttestsdf.drop(['subclass_N'], axis=1, inplace=True)
domainfirsttestsdf
#Cat to Num conversion


# In[ ]:


def vowelRatio(word):
    count = 0
    for letter in word:
        if letter == "a" or letter == "e" or letter == "i" or letter == "o" or letter == "u":
            count = count + 1
    consonant = len(word) - count
    ratio = count / len(word)
    return ratio

domainfirsttestsdf['vowelratio'] = domainfirsttestsdf['domain'].apply(vowelRatio)
print(domainfirsttestsdf.tail())

#Vowel ratio column creator function
#Significant outside of the digit count


# In[ ]:


print(domainfirsttestsdf.columns)


# In[ ]:


def divideTLD(tld):
    if tld in ['to', 'ug', 'sc', 'sh', 'so', 'su', 'sx', 'ac', 'pro', 'ga', 'bit', 'bz', 'cm', 'cx', 'im', 'ki', 'la', 'mn', 'ms', 'nf']:
        return 0 # perfect majority 0
    elif tld in ['ru', 'tv', 'biz', 'tw', 'co', 'co.uk', 'org', 'info', 'net', 'xxx']:
        return 1 # uneven majority 0
    elif tld in ['hr', 'io', 'tk', 'ie', 'hu', 'am', 'hk', 'gr', 'gov.tw', 'gov.br', 'gov', 'gouv.fr', 'gob.ar', 'tn', 'fr', 'pt', 'is', 'it', 'nl', 'pl', 'ph', 'pe', 'org.br', 'ro', 'no', 'net.cn', 'jus.br', 'mu', 'se', 'me', 'lv', 'lt', 'fi', 'fm', 'edu.sa', 'co.in', 'co.id', 'edu', 'cn', 'ua', 'cl', 'ch', 'ca', 'blogspot.com', 'blog.br', 'vn', 'be', 'ba', 'az', 'at', 'asia', 'com.ar', 'com.au', 'com.br', 'com.cn', 'co.kr', 'com.tw', 'com.mx', 'com.tr', 'co.il', 'presse.fr', 'co.jp', 'com.my', 'do', 'dk', 'cz']:
        return 2 # perfect majority 1
    elif tld in ['eu', 'ir', 'in', 'jp', 'mx', 'cc', 'us', 'de', 'com']:
        return 3 # uneven majority 1
    elif tld in ['nu', 'kz']:
        return 4 # ties
    
domainfirsttestsdf['tlddivided'] = domaincopydf['topleveldomain'].apply(divideTLD)
print(domainfirsttestsdf)
#Ignore this column as well


# In[ ]:


namemap = {0.0: 'tld_p0', 1.0: 'tld_m0', 2.0: 'tld_p1', 3.0: 'tld_m1', 4.0: 'tld_t01'} 
domainfirsttestsdf = domainfirsttestsdf.rename(columns=namemap) 
domainfirsttestsdf


# In[ ]:


domainfirsttestsdf


# In[ ]:


namemap = {0.0: 'tld_p0', 1.0: 'tld_m0', 2.0: 'tld_p1', 3.0: 'tld_m1', 4.0: 'tld_t01'} 
domainfirsttestsdf = domainfirsttestsdf.rename(columns=namemap) 
domainfirsttestsdf


# In[ ]:


#This is my current code!

