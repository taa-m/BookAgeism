# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:08:48 2021

Concrete Example

@author: john.atherfold

Can we predict the strength of the concrete given the various inputs
"""

#%% 0. Import the python libraries you think you'll require

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#from sklearn.linear_model import LinearRegression

#%% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')

# path =  './bookData_descriptions_nobox.csv'

# with open(path, 'r', encoding='utf-8', errors='ignore') as infile, open(path + 'bookData.csv', 'w') as outfile:
#      inputs = csv.reader(infile)
#      output = csv.writer(outfile)

#      for index, row in enumerate(inputs):
#          # Create file with no header
#          if index == 0:
#              continue
#          output.writerow(row)

bookData = pd.read_csv('./bookData.csv')

#%%Generate text-complexity predictors from book description##

#1. Description length including all characters and spaces

descriptionLength = []

for row in bookData['Description']:
    stringLength = len(row)
    descriptionLength.append(stringLength)

bookData['Description_Length'] = descriptionLength    

#2. Number of unique characters


#Define a function to count the distinct/unique number of items in a set
def countDis(str):
 
    # Stores all distinct characters
    s = set(str)
 
    # Return the size of the set
    return len(s)

descriptionUniqueChars = []

for row in bookData['Description']:
    uniqueChars = countDis(row)
    descriptionUniqueChars.append(uniqueChars)

bookData['Description_UniqueChars'] = descriptionUniqueChars   

#3. Number of unique words

uniqueWords = []

for row in bookData['Description']:
    wordList = row.split() #converting the string to a list
    uniqueWordsDes = countDis(wordList)
    uniqueWords.append(uniqueWordsDes)

bookData['Unique_Words'] = uniqueWords  

#4. Longest wordlength

longestWordLength = []

for row in bookData['Description']:
    wordList = row.split() #converting the string to a list
    longestWord = max(wordList, key=len)
    longestWordLen = len(longestWord)
    longestWordLength.append(longestWordLen)

bookData['Length_Longest_Word'] = longestWordLength

#5. Number of words

numberWords = []

for row in bookData['Description']:
    wordList = row.split() #converting the string to a list
    words = len(wordList)
    numberWords.append(words)
    

bookData['NumberofWords'] = numberWords  

#Longest Sentence length

sentenceLength = []

for row in bookData['Description']:
    sentList = sent_tokenize(row, language="english") #converting the string to a list of sentences
    maxSentence = max(sentList, key=len)
    wordsMaxSentence = word_tokenize(maxSentence, language="english")
    numberWordsSentence = len(wordsMaxSentence)
    sentenceLength.append(numberWordsSentence)

bookData['LongestSentenceLength'] = sentenceLength  

#Word frequency
#Function to count word frequency
def freq(str):
  
    # break the string into list of words 
    str = str.split()         
    str2 = []
    listwords = []
  
    # loop till string values present in list str
    for i in str:             
  
        # checking for the duplicacy
        if i not in str2:
  
            # insert value in str2
            str2.append(i) 
              
    for i in range(0, len(str2)):
  
        # count the frequency of each word(present 
        # in str2) in str and print
        listwords.append(str.count(str2[i]))  
        average = sum(listwords)/len(listwords)
        return average
    
avWordFreq = []

for row in bookData['Description']:
    wordFreq = freq(row)
    avWordFreq.append(wordFreq)
    

bookData['AverageWordFrequency'] = avWordFreq


#%% 2. Explore the data.
#
#   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)

bookData['Age_lower'].value_counts().plot(kind = 'bar')
bookData['Age_upper'].value_counts().plot(kind = 'bar')
sns.pairplot(bookData)

# for column in bookData.columns[4:]:
#     plt.figure()
#     bookSeries = bookData.groupby([column,"Age_lower"])[column].count().unstack("Age_lower").fillna(0)
#     bookSeries[["1","2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]].plot(kind = "bar", stacked = True)
    
# crossTab = pd.crosstab(bookData['AverageWordFrequency'], bookData['LongestSentenceLength'], margins = True)
# print(crossTab)
#chi2, pval = stats.chi2_contingency(crossTab.values)[0:2]
#%% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.

responses = bookData[bookData.columns[3:5]]
predictors = bookData[bookData.columns[5:13]]

#xTrainValid = predictors[0:int(0.85*len(predictors))]
#yTrainValid = response[0:int(0.85*len(response))]

#xTest = predictors[int(0.85*len(predictors)):]
#yTest = response[int(0.85*len(response)):]

#%% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

#mdl = LinearRegression()
#mdl.fit(xTrainValid, yTrainValid)

#yHatTrainValid = mdl.predict(xTrainValid)
#yHatTest = mdl.predict(xTest)

#plt.figure()
#plt.hist(yTrainValid - yHatTrainValid)
#plt.title('Residuals')

#plt.figure()
#plt.hist(yTest - yHatTest)
#plt.title('Residuals')

#%% 5. Identify a Suitable Machine Learning Model



#%% 6. Compare your Model to the Baseline
#   Use the appropriate performance metrics where appropriate.
#   Classification - Accuracy, Precision, Recall, F1 score, AUC, ROC, etc.
#   Regression - MSE, Error distributions, R-squared


#%% 7. Add Complexity if Required

#%% 8. Repeat the Analysis as required

#%% 9. Answer the Question