# -*- coding: utf-8 -*-
"""
Created on 9th April 2022

Book Ageism

@author: Taahirah Mangera

Can we predict the age group a children's book belongs to from its description'
"""

# %% 0. Import the python libraries you think you'll require

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skopt import BayesSearchCV
from sklearn import tree

#from sklearn.linear_model import LinearRegression

# %% 1. Load in the data. This can be done in a number of ways.
#   e.g. dataFrame = pd.read_csv('path to data file')

# path =  './bookData.csv'

# with open(path, 'r', encoding='utf-8', errors='ignore') as infile, open(path + 'enc.csv', 'w') as outfile:
#       inputs = csv.reader(infile)
#       output = csv.writer(outfile)

#       for index, row in enumerate(inputs):
#           # Create file with no header
#           if index == 0:
#               continue
#           output.writerow(row)

bookData = pd.read_csv('./bookDataenc.csv')

#%%Generate text-complexity predictors from book description##
#%% Length
# 1. Description length including all characters and spaces
descriptionLength = []

for row in bookData['Description']:
    stringLength = len(row)
    descriptionLength.append(stringLength)

bookData['Description_Length'] = descriptionLength

#%%UniqueChars
# 2. Number of unique characters


# Define a function to count the distinct/unique number of items in a set
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
#%%UniqueWords
# 3. Number of unique words

uniqueWords = []

for row in bookData['Description']:
    wordList = row.split()  # converting the string to a list
    uniqueWordsDes = countDis(wordList)
    uniqueWords.append(uniqueWordsDes)

bookData['Unique_Words'] = uniqueWords
#%%Longest word
# 4. Longest wordlength

longestWordLength = []

for row in bookData['Description']:
    wordList = row.split()  # converting the string to a list
    longestWord = max(wordList, key=len)
    longestWordLen = len(longestWord)
    longestWordLength.append(longestWordLen)

bookData['Length_Longest_Word'] = longestWordLength
#%%Number of words
# 5. Number of words

numberWords = []

for row in bookData['Description']:
    wordList = row.split()  # converting the string to a list
    words = len(wordList)
    numberWords.append(words)


bookData['NumberofWords'] = numberWords
#%%Longest Sentence and number of sentences
# Longest Sentence length

sentList = []
sentenceLength = []
numberSentences = []
 
for row in bookData['Description']:
    # converting the string to a list of sentences
    sentList = sent_tokenize(row, language="english")
    numberSentences.append(len(sentList))
    maxSentence = max(sentList, key=len)
    wordsMaxSentence = word_tokenize(maxSentence, language="english")
    numberWordsSentence = len(wordsMaxSentence)
    sentenceLength.append(numberWordsSentence)

bookData['LongestSentenceLength'] = sentenceLength
bookData['NumberSentences'] = numberSentences

#%%Word Frequency
# Word frequency
# Function to count word frequency


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

#%% Create key words as a text feature

# keyWordsNineTwelve = ["children", "story", "make", "reader", "help", "world", "series", "time", "friend", "adventure", "school", "life", "find"]
# keyWordsBabyTwo = ["board", "toddler", "fun", "parent", "baby", "babies", "learn", "little", "find", "touch", "feel"]
# keyWordsThreeFive = ["toddler", "story", "make", "reader", "help", "first", "learn", "little", "find", "touch", "feel"]
# keyWordsSixEight = ["fun","learn", "children", "story", "make", "reader", "help", "find", "world", "series", "friend"]
# keyWordsNineTwelve = ["children", "story", "make", "reader", "help", "world", "series", "time", "friend", "adventure", "school", "life", "find"]

keyWordsBabyTwo = ["board", "fun", "little", "help", "baby", "learn", "toddler"]
keyWordsThreeFive = ["story", "little", "fun", "make"]
keyWordsSixEight = ["help","world", "make", "fun"]
keyWordsNineTwelve = ["adventure", "friend", "time"]
n1 = len(keyWordsBabyTwo)
n2 = len(keyWordsThreeFive)
n3 = len(keyWordsSixEight)
n4 = len(keyWordsNineTwelve)

findKeyword1 = []
findKeyword2 = []
findKeyword3 = []
findKeyword4 = []

def sum_of_list(l):
  total = 0
  for val in l:
    total = total + val
  return total

keyword1 = []
for row in bookData['Description']:
        for str in keyWordsBabyTwo:
            if str in row:
                findKeyword1 = 1
                keyword1.append(findKeyword1)
            else:
                findKeyword1 = 0
                keyword1.append(findKeyword1)

keyword1Splits = [keyword1[x:x+n1] for x in range(0, len(keyword1), n1)]

percKeyFound1 = []
for row in keyword1Splits:
    percFound = round(sum_of_list(row)/n1*100)
    percKeyFound1.append(percFound)

bookData['Keywords0-2'] = percKeyFound1

keyword2 = []
for row in bookData['Description']:
    for str in keyWordsThreeFive:
        if str in row:
            findKeyword2 = 1
            keyword2.append(findKeyword2)
        else:
            findKeyword2 = 0
            keyword2.append(findKeyword2)

keyword2Splits = [keyword2[x:x+n2] for x in range(0, len(keyword2), n2)]

percKeyFound2 = []
for row in keyword2Splits:
    percFound = round(sum_of_list(row)/n2*100)
    percKeyFound2.append(percFound)

bookData['Keywords3-5'] = percKeyFound2

keyword3 = []
for row in bookData['Description']:
    for str in keyWordsSixEight:
        if str in row:
            findKeyword3 = 1
            keyword3.append(findKeyword3)
        else:
            findKeyword3 = 0
            keyword3.append(findKeyword3)

keyword3Splits = [keyword3[x:x+n3] for x in range(0, len(keyword3), n3)]

percKeyFound3 = []
for row in keyword3Splits:
    percFound = round(sum_of_list(row)/n3*100)
    percKeyFound3.append(percFound)

bookData['Keywords6-8'] = percKeyFound3

keyword4 = []
for row in bookData['Description']:
    for str in keyWordsNineTwelve:
        if str in row:
            findKeyword4 = 1
            keyword4.append(findKeyword4)
        else:
            findKeyword4 = 0
            keyword4.append(findKeyword4)

keyword4Splits = [keyword4[x:x+n4] for x in range(0, len(keyword4), n4)]

percKeyFound4 = []
for row in keyword4Splits:
    percFound = round(sum_of_list(row)/n4*100)
    percKeyFound4.append(percFound)

bookData['Keywords9-12'] = percKeyFound4



# keyWordsBabyTwo = ["board", "toddler", "fun", "parent", "baby", "babies", "learn", "little", "find", "touch", "feel"]
# keyWordsThreeFive = ["toddler", "story", "make", "reader", "help", "first", "learn", "little", "find", "touch", "feel"]
# keyWordsSixEight = ["fun","learn", "children", "story", "make", "reader", "help", "find", "world", "series", "friend"]
# keyWordsNineTwelve = ["children", "story", "make", "reader", "help", "world", "series", "time", "friend", "adventure", "school", "life", "find"]

keyWordsBabyTwo = ["board", "fun", "little", "help", "baby", "learn", "toddler"]
keyWordsThreeFive = ["story", "little", "fun", "make"]
keyWordsSixEight = ["help","world", "make", "fun"]
keyWordsNineTwelve = ["adventure", "friend", "time"]

n1 = len(keyWordsBabyTwo)
n2 = len(keyWordsThreeFive)
n3 = len(keyWordsSixEight)
n4 = len(keyWordsNineTwelve)

findKeyword1 = []
findKeyword2 = []
findKeyword3 = []
findKeyword4 = []

def sum_of_list(l):
  total = 0
  for val in l:
    total = total + val
  return total

keyword1 = []
for row in bookData['Description']:
        for str in keyWordsBabyTwo:
            if str in row:
                findKeyword1 = 1
                keyword1.append(findKeyword1)
            else:
                findKeyword1 = 0
                keyword1.append(findKeyword1)
            
keyword1Splits = [keyword1[x:x+n1] for x in range(0, len(keyword1), n1)]

percKeyFound1 = []
for row in keyword1Splits:
    percFound = round(sum_of_list(row)/n1*100)
    percKeyFound1.append(percFound)

bookData['Keywords0-2'] = percKeyFound1

keyword2 = []
for row in bookData['Description']:
    for str in keyWordsThreeFive:
        if str in row:
            findKeyword2 = 1
            keyword2.append(findKeyword2)
        else:
            findKeyword2 = 0
            keyword2.append(findKeyword2)
            
keyword2Splits = [keyword2[x:x+n2] for x in range(0, len(keyword2), n2)]

percKeyFound2 = []
for row in keyword2Splits:
    percFound = round(sum_of_list(row)/n2*100)
    percKeyFound2.append(percFound)

bookData['Keywords3-5'] = percKeyFound2

keyword3 = []
for row in bookData['Description']:
    for str in keyWordsSixEight:
        if str in row:
            findKeyword3 = 1
            keyword3.append(findKeyword3)
        else:
            findKeyword3 = 0
            keyword3.append(findKeyword3)
            
keyword3Splits = [keyword3[x:x+n3] for x in range(0, len(keyword3), n3)]

percKeyFound3 = []
for row in keyword3Splits:
    percFound = round(sum_of_list(row)/n3*100)
    percKeyFound3.append(percFound)

bookData['Keywords6-8'] = percKeyFound3

keyword4 = []
for row in bookData['Description']:
    for str in keyWordsNineTwelve:
        if str in row:
            findKeyword4 = 1
            keyword4.append(findKeyword4)
        else:
            findKeyword4 = 0
            keyword4.append(findKeyword4)
            
keyword4Splits = [keyword4[x:x+n4] for x in range(0, len(keyword4), n4)]

percKeyFound4 = []
for row in keyword4Splits:
    percFound = round(sum_of_list(row)/n4*100)
    percKeyFound4.append(percFound)

bookData['Keywords9-12'] = percKeyFound4        

# %% 2. Explore the data.
# #
# #   2.1 Visualise the data (How do the predictors/inputs relate to the responses/outputs?)

# crossTabWordFreq = pd.crosstab(
#     bookData['Age'], bookData['AverageWordFrequency'], margins=True)
# barPlotWordFreq = crossTabWordFreq.plot.bar(rot=0)

# crossTabSentLen = pd.crosstab(
#     bookData['Age'], bookData['LongestSentenceLength'], margins=True)
# print(crossTabSentLen)
# #barPlotSentLen = crossTabSentLen.plot.bar(rot=0)

# crossTabWordNo = pd.crosstab(
#     bookData['Age'], bookData['NumberofWords'], margins=True)
# barPlotWordNo = crossTabWordNo.plot.bar(rot=0)

# crossTabLongWord = pd.crosstab(
#     bookData['Age'], bookData['Length_Longest_Word'], margins=True)
# barPlotLongWord = crossTabLongWord.plot.bar(rot=0)

# crossTabUniqueWord = pd.crosstab(
#     bookData['Age'], bookData['Unique_Words'], margins=True)
# barPlotUniqueWord = crossTabUniqueWord.plot.bar(rot=0)

# crossTabUniqueChars = pd.crosstab(
#     bookData['Age'], bookData['Description_UniqueChars'], margins=True)
# barPlotUniqueChars = crossTabUniqueChars.plot.bar(rot=0)

# crossTabLength = pd.crosstab(
#     bookData['Age'], bookData['Keywords0-2'], margins=True)
# barPlotLength = crossTabLength.plot.bar(rot=0)

# crossTabLength = pd.crosstab(
#     bookData['Age'], bookData['Keywords3-5'], margins=True)
# barPlotLength = crossTabLength.plot.bar(rot=0)

# crossTabLength = pd.crosstab(
#     bookData['Age'], bookData['Keywords6-8'], margins=True)
# barPlotLength = crossTabLength.plot.bar(rot=0)

# crossTabLength = pd.crosstab(
#     bookData['Age'], bookData['Keywords9-12'], margins=True)
# barPlotLength = crossTabLength.plot.bar(rot=0)

# crossTabLength = pd.crosstab(
#     bookData['Age'], bookData['Description_Length'], margins=True)
# barPlotLength = crossTabLength.plot.bar(rot=0)

# bookSeries = []
# for column in bookData.columns[3:]:
#     plt.figure()
#     bookSeries = bookData.groupby([column, "Age"])[
#         column].count().unstack("Age").fillna(0)
#     bookSeries[["0 to 2", "3 to 5", "6 to 8", "9 to 12"]].plot(
#         kind="bar", stacked=True)

sns.pairplot(bookData)

# %% 3. Formulate the Problem
#   Decide on the predictors and the responses. If you had to implement this model
#   with live data streaming in, waht would it look like? Include pre-processing
#   where appropriate.

response = bookData[bookData.columns[2:3]]
predictorsAll = bookData[bookData.columns[3:14]]
predictors = predictorsAll.drop(['AverageWordFrequency'], axis = 1)

xTrainValid = predictors[0:int(0.85*len(predictors))]
yTrainValid = response[0:int(0.85*len(response))]

xTest = predictors[int(0.85*len(predictors)):]
yTest = response[int(0.85*len(response)):]

# %% 4. Create Baseline Model for Comparison
#   This is typically something extremely naive and simple. A "back-of-an-envelope"
#   attempt at the problem. Typical examples include assuming all the data belongs
#   to a single class (classification), or assuming the current prediction equals
#   the previous prediction (time series regression)

yHatBaseline = [True for i in range(len(yTrainValid))]
print(metrics.confusion_matrix(yTrainValid == '1', yHatBaseline))
print(metrics.accuracy_score(yTrainValid == '1', yHatBaseline))

# yHatBaseline =  KNeighborsClassifier(n_neighbors=2)
# yHatBaseline.fit(xTrainValid, yTrainValid)
# knn = yHatBaseline.predict(xTrainValid)
# print(metrics.confusion_matrix(yTrainValid, knn))
# print(metrics.accuracy_score(yTrainValid, knn))

# %% 5. Identify a Suitable Machine Learning Model

crossValObj = KFold(n_splits = 30)
yHatValidTotal = []
for trainIdx, validIdx in crossValObj.split(xTrainValid):
    xTrain, xValid = xTrainValid.loc[trainIdx], xTrainValid.loc[validIdx]
    yTrain, yValid = yTrainValid.loc[trainIdx], yTrainValid.loc[validIdx]
    mdl = KNeighborsClassifier(n_neighbors=2)
    mdl.fit(xTrain, yTrain)
    yHatValid = mdl.predict(xValid)
    yHatValidTotal = np.concatenate((yHatValidTotal, yHatValid), axis = 0)
print(metrics.confusion_matrix(yTrainValid, yHatValidTotal))
print(metrics.accuracy_score(yTrainValid, yHatValidTotal))

#%% Looking for Optimal Hyperparameters

params = {'n_neighbors':[1,2,3,4,5]}
opt = BayesSearchCV(mdl, params, cv = crossValObj, verbose = 5, n_iter = 10)
searchResults = opt.fit(xTrainValid, yTrainValid.value.ravel())

# %% 6. Compare your Model to the Baseline
#   Use the appropriate performance metrics where appropriate.
#   Classification - Accuracy, Precision, Recall, F1 score, AUC, ROC, etc.
#   Regression - MSE, Error distributions, R-squared


# %% 7. Add Complexity if Required

# %% 8. Repeat the Analysis as required

# %% 9. Answer the Question
