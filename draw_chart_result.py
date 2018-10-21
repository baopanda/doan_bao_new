import pickle
from os.path import join
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import PreProcessing_valid

def LoadData(path_data,path_label):
    datas = []
    labels = []
    with open(path_data, 'r', encoding='utf-8')as file:
        for i in file:
            datas.append(i)

    with open(path_label, 'r', encoding='utf-8')as file:
        for i in file:
            labels.append(i)
    return datas, labels

NB=[]
SVC=[]
RF=[]

#AMBIENCE
accurancy_ambience = []
datas_valid = []
labels_valid = []
with open(join("data_test", "datas_AMBIENCE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_AMBIENCE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        labels_valid.append(i)
X_valid = datas_valid

load_file = open(join("models_NB","AMBIENCE.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_ambience.append(accuracy_score(labels_valid, a)*100)
NB.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_SVC","AMBIENCE.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_ambience.append(accuracy_score(labels_valid, a)*100)
SVC.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_RF","AMBIENCE.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_ambience.append(accuracy_score(labels_valid, a)*100)
RF.append(accuracy_score(labels_valid,a)*100)
print(accurancy_ambience)

#GENERAL
accurancy_general = []
datas_valid = []
labels_valid = []
with open(join("data_test", "datas_GENERAL.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_GENERAL.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        labels_valid.append(i)
X_valid = datas_valid

load_file = open(join("models_NB","GENERAL.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_general.append(accuracy_score(labels_valid, a)*100)
NB.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_SVC","GENERAL.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_general.append(accuracy_score(labels_valid, a)*100)
SVC.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_RF","GENERAL.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_general.append(accuracy_score(labels_valid, a)*100)
RF.append(accuracy_score(labels_valid,a)*100)
print(accurancy_general)

#LOCATION
accurancy_location = []
datas_valid = []
labels_valid = []
with open(join("data_test", "datas_LOCATION.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_LOCATION.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        labels_valid.append(i)
X_valid = datas_valid

load_file = open(join("models_NB","LOCATION.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_location.append(accuracy_score(labels_valid, a)*100)
NB.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_SVC","LOCATION.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_location.append(accuracy_score(labels_valid, a)*100)
SVC.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_RF","LOCATION.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_location.append(accuracy_score(labels_valid, a)*100)
RF.append(accuracy_score(labels_valid,a)*100)
print(accurancy_location)

#PRICES
accurancy_prices = []
datas_valid = []
labels_valid = []
with open(join("data_test", "datas_PRICES.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_PRICES.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        labels_valid.append(i)
X_valid = datas_valid

load_file = open(join("models_NB","PRICES.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_prices.append(accuracy_score(labels_valid, a)*100)
NB.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_SVC","PRICES.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_prices.append(accuracy_score(labels_valid, a)*100)
SVC.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_RF","PRICES.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_prices.append(accuracy_score(labels_valid, a)*100)
RF.append(accuracy_score(labels_valid,a)*100)
print(accurancy_prices)

#QUALITY
accurancy_quality = []
datas_valid = []
labels_valid = []
with open(join("data_test", "datas_QUALITY.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_QUALITY.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        labels_valid.append(i)
X_valid = datas_valid

load_file = open(join("models_NB","QUALITY.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_quality.append(accuracy_score(labels_valid, a)*100)
NB.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_SVC","QUALITY.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_quality.append(accuracy_score(labels_valid, a)*100)
SVC.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_RF","QUALITY.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_quality.append(accuracy_score(labels_valid, a)*100)
RF.append(accuracy_score(labels_valid,a)*100)
print(accurancy_quality)

#SERVICE
accurancy_service = []
datas_valid = []
labels_valid = []
with open(join("data_test", "datas_SERVICE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_SERVICE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        labels_valid.append(i)
X_valid = datas_valid

load_file = open(join("models_NB","SERVICE.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_service.append(accuracy_score(labels_valid, a)*100)
NB.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_SVC","SERVICE.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_service.append(accuracy_score(labels_valid, a)*100)
SVC.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_RF","SERVICE.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_service.append(accuracy_score(labels_valid, a)*100)
RF.append(accuracy_score(labels_valid,a)*100)
print(accurancy_service)

#STYLEOPTIONS
accurancy_styleoptions = []
datas_valid = []
labels_valid = []
with open(join("data_test", "datas_STYLEOPTIONS.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_STYLEOPTIONS.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        labels_valid.append(i)
X_valid = datas_valid

load_file = open(join("models_NB","STYLEOPTIONS.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_styleoptions.append(accuracy_score(labels_valid, a)*100)
NB.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_SVC","STYLEOPTIONS.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_styleoptions.append(accuracy_score(labels_valid, a)*100)
SVC.append(accuracy_score(labels_valid,a)*100)

load_file = open(join("models_RF","STYLEOPTIONS.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
accurancy_styleoptions.append(accuracy_score(labels_valid, a)*100)
RF.append(accuracy_score(labels_valid,a)*100)
print(accurancy_styleoptions)

print(NB)
print(SVC)
print(RF)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

name =['AMBIENCE','GENERAL','LOCATION','PRICES','QUALITY','SERVICE','STYLEOPTIONS']

df = pd.DataFrame({'models':name,'Naive_Bayes':NB,'Support_Vector_Machine': SVC,'Random_Forest': RF})
print(df)

# Setting the positions and width for the bars
pos = list(range(len(df['Naive_Bayes'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(10, 5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        df['Naive_Bayes'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='r',
        # with label the first value in first_name
        label=df['models'][0])

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['mid_score'] data,
        df['Support_Vector_Machine'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='g',
        # with label the second value in first_name
        label=df['models'][1])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width * 2 for p in pos],
        # using df['post_score'] data,
        df['Random_Forest'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='b',
        # with label the third value in first_name
        label=df['models'][2])

# Set the y axis label
ax.set_ylabel('Score')

# Set the chart's title
ax.set_title('Result')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['models'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 4)
plt.ylim([0, 110])

# Adding the legend and showing the plot
plt.legend(['Naive Bayes', 'Support Vector Machine', 'Random Forest'], loc='upper right')
plt.grid()
plt.savefig(join("images", "result.png"))
plt.show()

