import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import stats

red_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep = ';')
white_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep = ';')

# create a new variable 'wine_type'
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

# bucket wine quality scores into qualitative quality labels
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
if value <= 5 else 'medium'
if value <= 7 else 'high')

red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'],
categories=['low', 'medium', 'high'])

white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
if value <= 5 else 'medium'
if value <= 7 else 'high')

white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'],
categories=['low', 'medium', 'high'])

wines = pd.concat([red_wine, white_wine])

wtp_features = wines.iloc[:,:-3]
wtp_feature_names = wtp_features.columns
wtp_class_labels = np.array(wines['wine_type'])

wtp_train_X, wtp_test_X, wtp_train_y, wtp_test_y = train_test_split(wtp_features,
wtp_class_labels, test_size=0.3, random_state=42)

print(Counter(wtp_train_y), Counter(wtp_test_y))
print('Features:', list(wtp_feature_names))
#wines.head(2)
# Define the scaler
wtp_ss = StandardScaler().fit(wtp_train_X)
# Scale the train set
wtp_train_SX = wtp_ss.transform(wtp_train_X)
# Scale the test set
wtp_test_SX = wtp_ss.transform(wtp_test_X)

wtp_lr = LogisticRegression()
wtp_lr.fit(wtp_train_SX, wtp_train_y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
verbose=0, warm_start=False)

wtp_lr_predictions = wtp_lr.predict(wtp_test_SX)

print(classification_report(wtp_test_y,wtp_lr_predictions, target_names=['red', 'white']))
#f, ax = plt.subplots(1, 2, figsize=(10, 4))
#sns.set_theme(style="ticks", palette="pastel")
#f.suptitle('Wine Quality - Alcohol Content/ Sulphates', fontsize=14)

#f.subplots_adjust(top=0.85, wspace=0.3)

#sns.boxplot(x="wine_type", y="alcohol", data=wines, ax=ax[0])

#ax[0].set_xlabel("Wine Quality Class",size = 12,alpha=0.8)
#ax[0].set_ylabel("Wine Alcohol %",size = 12,alpha=0.8)

#sns.boxplot(x="wine_type", y="sulphates", data=wines, ax=ax[1])

#ax[1].set_xlabel("Wine Quality Class",size = 12,alpha=0.8)
#ax[1].set_ylabel("Wine Alcohol %",size = 12,alpha=0.8)

#sns.set_theme(style="ticks", palette="pastel")

#f, ax = plt.subplots(figsize=(10, 4))

# Draw a nested boxplot to show bills by day and time
#sns.boxplot(x="quality_label", y="alcohol", palette=["m", "g"], hue = 'wine_type', data=wines, ax = ax)
#sns.boxplot(x="quality_label", y="sulphates", hue = 'wine_type', data=wines, ax = ax)

#sns.despine(offset=10, trim=True)
#ax.legend(loc='upper left')

#plt.pie([x*100 for x in d_cuisine.values()],labels=[x for x in
#d_cuisine.keys()],autopct='%0.1f',explode=[0,0,0.1,0])

#wines['quality_label'].value_counts().plot.pie()
#label the plot

#plt.title('Wine Quality %')





#wines.quality_label.value_counts().plot(kind = 'bar')

#sns.kdeplot(data=wines, x="alcohol", hue = "quality" , multiple="stack")

#sns.kdeplot(data=wines[wines['wine_type'] == 'red'], x="alcohol")

#sns.kdeplot(data=wines[wines['wine_type'] == 'white'], x="alcohol")

#wines.plot.scatter(x='sulphates', y='alcohol')

#wines.columns 

#sns.lmplot(x='residual sugar', y='alcohol', data = wines, hue = 'wine_type')

#F, p = stats.f_oneway(wines[wines['quality_label'] == 'low']['alcohol'],
#wines[wines['quality_label'] == 'medium']['alcohol'],
#wines[wines['quality_label'] == 'high']['alcohol'])
#print('ANOVA test for mean alcohol levels across wine samples with different quality ratings')
#print('F Statistic:', F, '\tp-value:', p)

#red_wine.hist(bins=15, color='red', edgecolor='black', linewidth=1.0,
#xlabelsize=8, ylabelsize=8, grid=False)

#plt.tight_layout(rect=(0, 0, 1.2, 1.2))

#rt = plt.suptitle('Red Wine Univariate Plots', x=0.65, y=1.25, fontsize=15)

#fig = plt.figure(figsize = (10,4))
#title = fig.suptitle("Residual Sugar Content in Wine", fontsize=14)
#fig.subplots_adjust(top=0.85, wspace=0.3)
#ax1 = fig.add_subplot(1,2, 1)
#ax1.set_title("Red Wine")
#ax1.set_xlabel("Residual Sugar")
#ax1.set_ylabel("Frequency")
#ax1.set_ylim([0, 2500])
#ax1.text(8, 1000, r'$\mu$='+str(round(red_wine['residual sugar'].mean(),2)),
#fontsize=12)
#r_freq, r_bins, r_patches = ax1.hist(red_wine['residual sugar'], color='red', bins=15,
#edgecolor='black', linewidth=1)
#ax2 = fig.add_subplot(1,2, 2)
#ax2.set_title("White Wine")
#ax2.set_xlabel("Residual Sugar")
#ax2.set_ylabel("Frequency")
#ax2.set_ylim([0, 2500])
#ax2.text(30, 1000, r'$\mu$='+str(round(white_wine['residual sugar'].mean(),2)),
#fontsize=12)
#w_freq, w_bins, w_patches = ax2.hist(white_wine['residual sugar'], color='white', bins=15,
#edgecolor='black', linewidth=1)

#rj = sns.jointplot(x='quality', y='sulphates', data=red_wine,
#kind='reg', ylim=(0, 2),
#color='red', space=0, size=4.5, ratio=4)
#rj.ax_joint.set_xticks(list(range(3,9)))
#fig = rj.fig
#fig.subplots_adjust(top=0.9)
#t = fig.suptitle('Red Wine Sulphates - Quality', fontsize=12)

#plt.show()


#plt.savefig('C:\\Users\\HP\\Downloads\\Wein\\plots\\plotting_.png',dpi=300,bbox_inches='tight')