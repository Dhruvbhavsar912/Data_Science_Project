import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import pydotplus
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
a=pd.read_csv('DS_DATESET.csv')

df=pd.DataFrame(a)

df1 = df.drop(['Certifications/Achievement/ Research papers','Link to updated Resume (Google/ One Drive link preferred)','link to Linkedin profile'],axis=1)

#print(df1.head)


#-----------------------------------------1--------------------------------
plt.figure(1)

x1=df1['Areas of interest']
y1= collections.Counter(x1)
x=y1.keys()
y=y1.values()


index =np.arange(len(x))
plt.bar(x,y)

plt.xlabel('Technologies')
plt.ylabel('Number of students ')

plt.xticks(index,x,fontsize=7,rotation=30)

plt.title('Number of students applied to different Technologies')
plt.savefig("om1.png")



#-----------------------------------------2--------------------------------------------------

plt.figure(2)
x1=df1['Areas of interest']
x2= df1['Programming Language Known other than Java (one major)']
py=0
pyn=0
x1=list(x1)
x2=list(x2)
for i in range(len(x1)):
    if x1[i]== x1[7] and x2[i]==x2[4]:
        py=py+1
    elif x1[i]== x1[7] and x2[i] != x2[4]:
        pyn=pyn +1

x01=['YES' , 'NO']
y01=[py,pyn]
index =np.arange(len(x01))
plt.bar(x01,y01)
plt.xlabel('Students who knew Python')
plt.ylabel('Number of students ')

plt.title('Number of students applied for Data Science who knew ‘’Python” ')
plt.savefig("om2.png")

plt.show()

#-----------------------------------------3---------------------------------


plt.figure(3)
x1=df1['How Did You Hear About This Internship?']
y1= collections.Counter(x1)
x=y1.keys()
y=y1.values()
index =np.arange(len(x))
plt.bar(x,y)

plt.xlabel('Source')
plt.ylabel('Number of students ')

plt.xticks(index,x,fontsize=7,rotation=30)

plt.title('Different ways students learned about this program.')
plt.savefig("om3.png")
plt.show()


#-----------------------------------4------------------------------------


plt.figure(4)
x1=df1['Which-year are you studying in?']
x2= df1['CGPA/ percentage']
cg=0
cgn=0
x1=list(x1)
x2=list(x2)
for i in range(len(x1)):
    if x1[i]== x1[1] and x2[i]>=8:
        cg=cg+1
    elif x1[i]== x1[1] and x2[i]<8:
        cgn=cgn +1

x01=['YES' , 'NO']
y01=[cg,cgn]
index =np.arange(len(x01))
plt.bar(x01,y01)
plt.xlabel('Students with CGPA greater than 8.0')
plt.ylabel('Number of students ')

plt.title('Students in the fourth year and have a CGPA greater than 8.0')
plt.savefig("om4.png")

plt.show()



#-----------------------------------------5--------------------------------------------------

plt.figure(5)
x1=df1['Areas of interest']
x2= df1['Rate your written communication skills [1-10]']
x3= df1['Rate your verbal communication skills [1-10]']
dm=0
dmn=0
x1=list(x1)
x2=list(x2)
for i in range(len(x1)):
    if x1[i]== x1[3] and x2[i]>=8 and x3[i]>=8:
        dm=dm+1
    elif x1[i]== x1[3] and x2[i]<8 and x3[i]<8:
        dmn=dmn +1

x01=['YES' , 'NO']
y01=[dm,dmn]
index =np.arange(len(x01))
plt.bar(x01,y01)
#plt.legend()
plt.xlabel('Verbal and written communication score greater than 8')
plt.ylabel('Number of students ')

plt.title('Digital Marketing Students with score greater than 8.')
plt.savefig("om5.png")
plt.show()

#-----------------------------------------6---------------------------------


x21=df1['Expected Graduation-year']
x21=list(x21)
a2=list(set(x21))


x22 =df1['Areas of interest']
for i in range(len(a2)):
    plt.figure(6+i)
    a1=[]
    for j in range(len(x21)):
        if x21[j]==a2[i]:
            a1.append(x22[j])

    y1= collections.Counter(a1)

    x=y1.keys()
    y=y1.values()
    index =np.arange(len(x))
    plt.bar(x,y)
    #plt.legend()
    plt.xlabel('Source')
    plt.ylabel('Number of students ')
    plt.xticks(index,x,fontsize=7,rotation=30)
    plt.title(a2[i])
    plt.savefig("om%s.png"%(6+i))
    plt.show()



#-----------------------------------------7.1--------------------------------
plt.figure(10)

x1=df1['City']
y1= collections.Counter(x1)
x=y1.keys()
y=y1.values()
index =np.arange(len(x))
plt.bar(x,y)

plt.xlabel('Cities')
plt.ylabel('Number of students ')

plt.xticks(index,x,fontsize=7,rotation=30)

plt.title('Number of students from different City')
plt.savefig("om10.png")
plt.show()



#-----------------------------------------7.2--------------------------------
plt.figure(11)

x1=df1['College name']
y1= collections.Counter(x1)
x=y1.keys()
y=y1.values()
df001=pd.DataFrame(y)
for i, val in enumerate(df001.values):
    plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})



index =np.arange(len(x))
plt.bar(x,y)
#plt.scatter(x,y)
plt.xlabel('College names')
plt.ylabel('Number of students ')

plt.xticks(index,x,fontsize=7,rotation='vertical')

plt.title('Number of students from different College')
plt.savefig("om11.png")
plt.show()



#==============================================Classification===========================================


df["Expected Graduation-year"] = df["Expected Graduation-year"].astype('category').cat.codes
df['CGPA/ percentage'] = df['CGPA/ percentage'].astype('category').cat.codes


df["Areas of interest"] = df["Areas of interest"].astype('category').cat.codes
df["Have you worked core Java"] = df["Have you worked core Java"].astype('category').cat.codes

df["Programming Language Known other than Java (one major)"] = df["Programming Language Known other than Java (one major)"].astype('category').cat.codes
df["Have you worked on MySQL or Oracle database"] = df["Have you worked on MySQL or Oracle database"].astype('category').cat.codes

df["Have you studied OOP Concepts"] = df["Have you studied OOP Concepts"].astype('category').cat.codes
df["How Did You Hear About This Internship?"] = df["How Did You Hear About This Internship?"].astype('category').cat.codes





front = df['Label']
df.drop(labels=['Label'], axis=1,inplace = True)
df.insert(0, 'Label', front)

f_cols = ['CGPA/ percentage','Expected Graduation-year','Areas of interest','Have you worked core Java','Programming Language Known other than Java (one major)','Have you worked on MySQL or Oracle database','Have you studied OOP Concepts','Rate your written communication skills [1-10]','Rate your verbal communication skills [1-10]','How Did You Hear About This Internship?']
x = df[f_cols]
y = df.Label



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = RandomForestClassifier(n_estimators= 1)

#clf = svm.SVC()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print('Accuracy Score :', (accuracy_score(y_test, y_pred)*100))
print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred))


print(classification_report(y_test, y_pred))



