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
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
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
df001=pd.DataFrame(y)
for i, val in enumerate(df001.values):
    plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})

index =np.arange(len(x))

plt.bar(x,y)

plt.xlabel('Technologies')
plt.ylabel('Number of students ')

plt.xticks(index,x,fontsize=7,rotation=30)

plt.title('Number of students applied to different Technologies')
plt.savefig("om1.png")
plt.show()


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
df001=pd.DataFrame(y01)
for i, val in enumerate(df001.values):
    plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})




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
df001=pd.DataFrame(y)
for i, val in enumerate(df001.values):
    plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})

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

df001=pd.DataFrame(y01)
for i, val in enumerate(df001.values):
    plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})

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
df001=pd.DataFrame(y01)
for i, val in enumerate(df001.values):
    plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})

index =np.arange(len(x01))
plt.bar(x01,y01)
#plt.legend()
plt.xlabel('Verbal and written communication score greater than 8')
plt.ylabel('Number of students ')

plt.title('Digital Marketing Students with score greater than 8.')
plt.savefig("om5.png")
plt.show()

#-----------------------------------------6---------------------------------


x21=df1['Which-year are you studying in?']
x21=list(x21)
a2=[x21[4],x21[0],x21[6],x21[1]]

fig,a = plt.subplots(2,2)

ko=0
x22 =df1['Areas of interest']
plt.figure(6)
for i in range(len(a2)):
    a1=[]
    for j in range(len(x21)):
        if x21[j]==a2[i]:
            a1.append(x22[j])

    y1= collections.Counter(a1)

    x=y1.keys()
    y=y1.values()
    df001 = pd.DataFrame(y)
    for i1, val in enumerate(df001.values):
        plt.text(i1, val, int(val), horizontalalignment='center', verticalalignment='bottom',
                 fontdict={'fontweight': 500, 'size': 9})

    index =np.arange(len(x))
    #plt.bar(x,y)
    #plt.legend()
    if ko==0:
        a[0][0].bar(x,y)
        a[0][0].set_title(a2[i])
        plt.xticks(index, x, fontsize=7, rotation=30)
        plt.xlabel('Source')
        plt.ylabel('Number of students ')
    elif ko==1:
        a[0][1].bar(x,y)
        a[0][1].set_title(a2[i])
        plt.xticks(index, x, fontsize=7, rotation=30)
        plt.xlabel('Source')
        plt.ylabel('Number of students ')
    elif ko==2:
        a[1][0].bar(x,y)
        a[1][0].set_title(a2[i])
        plt.xticks(index, x, fontsize=7, rotation=30)
        plt.xlabel('Source')
        plt.ylabel('Number of students ')
    else:
        a[1][1].bar(x, y)
        a[1][1].set_title(a2[i])
        plt.xticks(index, x, fontsize=7, rotation=90)
        plt.xlabel('Source')
        plt.ylabel('Number of students ')
    ko+=1


    #plt.title(a2[i])
plt.savefig("om%s.png"%(6))
plt.show()



#-----------------------------------------7.1--------------------------------
plt.figure(10)

x1=df1['City']
y1= collections.Counter(x1)
x=y1.keys()
y=y1.values()
df001=pd.DataFrame(y)
for i, val in enumerate(df001.values):
    plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})

index =np.arange(len(x))

aa=list(y)

sizes = []
s=sum(aa)
for i in range(len(aa)):
    d= aa[i]/s
    sizes.append(d)

colors =['yellowgreen','gold','orange','lightcoral','lightskyblue','grey']
explode = (0, 0, 0.1, 0,0,0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=x, autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
'''
plt.show()

plt.xlabel('Cities')
plt.ylabel('Number of students ')

plt.xticks(index,x,fontsize=7,rotation=30)
'''
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




#-----------------------------------------8--------------------------------
plt.figure(12)

width=0.35


le=[]
lie=[]

x21=df1['Areas of interest']
x21=list(x21)
a2=list(set(x21))

front = df['Label']
df.drop(labels=['Label'], axis=1, inplace=True)
df.insert(0, 'Label', front)

y21 = df.Label
y21 = list(y21)

for i in range(len(a2)):
    e = 0
    ie = 0
    for j in range(len(x21)):
        if a2[i]==x21[j]:
            if y21[j] == y21[1]:
                e = e + 1
            else:
                ie = ie + 1
    le.append(e)
    lie.append(ie)


# the x locations for the groups
width = 0.35
# the width of the bars: can also be len(x) sequence
index =np.arange(len(a2))
p1 = plt.bar(index, lie, width)
p2 = plt.bar(index, le, width)


plt.ylabel('No. of Students')
plt.title('Area of Interest v/s Target Variables')
plt.xticks(index, a2,fontsize=7,rotation='vertical')
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Not Eligible', 'Eligible'))



plt.savefig("om12.png")
plt.show()





#-----------------------------------------9--------------------------------
plt.figure(14)

width=0.35


le=[]
lie=[]

x21=df1['Major/Area of Study']
x21=list(x21)
a2=list(set(x21))

front = df['Label']
df.drop(labels=['Label'], axis=1, inplace=True)
df.insert(0, 'Label', front)

y21 = df.Label
y21 = list(y21)

for i in range(len(a2)):
    e = 0
    ie = 0
    for j in range(len(x21)):
        if a2[i]==x21[j]:
            if y21[j] == y21[1]:
                e = e + 1
            else:
                ie = ie + 1
    le.append(e)
    lie.append(ie)


# the x locations for the groups
width = 0.35
# the width of the bars: can also be len(x) sequence
index =np.arange(len(a2))
p1 = plt.bar(index, lie, width)
p2 = plt.bar(index, le, width)


plt.ylabel('No. of Students')
plt.title('Majors v/s Target Variables')
plt.xticks(index, a2,fontsize=7,rotation='vertical')
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Not Eligible', 'Eligible'))
plt.savefig("om14.png")
plt.show()








#-----------------------------------------10--------------------------------
plt.figure(15)

width=0.35


le=[]
lie=[]

x21=df1['Which-year are you studying in?']
x21=list(x21)
a2=[x21[4],x21[0],x21[6],x21[1]]

front = df['Label']
df.drop(labels=['Label'], axis=1, inplace=True)
df.insert(0, 'Label', front)

y21 = df.Label
y21 = list(y21)

for i in range(len(a2)):
    e = 0
    ie = 0
    for j in range(len(x21)):
        if a2[i]==x21[j]:
            if y21[j] == y21[1]:
                e = e + 1
            else:
                ie = ie + 1
    le.append(e)
    lie.append(ie)



# the x locations for the groups
width = 0.35
# the width of the bars: can also be len(x) sequence
index =np.arange(len(a2))
p1 = plt.bar(index, le, width)
p2 = plt.bar(index, lie, width)


plt.ylabel('No. of Students')
plt.title('Year of Study v/s Target Variables')
plt.xticks(index, a2,fontsize=7,rotation='vertical')
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p2[0], p1[0]), ('Not Eligible', 'Eligible'))
plt.savefig("om15.png")
plt.show()



#-----------------------------------------11--------------------------------
plt.figure(16)

width=0.35


le=[]
lie=[]

x21=df1['CGPA/ percentage']
x21=list(x21)

for i in range(len(x21)):
    if x21[i]>=9.5:
        x21[i]='More than 9.5'
    elif 9.0<=x21[i]<9.5:
        x21[i]='9.0-9.5'
    elif 8.5<=x21[i]<9.0:
        x21[i]='8.5-9.0'
    elif 8.0<=x21[i]<8.5:
        x21[i]='8.0-8.5'
    elif 7.5<=x21[i]<8.0:
        x21[i]='7.5-8.0'
    else:
        x21[i]='Less than 7.5'


a2=['More than 9.5','9.0-9.5','8.5-9.0','7.5-8.0','Less than 7.5']

front = df['Label']
df.drop(labels=['Label'], axis=1, inplace=True)
df.insert(0, 'Label', front)

y21 = df.Label
y21 = list(y21)

for i in range(len(a2)):
    e = 0
    ie = 0
    for j in range(len(x21)):
        if a2[i]==x21[j]:
            if y21[j] == y21[1]:
                e = e + 1
            else:
                ie = ie + 1
    le.append(e)
    lie.append(ie)



# the x locations for the groups
width = 0.35
# the width of the bars: can also be len(x) sequence
index =np.arange(len(a2))
p1 = plt.bar(index, lie, width)
p2 = plt.bar(index, le, width)

plt.xlabel('CGPA')
plt.ylabel('No. of Students')
plt.title('CGPA v/s Target Variables')
plt.xticks(index, a2,fontsize=7)
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p2[0], p1[0]), ('Eligible', 'Not Eligible'))
plt.savefig("om16.png")
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
#clf=GaussianNB()
#clf = svm.SVC()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print('Accuracy Score :', (accuracy_score(y_test, y_pred)*100))
print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred))


print(classification_report(y_test, y_pred))
'''

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = f_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Decision_Tree.png')
Image(graph.create_png())
'''
