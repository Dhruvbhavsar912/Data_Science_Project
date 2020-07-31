import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import accuracy_score,f1_score
import warnings

warnings.simplefilter("ignore",UserWarning)


a=pd.read_csv(sys.argv[1], header = 0)

df=pd.DataFrame(a)

df1 = df.drop(['Certifications/Achievement/ Research papers','Link to updated Resume (Google/ One Drive link preferred)','link to Linkedin profile'],axis=1)

#print(df1.head)


#pp=PdfPages('OM.pdf')
#-----------------------------------------1--------------------------------

with PdfPages('Output.pdf') as pdf:
    #plt.figure(figsize=(18,12))
    rcParams.update({'figure.autolayout': True})

    x1=df1['Areas of interest']
    y1= collections.Counter(x1)
    x=y1.keys()
    y=y1.values()
    df001=pd.DataFrame(y)
    for i, val in enumerate(df001.values):
        plt.text(i,val,int(val),horizontalalignment = 'center', verticalalignment ='bottom', fontdict={'fontweight':500,'size':9})

    index =np.arange(len(x))
    x=list(x)
    y=list(y)


    plt.plot(x,y)

    plt.xlabel('Technologies',fontsize=9)
    plt.ylabel('Number of students ',fontsize=9)

    plt.xticks(index,x,fontsize=10,rotation=90)

    plt.title('Number of students applied to different Technologies',fontsize=10)
    #plt.savefig("om1.pdf")
    pdf.savefig()
    plt.close()


    #-----------------------------------------2--------------------------------------------------

    #plt.figure(figsize=(18, 12))
    rcParams.update({'figure.autolayout': True})
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
    aa = y01
    sizes = []
    s=sum(aa)
    for i in range(len(aa)):
        d= aa[i]/s
        sizes.append(d)

    colors =['grey','orange']
    explode = (0, 0.1)

    plt.pie(sizes, explode=explode, labels=x01, autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



    plt.title('Number of students applied for Data Science who knew ‘’Python” ')
    #plt.savefig("om1.pdf")
    pdf.savefig()

    plt.close()

    #-----------------------------------------3---------------------------------


    rcParams.update({'figure.autolayout': True})
    x1=df1['How Did You Hear About This Internship?']
    y1= collections.Counter(x1)
    x=y1.keys()
    y=y1.values()


    aa=list(y)

    sizes = []
    s=sum(aa)
    for i in range(len(aa)):
        d= aa[i]/s
        sizes.append(d)

    colors =['violet','yellowgreen','gold','orange','indigo','lightcoral','lightskyblue','grey','blue']
    explode = (0,0,0.1,0, 0, 0, 0,0,0)

    plt.pie(sizes, explode=explode, labels=x, autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


    plt.title('Different ways students learned about this program.')
    pdf.savefig()
    plt.close()


    #-----------------------------------4------------------------------------


    rcParams.update({'figure.autolayout': True})
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
    plt.bar(x01,y01,color=('lightcoral','grey'))
    plt.xlabel('Students with CGPA greater than 8.0')
    plt.ylabel('Number of students ')

    plt.title('Students in the fourth year and have a CGPA greater than 8.0')
    pdf.savefig()
    plt.close()



    #-----------------------------------------5--------------------------------------------------

    rcParams.update({'figure.autolayout': True})
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

    aa = y01
    sizes = []
    s=sum(aa)
    for i in range(len(aa)):
        d= aa[i]/s
        sizes.append(d)

    colors =['yellowgreen','lightcoral']
    explode = (0, 0.1)

    plt.pie(sizes, explode=explode, labels=x01, autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


    plt.title('Digital Marketing Students with score greater than 8.')
    pdf.savefig()
    plt.close()

    #-----------------------------------------6---------------------------------
    x21=df1['Which-year are you studying in?']
    x21=list(x21)
    a2=[x21[4],x21[0],x21[6],x21[1]]

    x22 =df1['Areas of interest']
    for i in range(len(a2)):
        rcParams.update({'figure.autolayout': True})
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
        plt.bar(x,y)
        plt.xlabel('Source')
        plt.ylabel('Number of students ')
        plt.xticks(index,x,fontsize=8,rotation=90)
        plt.title(a2[i])
        pdf.savefig()
        plt.close()



    #-----------------------------------------7.1--------------------------------
    rcParams.update({'figure.autolayout': True})

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

    plt.pie(sizes, explode=explode, labels=x, autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('Number of students from different City')

    pdf.savefig()
    plt.close()

    #-----------------------------------------7.2--------------------------------
    rcParams.update({'figure.autolayout': True})
    qualitative_colors = sns.color_palette("Set3")

    college_wise_keys_org = df['College name'].value_counts().keys().tolist()
    college_wise = df['College name'].value_counts().tolist()
    college_wise_keys = [label.replace(' ', '\n') for label in college_wise_keys_org]
    df = pd.DataFrame({'college wise': college_wise})
    plt.figure(9)
    ax = df['college wise'].plot(kind='bar', figsize=(30, 13), color=qualitative_colors, fontsize=13);
    ax.set_alpha(0.8)
    ax.set_title("College wise classification of students", fontsize=18)
    ax.set_ylabel("Number of students", fontsize=18)
    ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
    ax.set_xticklabels(college_wise_keys, rotation=0, fontsize=11)
    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height() + 5, str(i.get_height()), fontsize=13, color='dimgrey')

    pdf.savefig()
    plt.close()




    #-----------------------------------------8--------------------------------

    width=0.35
    rcParams.update({'figure.autolayout': True})


    le=[]
    lie=[]

    df = pd.DataFrame(a)

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


    pd.crosstab(df1['Areas of interest'],df['Label']).plot(kind='bar')


    plt.ylabel('No. of Students')
    plt.title('Area of Interest v/s Target Variables')


    pdf.savefig()
    plt.close()





    #-----------------------------------------9--------------------------------

    width=0.35
    rcParams.update({'figure.autolayout': True})


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



    pd.crosstab(df1['Major/Area of Study'],df1['Label']).plot(kind='bar')



    plt.ylabel('No. of Students')
    plt.title('Majors v/s Target Variables')


    pdf.savefig()
    plt.close()








    #-----------------------------------------10--------------------------------
    rcParams.update({'figure.autolayout': True})

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


    pd.crosstab(df1['Which-year are you studying in?'],df1['Label']).plot(kind='bar')

    plt.ylabel('No. of Students')
    plt.title('Year of Study v/s Target Variables')

    pdf.savefig()
    plt.close()



    #-----------------------------------------11--------------------------------


    width=0.35
    rcParams.update({'figure.autolayout': True})


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
    plt.xticks(index, a2,fontsize=12)
    #plt.yticks(np.arange(0, 81, 10))
    plt.legend((p2[0], p1[0]), ('Eligible', 'Not Eligible'))
    pdf.savefig()
    plt.close()



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



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = RandomForestClassifier(n_estimators= 1)
#clf=GaussianNB()
#clf = svm.SVC()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

#print('Accuracy Score :', (accuracy_score(y_test, y_pred)*100))
print(f1_score(y_test, y_pred,average="weighted"))


