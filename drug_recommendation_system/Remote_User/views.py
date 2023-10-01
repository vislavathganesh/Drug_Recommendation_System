from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,drug_recommendation_Type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Drug_Recommendation_Type(request):
    if request.method == "POST":
        review = request.POST.get('keyword')
        if request.method == "POST":
            review = request.POST.get('keyword')
            dname = request.POST.get('dname')

        df = pd.read_csv('Drugs_Review_Datasets.csv')
        df
        df.columns
        df.rename(columns={'rating': 'Rating', 'review': 'Review'}, inplace=True)

        def apply_recommend(Rating):
            if (Rating <= 7):
                return 0  # Neagtive
            else:
                return 1  # Positive

        df['recommend'] = df['Rating'].apply(apply_recommend)
        df.drop(['Rating'], axis=1, inplace=True)
        recommend = df['recommend'].value_counts()

        # df.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Time','Summary'], axis=1, inplace=True)

        def preprocess_text(text):
            '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
            and remove words containing numbers.'''
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = re.sub('"@', '', text)
            text = re.sub('@', '', text)
            text = re.sub('https: //', '', text)
            text = re.sub('\n\n', '', text)
            text = re.sub('""', '', text)

            return text

        df['processed_content'] = df['Review'].apply(lambda x: preprocess_text(x))

        cv = CountVectorizer()
        X = df['processed_content']
        y = df['recommend']

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        models.append(('svm', lin_clf))


        #classifier = VotingClassifier(models)
        #classifier.fit(X_train, y_train)
        #y_pred = classifier.predict(X_test)


        review_data = [review]
        vector1 = cv.transform(review_data).toarray()
        predict_text = lin_clf.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Negative'
        else:
            val = 'Positive'

        print(val)
        print(pred1)

        drug_recommendation_Type.objects.create(Drug_Name=dname,Drug_Review=review, Prediction=val)

        return render(request, 'RUser/Predict_Drug_Recommendation_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Drug_Recommendation_Type.html')



