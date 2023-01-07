from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
import os
import tensorflow as tf
from joblib import load
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from dash import Dash, html, dcc

model_path= os.path.join(os.getcwd(), 'model', 'tf')
model= tf.keras.models.load_model(model_path)
sk_path=os.path.join(os.getcwd(), 'model', 'sklearn')
df=pd.read_csv(sk_path+'/diabetes.csv')

ddy=list(np.arange(0.001,0.1,0.01))
b=.06
a=(1-b*3)/4
d=[]
for i in range(4):
  l=i*(b+a)
  o=[l,l+a]
  d.append(o)
dy=[[.0,0.475],[.525, 1]]
names=['Age', 'PF','BMI', 'Insulin', 'ST', 'BP', 'Glu', 'num Preg'][::-1]
id=['age', 'pf','bmi', 'il', 'st','bp','glu','np' ][::-1]



def home(request):
    bol=False
    prob='none'
    if request.method == 'POST':

        upload = [request.POST['np'], request.POST['glu'], request.POST['bp'], request.POST['st'], request.POST['il'], request.POST['bmi'], request.POST['pf'],request.POST['age']]        
        for i, j in enumerate(upload):
            upload[i]=float(j)
        upload=tf.reshape(tf.constant(upload), (1,8)).numpy()
        scaler = load(sk_path+'/scaler.joblib') 
        upload_scaled=scaler.transform(upload)

        if request.POST['model'] == '1':
            prob = model.predict(upload_scaled)[0]
        
        if request.POST['model'] == '2':
            rf= load(sk_path+'/rf.joblib')
            prob = rf.predict_proba(upload_scaled)[0]

        if request.POST['model'] == '3':
            xgb= load(sk_path+'/XGBoost.joblib')
            prob = xgb.predict_proba(upload_scaled)[0]
        
        if request.POST['model'] == '4':
            svc= load(sk_path+'/svc.joblib')
            prob = model.predict(upload_scaled)[0]

        if request.POST['model'] == '5':
            dt= load(sk_path+'/dt.joblib')
            prob = dt.predict_proba(upload)[0]

        if request.POST['model'] == '6':
            log= load(sk_path+'/Log_Reg.joblib')
            prob = log.predict_proba(upload_scaled)[0]

        prob=prob[0]
        fig1=go.Figure().add_trace(go.Histogram(x=df['Age'], histnorm='probability', name='Age'))
        fig3=go.Figure().add_trace(go.Histogram(x=df['BMI'], histnorm='probability', name='BMI'))
        fig7=go.Figure().add_trace(go.Histogram(x=df['Glucose'], histnorm='probability', name='Glu'))
        fig5=go.Figure().add_trace(go.Histogram(x=df['SkinThickness'], histnorm='probability', name='ST'))
        fig6=go.Figure().add_trace(go.Histogram(x=df['BloodPressure'], histnorm='probability', name='BP'))
        fig8=go.Figure().add_trace(go.Histogram(x=df['Pregnancies'], histnorm='probability', name='Preg'))
        fig4=go.Figure().add_trace(go.Histogram(x=df['Insulin'], histnorm='probability', name='Insulin'))
        fig2=go.Figure().add_trace(go.Histogram(x=df['DiabetesPedigreeFunction'], histnorm='probability', name='PF'))
        data=[fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8][::-1]   
        for i, j in enumerate(data):
            #j.update_layout(autosize=False, width=500, height=500,font=dict(size=16))
            j.add_trace(go.Scatter(x=[request.POST[id[i]]]*10, y=ddy, mode='lines',name='your'+ names[i]))
            
        
        
        obj1 = plot({'data': fig1}, output_type='div')
        obj2 = plot({'data': fig2}, output_type='div')
        obj3 = plot({'data': fig3}, output_type='div')
        obj4 = plot({'data': fig4}, output_type='div')
        obj5 = plot({'data': fig5}, output_type='div')
        obj6 = plot({'data': fig6}, output_type='div')
        obj7 = plot({'data': fig7}, output_type='div')
        obj8 = plot({'data': fig8}, output_type='div')
    
        bol=True

        return render(request, 'dm_dx/upload.html', {'bol': bol, 'prob': prob , 'obj1': obj1,'obj2': obj2,'obj3': obj3,'obj4': obj5,'obj5': obj5,'obj6': obj6,'obj7': obj7,'obj8': obj8, })
    return render(request, 'dm_dx/upload.html', {'bol': bol})
