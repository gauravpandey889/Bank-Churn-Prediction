from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from train_utils import engineer
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Bank Churn Intelligence',page_icon='🏦',layout='wide')
@st.cache_resource
def load():return joblib.load(ROOT/'models/churn_model.joblib')
model=load(); meta=json.loads((ROOT/'models/metadata.json').read_text()); df=pd.read_csv(ROOT/'outputs/customer_risk_scores.csv'); imp=pd.read_csv(ROOT/'outputs/feature_importance.csv');comparison=pd.read_csv(ROOT/'outputs/model_comparison.csv')
st.title('🏦 Bank Customer Churn Intelligence')
st.caption('Predictive risk scoring | Internship project | Risk is an estimate, not a certainty')
tabs=st.tabs(['Overview','Risk Calculator','What-if Simulator','Model Explainability','Model Evaluation'])
with tabs[0]:
    a,b,c,d=st.columns(4);a.metric('Customers',f'{len(df):,}');b.metric('Observed churn',f'{df.Exited.mean():.1%}');c.metric('High risk customers',f'{(df.risk_band=="High").sum():,}');d.metric('Held-out ROC-AUC',f'{meta["test_roc_auc"]:.3f}')
    st.plotly_chart(px.histogram(df,x='churn_probability',color='risk_band',nbins=30,title='Predicted churn probability distribution'),use_container_width=True)
    st.dataframe(df.head(100),use_container_width=True)
    st.download_button('Download customer risk scores',df.to_csv(index=False),'customer_risk_scores.csv')
with tabs[1]:
    st.subheader('Customer churn risk calculator')
    c1,c2,c3=st.columns(3)
    with c1: credit=st.number_input('Credit score',300,900,650);geo=st.selectbox('Geography',['France','Spain','Germany']);gender=st.selectbox('Gender',['Female','Male']);age=st.slider('Age',18,95,42)
    with c2: tenure=st.slider('Tenure',0,10,5);balance=st.number_input('Balance',min_value=0.,value=75000.,step=1000.);salary=st.number_input('Estimated annual salary',min_value=0.,value=100000.,step=1000.)
    with c3: products=st.slider('Number of products',1,4,2);card=st.selectbox('Has credit card',[1,0]);active=st.selectbox('Active member',[1,0])
    customer={'CreditScore':credit,'Geography':geo,'Gender':gender,'Age':age,'Tenure':tenure,'Balance':balance,'NumOfProducts':products,'HasCrCard':card,'IsActiveMember':active,'EstimatedSalary':salary}
    probability=float(model.predict_proba(engineer(pd.DataFrame([customer])))[0,1]);band='High' if probability>=.6 else 'Medium' if probability>=.3 else 'Low'
    st.metric('Predicted churn probability',f'{probability:.1%}');st.progress(probability);st.write(f'**Risk band:** {band} | **Model flag (validated threshold {meta["threshold"]:.2f}):** {"Likely churn" if probability>=meta["threshold"] else "Lower predicted risk"}')
with tabs[2]:
    st.subheader('What-if: adjust engagement and product usage')
    newactive=st.selectbox('Scenario: active member',[0,1],index=1);newproducts=st.slider('Scenario: number of products',1,4,2,key='whatproducts')
    original={'CreditScore':650,'Geography':'France','Gender':'Female','Age':42,'Tenure':5,'Balance':75000,'NumOfProducts':1,'HasCrCard':1,'IsActiveMember':0,'EstimatedSalary':100000}
    scenario={**original,'IsActiveMember':newactive,'NumOfProducts':newproducts}
    base=float(model.predict_proba(engineer(pd.DataFrame([original])))[0,1]);updated=float(model.predict_proba(engineer(pd.DataFrame([scenario])))[0,1]);st.metric('Scenario probability',f'{updated:.1%}',delta=f'{updated-base:+.1%} vs baseline',delta_color='inverse');st.caption('Associational model: changing an input does not prove a causal effect.')
with tabs[3]:
    st.plotly_chart(px.bar(imp.head(12).sort_values('importance'),x='importance',y='feature',orientation='h',title='Permutation importance on held-out data'),use_container_width=True)
    st.caption('Importance is predictive, not causal. Demographic fields require fairness review before deployment.')
with tabs[4]:
    st.dataframe(comparison.style.format({x:'{:.3f}' for x in ['accuracy','precision','recall','f1','roc_auc']}),use_container_width=True)
    st.image(str(ROOT/'outputs/roc_curves.png'));st.write('Best model:',meta['best_model']);st.write('Held-out confusion matrix at validation-selected threshold:',meta['test_confusion_matrix'])
