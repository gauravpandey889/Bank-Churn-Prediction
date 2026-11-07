from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
raw=pd.read_csv(ROOT/'European_Bank.csv')
def engineer(df):
    x=df.drop(columns=['Exited','CustomerId','Surname','Year'],errors='ignore').copy()
    x['BalanceSalaryRatio']=x['Balance']/(x['EstimatedSalary'].clip(lower=1))
    x['EngagementProducts']=x['IsActiveMember']*x['NumOfProducts']
    x['AgeTenure']=x['Age']*x['Tenure']
    x['ProductsPerTenure']=x['NumOfProducts']/(x['Tenure']+1)
    return x
X=engineer(raw); y=raw.Exited.astype(int)
cat=['Geography','Gender']; nums=[c for c in X if c not in cat]
pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Logistic Regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),'Decision Tree':DecisionTreeClassifier(max_depth=6,min_samples_leaf=30,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=180,min_samples_leaf=5,class_weight='balanced_subsample',random_state=42,n_jobs=-1),'Gradient Boosting':HistGradientBoostingClassifier(max_iter=160,learning_rate=.06,l2_regularization=2,random_state=42)}
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
rows=[]; trained={}; preds={}
for name,est in models.items():
    pipe=Pipeline([('preprocess',pre),('model',est)]);pipe.fit(Xtrain,ytrain);p=pipe.predict_proba(Xtest)[:,1];preds[name]=p;trained[name]=pipe
    label=(p>=.5).astype(int)
    rows.append({'model':name,'accuracy':accuracy_score(ytest,label),'precision':precision_score(ytest,label,zero_division=0),'recall':recall_score(ytest,label),'f1':f1_score(ytest,label),'roc_auc':roc_auc_score(ytest,p)})
results=pd.DataFrame(rows).sort_values('roc_auc',ascending=False);results.to_csv(ROOT/'outputs/model_comparison.csv',index=False)
best=results.iloc[0]['model']; pipe=trained[best]; p=preds[best]
joblib.dump(pipe,ROOT/'models/churn_model.joblib')
thresholds=np.linspace(.05,.95,181)
# Choose a threshold on a separate validation split from training only (avoid tuning on test).
Xt,Xv,yt,yv=train_test_split(Xtrain,ytrain,test_size=.25,stratify=ytrain,random_state=123)
valpipe=Pipeline([('preprocess',pre),('model',models[best])]);valpipe.fit(Xt,yt);vp=valpipe.predict_proba(Xv)[:,1]
threshold=float(max(thresholds,key=lambda t:f1_score(yv,vp>=t)))
scored=raw[['CustomerId','Geography','Age','IsActiveMember','NumOfProducts','Exited']].copy();scored['churn_probability']=pipe.predict_proba(X)[:,1];scored['risk_band']=pd.cut(scored.churn_probability,[-.001,.3,.6,1.001],labels=['Low','Medium','High']);scored.to_csv(ROOT/'outputs/customer_risk_scores.csv',index=False)
importance=permutation_importance(pipe,Xtest,ytest,n_repeats=3,random_state=42,scoring='roc_auc',n_jobs=1)
imp=pd.DataFrame({'feature':X.columns,'importance':importance.importances_mean}).sort_values('importance',ascending=False);imp.to_csv(ROOT/'outputs/feature_importance.csv',index=False)
fig,ax=plt.subplots(figsize=(9,5));ax.barh(imp.feature.head(10)[::-1],imp.importance.head(10)[::-1]);ax.set_xlabel('ROC-AUC decrease after permutation');ax.set_title('Top predictive features');fig.tight_layout();fig.savefig(ROOT/'outputs/feature_importance.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
for name,prob in preds.items():
    fpr,tpr,_=roc_curve(ytest,prob);ax.plot(fpr,tpr,label=f'{name} ({roc_auc_score(ytest,prob):.3f})')
ax.plot([0,1],[0,1],'--',color='gray');ax.legend(fontsize=8);ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Test ROC curves');fig.tight_layout();fig.savefig(ROOT/'outputs/roc_curves.png',dpi=150);plt.close(fig)
meta={'best_model':best,'threshold':threshold,'test_roc_auc':float(roc_auc_score(ytest,p)),'test_f1_at_threshold':float(f1_score(ytest,p>=threshold)),'test_confusion_matrix':confusion_matrix(ytest,p>=threshold).tolist(),'churn_rate':float(y.mean()),'rows':len(raw),'train_rows':len(Xtrain),'test_rows':len(Xtest),'features':list(X.columns),'year_unique':sorted(raw.Year.unique().tolist())}
(ROOT/'models/metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2));print(results.to_string(index=False))
