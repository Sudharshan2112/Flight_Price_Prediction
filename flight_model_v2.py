"""
=============================================================
  FLIGHT PRICE PREDICTION v3 — MERGED DATASET PIPELINE
  Datasets : Data_Train.xlsx + Data_Train_Additonal.xlsx
             Test_set.xlsx  + Test_set_Additonal.xlsx
  Model    : RF + HistGradientBoosting + ExtraTrees Ensemble
  Target   : 92%+ Cross-Validation R²
  Extra    : Seat Class Prediction
=============================================================
"""

import os
import pandas as pd
import numpy as np
import pickle, json, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestRegressor,
                               HistGradientBoostingRegressor,
                               ExtraTreesRegressor,
                               StackingRegressor,
                               RandomForestClassifier)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────
# STEP 1 — LOAD & MERGE ALL AVAILABLE DATASETS
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 1: Loading & Merging All Datasets")
print("="*65)

DATA_DIR = './data'

def safe_load(path):
    if os.path.exists(path):
        df = pd.read_excel(path)
        print(f"  OK Loaded  : {os.path.basename(path)}  -> {df.shape[0]} rows, {df.shape[1]} cols")
        return df
    else:
        print(f"  -- Skipped : {os.path.basename(path)}  (file not found)")
        return None

train_frames = []
for fname in ['Data_Train.xlsx', 'Data_Train_Additonal.xlsx']:
    df = safe_load(os.path.join(DATA_DIR, fname))
    if df is not None:
        train_frames.append(df)

test_frames = []
for fname in ['Test_set.xlsx', 'Test_set_Additonal.xlsx']:
    df = safe_load(os.path.join(DATA_DIR, fname))
    if df is not None:
        test_frames.append(df)

if not train_frames:
    raise FileNotFoundError("No training data found. Place Data_Train.xlsx inside ./data/")

train = pd.concat(train_frames, ignore_index=True)
test  = pd.concat(test_frames,  ignore_index=True) if test_frames else pd.DataFrame()

train.dropna(subset=['Price','Total_Stops','Route'], inplace=True)
train.drop_duplicates(inplace=True)
train.reset_index(drop=True, inplace=True)

print(f"\n  Combined Dataset:")
print(f"  Training samples : {train.shape[0]:,}")
print(f"  Test samples     : {test.shape[0]:,}")
print(f"  Price Min/Max/Mean: Rs{train['Price'].min():,.0f} / Rs{train['Price'].max():,.0f} / Rs{train['Price'].mean():,.0f}")

# ─────────────────────────────────────────────────────────
# STEP 2 — SEAT CLASS LABELING
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 2: Deriving Seat Class Labels")
print("="*65)

def derive_seat_class(row):
    airline = str(row['Airline']).strip()
    info    = str(row['Additional_Info']).strip().lower()
    price   = row.get('Price', 0) or 0
    if 'business' in airline.lower(): return 'Business'
    if 'premium economy' in airline.lower(): return 'Premium Economy'
    if 'business class' in info: return 'Business'
    if 'premium' in info: return 'Premium Economy'
    if price >= 20000: return 'Business'
    if price >= 12000: return 'Premium Economy'
    return 'Economy'

train['Seat_Class'] = train.apply(derive_seat_class, axis=1)
for cls, cnt in train['Seat_Class'].value_counts().items():
    pct = cnt/len(train)*100
    print(f"    {cls:<18} : {cnt:>5} ({pct:5.1f}%)")

# ─────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING (28 features)
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 3: Feature Engineering (28 Features)")
print("="*65)

TIER_MAP = {
    'IndiGo':1,'SpiceJet':1,'GoAir':1,'Air Asia':1,'Trujet':1,
    'Jet Airways':2,'Air India':2,'Multiple carriers':2,
    'Vistara':3,
    'Vistara Premium economy':4,'Multiple carriers Premium economy':4,
    'Jet Airways Business':5
}
INFO_MAP = {
    'no info':0,'no Info':0,'in-flight meal not included':1,
    'no check-in baggage included':2,'1 short layover':3,
    '1 long layover':4,'2 long layover':5,'change airports':6,
    'red-eye flight':7,'business class':8
}
STOPS_MAP = {'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}

def parse_duration(d):
    d=str(d).strip(); h,m=0,0
    if 'h' in d: h=int(d.split('h')[0].strip())
    if 'm' in d:
        part=d.split('h')[-1] if 'h' in d else d
        m_str=part.replace('m','').strip()
        m=int(m_str) if m_str else 0
    return h*60+m

def get_time_slot(hour):
    if 4<=hour<8: return 0
    if 8<=hour<12: return 1
    if 12<=hour<17: return 2
    if 17<=hour<21: return 3
    return 4

def feature_engineer(df, is_train=True):
    df=df.copy()
    dt=pd.to_datetime(df['Date_of_Journey'],dayfirst=True)
    df['Journey_Day']=dt.dt.day
    df['Journey_Month']=dt.dt.month
    df['Journey_Weekday']=dt.dt.weekday
    df['Journey_Quarter']=dt.dt.quarter
    df['Is_Weekend']=(dt.dt.weekday>=5).astype(int)
    df['Is_Month_Start']=(dt.dt.day<=5).astype(int)
    df['Is_Month_End']=(dt.dt.day>=25).astype(int)
    df['Is_Peak_Month']=dt.dt.month.isin([5,6,11,12]).astype(int)
    df['Is_Holiday_Week']=dt.dt.month.isin([1,8,10]).astype(int)
    df['Dep_Hour']=df['Dep_Time'].str.split(':').str[0].astype(int)
    df['Dep_Min']=df['Dep_Time'].str.split(':').str[1].astype(int)
    df['Dep_Slot']=df['Dep_Hour'].apply(get_time_slot)
    df['Arr_Hour']=df['Arrival_Time'].str.split(':').str[0].astype(int)
    df['Arr_Min']=(df['Arrival_Time'].str.split(':').str[1].str.split(' ').str[0].astype(int))
    df['Duration_mins']=df['Duration'].apply(parse_duration)
    df['Duration_hrs']=(df['Duration_mins']/60.0).round(3)
    df['Is_Long_Haul']=(df['Duration_mins']>=180).astype(int)
    df['Stops']=df['Total_Stops'].str.strip().map(STOPS_MAP).fillna(0).astype(int)
    df['Is_Direct']=(df['Stops']==0).astype(int)
    df['Num_Routes']=df['Route'].str.count('→')+1
    df['Airline_Tier']=df['Airline'].map(TIER_MAP).fillna(2).astype(int)
    df['Info_Code']=(df['Additional_Info'].str.strip().str.lower().map(INFO_MAP).fillna(0).astype(int))
    df['Speed_proxy']=(df['Duration_mins']/(df['Stops']+1)).round(2)
    df['Tier_x_Duration']=(df['Airline_Tier']*df['Duration_mins']).round(2)
    drop_cols=['Date_of_Journey','Dep_Time','Arrival_Time','Duration','Total_Stops','Route','Additional_Info']
    if is_train:
        df.drop(columns=[c for c in drop_cols if c in df.columns],inplace=True)
    else:
        df.drop(columns=drop_cols,inplace=True,errors='ignore')
    return df

train_fe = feature_engineer(train, is_train=True)
test_fe  = feature_engineer(test,  is_train=False) if not test.empty else pd.DataFrame()

feat_list=[c for c in train_fe.columns if c not in ['Price','Seat_Class']]
print(f"  Total features : {len(feat_list)}")
for f in feat_list: print(f"    - {f}")

# ─────────────────────────────────────────────────────────
# STEP 4 — LABEL ENCODING
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 4: Label Encoding")
print("="*65)

cat_cols=['Airline','Source','Destination']
encoders={}
for col in cat_cols:
    le=LabelEncoder()
    if not test_fe.empty and col in test_fe.columns:
        combined=pd.concat([train_fe[col],test_fe[col]],axis=0)
    else:
        combined=train_fe[col]
    le.fit(combined)
    train_fe[col]=le.transform(train_fe[col])
    if not test_fe.empty and col in test_fe.columns:
        test_fe[col]=le.transform(test_fe[col])
    encoders[col]=le
    print(f"  {col:<15}: {len(le.classes_)} classes")

seat_le=LabelEncoder()
train_fe['Seat_Class_Enc']=seat_le.fit_transform(train_fe['Seat_Class'])
encoders['Seat_Class']=seat_le
print(f"  Seat classes    : {list(seat_le.classes_)}")

# ─────────────────────────────────────────────────────────
# STEP 5 — PREPARE FEATURES
# ─────────────────────────────────────────────────────────
feature_cols=[c for c in train_fe.columns if c not in ['Price','Seat_Class','Seat_Class_Enc']]
X=train_fe[feature_cols]; y=train_fe['Price']; y_cls=train_fe['Seat_Class_Enc']
if not test_fe.empty:
    X_test=test_fe[[c for c in feature_cols if c in test_fe.columns]]
    for col in feature_cols:
        if col not in X_test.columns: X_test[col]=0
    X_test=X_test[feature_cols]
else:
    X_test=pd.DataFrame()
print(f"\n  X shape : {X.shape}")

# ─────────────────────────────────────────────────────────
# STEP 6 — TRAIN MODELS
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 6: Training Models")
print("="*65)

print("\n  [A] Random Forest (400 trees)")
rf=RandomForestRegressor(n_estimators=400,max_depth=None,min_samples_split=2,
    min_samples_leaf=1,max_features='sqrt',bootstrap=True,oob_score=True,random_state=42,n_jobs=-1)
rf.fit(X,y)
print(f"      OOB R2: {rf.oob_score_:.4f}  Train R2: {r2_score(y,rf.predict(X)):.4f}  MAE: Rs{mean_absolute_error(y,rf.predict(X)):,.0f}")

print("\n  [B] HistGradientBoosting (600 iters)")
hgb=HistGradientBoostingRegressor(max_iter=600,max_depth=9,learning_rate=0.04,
    min_samples_leaf=4,l2_regularization=0.08,max_bins=255,random_state=42,early_stopping=False)
hgb.fit(X,y)
print(f"      Train R2: {r2_score(y,hgb.predict(X)):.4f}  MAE: Rs{mean_absolute_error(y,hgb.predict(X)):,.0f}")

print("\n  [C] Extra Trees (300 trees)")
et=ExtraTreesRegressor(n_estimators=300,max_depth=None,min_samples_leaf=1,random_state=42,n_jobs=-1)
et.fit(X,y)
print(f"      Train R2: {r2_score(y,et.predict(X)):.4f}  MAE: Rs{mean_absolute_error(y,et.predict(X)):,.0f}")

# ─────────────────────────────────────────────────────────
# STEP 7 — STACKED ENSEMBLE
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 7: Stacked Ensemble")
print("="*65)
stacked=StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=250,min_samples_leaf=1,random_state=42,n_jobs=-1)),
        ('hgb',HistGradientBoostingRegressor(max_iter=500,max_depth=8,learning_rate=0.05,random_state=42)),
        ('et', ExtraTreesRegressor(n_estimators=200,random_state=42,n_jobs=-1))
    ],
    final_estimator=Ridge(alpha=10.0),cv=5,n_jobs=-1)
stacked.fit(X,y)
print(f"  Stack Train R2 : {r2_score(y,stacked.predict(X)):.4f}")
print(f"  Stack Train MAE: Rs{mean_absolute_error(y,stacked.predict(X)):,.0f}")

# ─────────────────────────────────────────────────────────
# STEP 8 — WEIGHTED ENSEMBLE
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 8: Weighted Blend RF25 + HGB45 + ET30")
print("="*65)
def ensemble_predict(X_in):
    return 0.25*rf.predict(X_in)+0.45*hgb.predict(X_in)+0.30*et.predict(X_in)

ens_train=ensemble_predict(X)
ens_r2=r2_score(y,ens_train)
ens_mae=mean_absolute_error(y,ens_train)
ens_rmse=np.sqrt(mean_squared_error(y,ens_train))
print(f"  Weighted Train R2 : {ens_r2:.4f}")
print(f"  Weighted Train MAE: Rs{ens_mae:,.0f}")

# ─────────────────────────────────────────────────────────
# STEP 9 — 5-FOLD CV
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 9: 5-Fold Cross Validation")
print("="*65)
kf=KFold(n_splits=5,shuffle=True,random_state=42)
X_arr,y_arr=X.values,y.values
cv_scores,cv_maes=[],[]
for fold,(tr_idx,val_idx) in enumerate(kf.split(X_arr),1):
    Xtr,Xval=X_arr[tr_idx],X_arr[val_idx]
    ytr,yval=y_arr[tr_idx],y_arr[val_idx]
    rf_f=RandomForestRegressor(n_estimators=250,min_samples_leaf=1,random_state=42,n_jobs=-1)
    hgb_f=HistGradientBoostingRegressor(max_iter=500,max_depth=8,learning_rate=0.05,random_state=42)
    et_f=ExtraTreesRegressor(n_estimators=200,random_state=42,n_jobs=-1)
    rf_f.fit(Xtr,ytr); hgb_f.fit(Xtr,ytr); et_f.fit(Xtr,ytr)
    y_pred=0.25*rf_f.predict(Xval)+0.45*hgb_f.predict(Xval)+0.30*et_f.predict(Xval)
    fold_r2=r2_score(yval,y_pred); fold_mae=mean_absolute_error(yval,y_pred)
    cv_scores.append(fold_r2); cv_maes.append(fold_mae)
    print(f"  Fold {fold}: R2 = {fold_r2:.4f}  |  MAE = Rs{fold_mae:,.0f}")

cv_mean=np.mean(cv_scores); cv_std=np.std(cv_scores); cv_mae_mean=np.mean(cv_maes)

# ─────────────────────────────────────────────────────────
# STEP 10 — RESULTS
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 10: Final Performance")
print("="*65)
print(f"  Datasets merged  : {len(train_frames)}")
print(f"  Training samples : {len(X):,}")
print(f"  Features         : {len(X.columns)}")
print(f"  CV R2 Score      : {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"  Accuracy         : {cv_mean*100:.2f}%")
print(f"  CV MAE           : Rs{cv_mae_mean:,.0f}")
print(f"  Train R2         : {ens_r2:.4f}")

# ─────────────────────────────────────────────────────────
# STEP 11 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 11: Feature Importances")
print("="*65)
feat_imp=pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
for feat,imp in feat_imp.items():
    bar='#'*int(imp*60)
    print(f"  {feat:<25} {imp*100:>7.2f}%  {bar}")
feat_imp_dict={k:round(float(v)*100,2) for k,v in feat_imp.items()}

# ─────────────────────────────────────────────────────────
# STEP 12 — SEAT CLASS CLASSIFIER
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 12: Seat Class Classifier")
print("="*65)
clf=RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=1,random_state=42,n_jobs=-1)
clf.fit(X,y_cls)
cls_pred=clf.predict(X)
print(f"  Train Accuracy: {accuracy_score(y_cls,cls_pred)*100:.2f}%")
print(classification_report(y_cls,cls_pred,target_names=seat_le.classes_))
cls_cv=cross_val_score(RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1),
    X,y_cls,cv=5,scoring='accuracy')
print(f"  5-Fold CV Accuracy: {cls_cv.mean()*100:.2f}% +/- {cls_cv.std()*100:.2f}%")

# ─────────────────────────────────────────────────────────
# STEP 13 — TEST PREDICTIONS
# ─────────────────────────────────────────────────────────
if not X_test.empty:
    print("\n" + "="*65)
    print("  STEP 13: Test Set Predictions")
    print("="*65)
    test['Predicted_Price']=np.round(ensemble_predict(X_test),0).astype(int)
    test['Predicted_Class']=seat_le.inverse_transform(clf.predict(X_test))
    for i,(_,row) in enumerate(test.head(12).iterrows()):
        route=f"{row['Source']} -> {row['Destination']}"
        print(f"  {i+1:<4} {row['Airline']:<30} {route:<25} {row['Predicted_Class']:<18} Rs{row['Predicted_Price']:>10,}")

# ─────────────────────────────────────────────────────────
# STEP 14 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 14: Saving Model Artifacts")
print("="*65)
os.makedirs('./models',exist_ok=True)
with open('./models/rf_model.pkl','wb') as f: pickle.dump(rf,f)
with open('./models/hgb_model.pkl','wb') as f: pickle.dump(hgb,f)
with open('./models/et_model.pkl','wb') as f: pickle.dump(et,f)
with open('./models/clf_model.pkl','wb') as f: pickle.dump(clf,f)
with open('./models/encoders.pkl','wb') as f: pickle.dump(encoders,f)

model_info={
    'version':'3.0','datasets_used':len(train_frames),
    'models':['RandomForest','HistGradientBoosting','ExtraTrees'],
    'blend':'RF(25%) + HGB(45%) + ET(30%)',
    'cv_r2':round(cv_mean,4),'cv_std':round(cv_std,4),
    'accuracy_pct':round(cv_mean*100,2),'train_r2':round(ens_r2,4),
    'mae':round(float(ens_mae),2),'cv_mae':round(float(cv_mae_mean),2),
    'rmse':round(float(ens_rmse),2),'n_train':int(len(X)),
    'n_features':int(len(X.columns)),'features':list(X.columns),
    'feature_importances':feat_imp_dict,'seat_classes':list(seat_le.classes_),
    'seat_class_cv_acc':round(float(cls_cv.mean()*100),2),
    'airlines':list(encoders['Airline'].classes_),
    'sources':list(encoders['Source'].classes_),
    'destinations':list(encoders['Destination'].classes_),
    'airline_tiers':TIER_MAP,
    'price_stats':{
        'min':int(train['Price'].min()),'max':int(train['Price'].max()),
        'mean':int(train['Price'].mean()),'median':int(train['Price'].median()),
        'q25':int(train['Price'].quantile(0.25)),'q75':int(train['Price'].quantile(0.75))
    }
}
with open('./models/model_info.json','w') as f: json.dump(model_info,f,indent=2)
if not test.empty and 'Predicted_Price' in test.columns:
    test[['Airline','Date_of_Journey','Source','Destination','Dep_Time','Arrival_Time',
          'Duration','Total_Stops','Predicted_Price','Predicted_Class']].to_csv(
        './models/flight_predictions_v3.csv',index=False)

print(f"  OK rf_model.pkl, hgb_model.pkl, et_model.pkl")
print(f"  OK clf_model.pkl  (seat class classifier)")
print(f"  OK encoders.pkl, model_info.json")
print(f"\n{'='*65}")
print(f"  PIPELINE COMPLETE - v3 (Merged Datasets)")
print(f"  Training Samples  : {len(X):,}")
print(f"  Final CV Accuracy : {cv_mean*100:.2f}%")
print(f"  Seat Class CV Acc : {cls_cv.mean()*100:.2f}%")
print(f"{'='*65}\n")
