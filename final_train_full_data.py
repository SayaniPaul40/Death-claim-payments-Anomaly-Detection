# necessary imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,confusion_matrix,precision_score,f1_score,recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings('ignore')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import expit
import joblib
import missingno as msno

plt.style.use('ggplot')
df = pd.read_excel(r"C:\Users\acer\Desktop\WORK\MasterData\Final_Code\Death_Claim_deliverable\Final_input\input.xls",header=1)

df_cleaned= df.dropna(axis=1, how='all')

df_cleaned = df_cleaned.iloc[:, :-1]

# missing values count column wise
nan_sum=df_cleaned.isna().sum()

missing_percentage=(df_cleaned.isnull().sum()/len(df_cleaned))*100

#check dtypes
df_types = df_cleaned.dtypes

selected_cols = df_cleaned[[
'due_yyyy',
'due_mm',
'fup_mas_yyyy',
'fup_mas_mm',
'cm_mode',
'claim_status',
'date_of_death',
'accounted_on',
'policy_term',
'ord_prem_flag',
'life_death_ind',
'docket_availability',
'premium',
'paid_date',
'commencement_date',
'total_payment',
'total_deduction','claim_amount', 'vested_bonus', 'interim_bonus','Anomaly Flag'
]]

#status_change_date data validation and typecasting

selected_cols['paid_date']= pd.to_datetime(selected_cols['paid_date'], errors='coerce')
selected_cols['accounted_on']=pd.to_datetime(selected_cols['accounted_on'], errors='coerce')

#DATA QUALITY CHECK....................................................................................................

# data quality checks for commencement_date column
def check_doc(commencement_date):
    if pd.isnull(commencement_date):  # Check for NULL values
        return ' '
    else:
        try:
            doc_date = pd.to_datetime(commencement_date)  # Convert string to datetime object
            if doc_date.year == 1900 and doc_date.month == 1 and doc_date.day == 1:  # Check for '01-01-1900' dates
                return ' '
            elif doc_date.year == 0 or doc_date.month == 0 or doc_date.day == 0:  # Check for '00-00-000' dates
                return ""
            else:
                #return doc_date.strftime('%d-%m-%Y')  # Convert datetime object to string in 'dd-mm-yyyy' format
                return doc_date
        except ValueError:
            return ' '

# Apply data quality check function to DOC column
selected_cols['commencement_date'] = selected_cols['commencement_date'].apply(check_doc)
#selected_cols['commencement_date'] =selected_cols['commencement_date'].dropna(inplace=True)
#..........................................................................................................................

# data quality checks for commencement_date column
import re
pattern=r'\b\d{2}/\d{2}/d{4}\b'
def extract_date(cell):
    matches = re.findall(pattern,str(cell))
    return' '.join(matches) if matches else cell

def check_Accounted(Accounted_On):
    if pd.isnull(Accounted_On):  # Check for NULL values
        return ' '
    else:
        try:
            doc_date1 = pd.to_datetime(Accounted_On)  # Convert string to datetime object
            
            if doc_date1.year == 0 or doc_date1.month == 0 or doc_date1.day == 0:  # Check for '00-00-000' dates
                return ' '
            else:
                return doc_date1.strftime('%m-%d-%Y')  # Convert datetime object to string in 'dd-mm-yyyy' format
                #return doc_date1
        except ValueError:
             return ' '
# Apply data quality check function to DOC column
selected_cols['accounted_on'] = selected_cols['accounted_on'].apply(check_Accounted)
#selected_cols['Accounted On'] =selected_cols['Accounted On'].dropna(inplace=True)  
selected_cols['accounted_on']= selected_cols['accounted_on'].apply(extract_date)  
#selected_cols['Accounted On'] =selected_cols['Accounted On'].replace({'0000-00-00':pd.NaT},regex=True)    
selected_cols['accounted_on'] =pd.to_datetime(selected_cols['accounted_on'],errors='coerce')
selected_cols['accounted_on']=pd.to_datetime(selected_cols['accounted_on']).dt.date

#..........................................................................................................................

# Data quality checks for Policy Term column
def check_policy_term(policy_term):
    if pd.isnull(policy_term):  # Check for NULL values
        return ' '
    else:
        if policy_term < 1 or policy_term > 100:  # Check if policy term is out of range
            return ' '
        else:
            return policy_term

# Apply data quality check function to Policy Term column
selected_cols['policy_term'] = selected_cols['policy_term'].apply(check_policy_term)

#..............................................................................................................................

def format_master_fup(fup_mas_yyyy):
    if fup_mas_yyyy <= 0:
        return ' '
    else:
        #master_fup_str = str(master_fup)
        
        # year_pattern= r'^\d{4}$'
        # selected_cols['fup_mas_yyyy']= selected_cols['fup_mas_yyyy'].match(year_pattern)
        return fup_mas_yyyy

selected_cols['fup_mas_yyyy'] = selected_cols['fup_mas_yyyy'].apply(format_master_fup)

# Function to preprocess Master FUP month
def format_master_fup_mm(month):
    if month < 1 or month > 12:
        return ' '
    else:
        
        return month

selected_cols['fup_mas_mm'] = selected_cols['fup_mas_mm'].apply(format_master_fup_mm)
#..............................................................................................................................
#DROPING BLACK WHITE space
selected_cols['fup_mas_yyyy'].replace(r'^\s*$',pd.NaT, regex=True, inplace=True)
selected_cols['fup_mas_mm'].replace(r'^\s*$',pd.NaT, regex=True, inplace=True)
selected_cols['fup_mas_yyyy']=selected_cols['fup_mas_yyyy'].dropna()
selected_cols['fup_mas_mm']=selected_cols['fup_mas_mm'].dropna()
# Function to preprocess
def preprocess_numeric(value):
    if value < 0 or value> 999999999:  # Check for negative values (assuming negative values are errors)
        #print('numeric_columns Data Error')  # Replace negative values with None for data error
        return ' ' # Replace invalid dates with None for data error
    else:
        return value
        

numeric_columns = ['claim_amount', 'vested_bonus', 'interim_bonus']
for column in numeric_columns:
    selected_cols[column] = selected_cols[column].apply(preprocess_numeric)

#..............................................................................................................................
#'premium' column

def preprocess_premium(premium):
    if premium < 10.00 or premium> 999999999.99:  # Check for negative values (assuming negative values are errors)
        print('premium_column Data Error')  # Replace negative values with None for data error
        return ' ' # Replace invalid dates with None for data error
    else:
        return premium

selected_cols['premium'] = selected_cols['premium'].apply(preprocess_premium)

#..............................................................................................................................
# Function to preprocess Mode
def preprocess_mode(cm_mode):
    if cm_mode not in [0, 1, 3, 6, 12]:  # Check for invalid mode values
        return ' '  # Replace invalid mode values with None for data error
    else:
        return cm_mode

selected_cols['cm_mode'] = selected_cols['cm_mode'].apply(preprocess_mode)

#numerical corelation
num_df = selected_cols.select_dtypes(include = ['int64','float64','int32'])
print(num_df.columns)

# Scaling the numeric values in the dataset
num_df_cols=num_df.columns

#create month and year cols from date cols

selected_cols['date_of_death_month']=pd.to_datetime(selected_cols['date_of_death']).dt.month
selected_cols['date_of_death_year']=pd.to_datetime(selected_cols['date_of_death']).dt.year

selected_cols['paid_date_month']=pd.to_datetime(selected_cols['paid_date']).dt.month
selected_cols['paid_date_year']=pd.to_datetime(selected_cols['paid_date']).dt.year

selected_cols['commencement_date_month']=pd.to_datetime(selected_cols['commencement_date']).dt.month
selected_cols['commencement_date_year']=pd.to_datetime(selected_cols['commencement_date']).dt.year

selected_cols['accounted_on_month']=pd.to_datetime(selected_cols['accounted_on']).dt.month
selected_cols['accounted_on_year']=pd.to_datetime(selected_cols['accounted_on']).dt.year

#droping date column after spliting into month year
selected_cols=selected_cols.drop(['date_of_death','paid_date','commencement_date','accounted_on'], axis=1)
#selected_cols=pd.DataFrame(selected_cols)
selected_cols1 =selected_cols.dropna()



#cahnge datatype from object to int
object_columns =['policy_term','fup_mas_yyyy','fup_mas_mm']
def convert_to_nonobject(selected_cols1, object_columns):
    for column in object_columns:
            selected_cols1[column]=selected_cols1[column].astype('int64')
    return selected_cols1
selected_cols1=convert_to_nonobject(selected_cols1,object_columns)


object_columns1 =['premium']
def convert_to_nonobject1(selected_cols1, object_columns1):
    for column in object_columns1:
            selected_cols1[column]=selected_cols1[column].astype('float64')
    return selected_cols1
selected_cols1=convert_to_nonobject1(selected_cols1,object_columns1)
#....................................................................................................................

# extracting categorical columns
def convert_to_object(selected_cols1, non_object_columns):
    for column in non_object_columns:
        selected_cols1[column]=selected_cols1[column].astype(str)
    return selected_cols1
non_object_columns =['life_death_ind','ord_prem_flag','cm_mode']
selected_cols1=convert_to_object(selected_cols1,non_object_columns)

#selected_cols1.to_csv(r"C:\\Users\acer\Desktop\WORK\MasterData\Latest_Dataset\output\before_encoding.csv", header=True)


catagorical_columns=selected_cols1.select_dtypes(include=['object']).columns

#.................................................................................................
#one hot encoding
cat_one_hot_encoded = pd.get_dummies(selected_cols1,columns=catagorical_columns).astype(int)

selected_cols1_encoded1=cat_one_hot_encoded.copy()

df_encoded=selected_cols1_encoded1.copy()

#....................................................................................................

X = df_encoded.drop(columns=['Anomaly Flag'])

Y = df_encoded['Anomaly Flag']

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#ridge classifier
ridge_classifier= RidgeClassifier(alpha=1.0,max_iter=1000,solver="auto",tol=1e-3)

#for 80% train data fitting
ridge_classifier.fit(X,Y)

Y_pred= ridge_classifier.predict(X)
# Y_pred_train=ridge_classifier.predict(X_train)

#*******************************************Full data**************************************************************
acc_train=accuracy_score(Y,Y_pred)
print("accuracy:",acc_train)

#for train precision recall
precision_score_train=precision_score(Y,Y_pred)
print("precision score",precision_score_train)
recall_score_train=recall_score(Y,Y_pred)
print("recall score",recall_score_train)

#confusion matrix train
conf_matrix1= confusion_matrix(Y,Y_pred)
sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("confusion matrix")
plt.show()

#auc chcek train
auc=roc_auc_score(Y,Y_pred)
#plot roc curve train
fpr,tpr, _ =roc_curve(Y,Y_pred)
plt.plot(fpr,tpr, label='ROC curve(area=%0.2f)' % auc)
plt.title('roc curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()

print("AUC-ROC SCORE", auc)

if auc>=0.8 and (precision_score_train>0.5 or recall_score_train> 0.5):
    print("model performance is promising for the unbalanced dataset")

elif auc>=0.7:
    print("model performance might be acceptable for unbalanced dataset")
else:
    print("model performance needs improvement for unbalanced dataset")


#top feature coefficients
def get_top_features(ridge_classifier, feature_names,meaningful_cols_names,num_features=5):
    coefs=np.abs(ridge_classifier.coef_.flatten())
    feature_importance=zip(feature_names,coefs)
    sorted_features=sorted(feature_importance,key=lambda x: x[1],reverse=True)
    top_features=[feature[0] for feature in sorted_features[:len(feature_names)]]
    #print("topfeature:............",top_features)
    new_meaningful_features = []
    count = 0
    for col_name in meaningful_cols_names:
        for top_feature in top_features:
            if col_name in top_feature:
                new_meaningful_features.append(col_name)
                break
        count += 1
        if count == num_features:
            return new_meaningful_features
    print('top_features',new_meaningful_features)    
    return new_meaningful_features    
    #return top_features

def create_cobined_df(X,Y,confidence,pred):
    if len(X) !=len(Y):
        raise ValueError("Dataframe X and Y must have same length")
    
    Y.reset_index(drop=True,inplace=True)
    combined_data=pd.concat([X.reset_index(drop=True),Y], axis=1)
    # combined_data=pd.concat([X,Y], ignore_index=True)
    combined_data['pred']=pred
    combined_data['confidence']=confidence

    return combined_data



def anomaly_message(model, X,Y, feature_names,meaningful_cols_names):
        results=[]
        prediction = model.predict(X)# Get prediction for the data point
        proba = model.decision_function(X) # Get probability of each class
        for i in range(len(prediction)):
            confidence_score=expit(proba[i])
            #print("confidence_score prob:",confidence_score,proba[i])
            important_features = get_top_features(model, feature_names, meaningful_cols_names) # Extract top featur
            anomaly_label = "Yes" if prediction[i] == 1 else "No"
            confidence_score = confidence_score if prediction[i] == 1 else (100-confidence_score)
            top_features_str = ", ".join(important_features) # Join top features with commas
            message = f"""The record is classified as an {anomaly_label} based on a confidence score of {round(confidence_score*100)} and the decision is made based on the important variables{top_features_str}"""
            results.append(message)
        
        combined_df= create_cobined_df(X,Y,expit(proba),prediction) 
       
       
        

#Example usage (assuming you have a trained model, data point, and feature names)
model=ridge_classifier
X=X.copy()
feature_names=X.columns
meaningful_cols_names=selected_cols1.columns
message_dict=anomaly_message(model, X,Y, feature_names, meaningful_cols_names)
#print(message_dict)

models_dict={
    'ridge_classifier':ridge_classifier,
    'cat_one_hot_encoded':cat_one_hot_encoded
}
#save
joblib.dump(models_dict,'models.sav')