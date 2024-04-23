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
import json
import missingno as msno
import re

class Main:
    def __init__(self):
        
        self.loaded=joblib.load('input_path/models.sav')
        self.loaded_model=self.loaded['ridge_classifier']
        self.cat_one_hot_encoded=self.loaded['cat_one_hot_encoded']
        


    def predict(self, json_input):
        #json_input=json.loads('{"due_yyyy": 2023, "due_mm": 6,"fup_mas_yyyy": 2001,"fup_mas_mm": 10,"cm_mode": 12,"claim_status": "P", "Date of Death": "4/25/2023","Accounted On": "09/11/2000 00:00","policy_term": 5,"ord_prem_flag": 0,"life_death_ind": 0,"docket_availability": "Y","premium": 10000,"paid_date": "6/9/2023 00:00","commencement_date": "09/10/1996","total_payment": 54441,"total_deduction": 1888,"claim_amount": 54317,"vested_bonus": 0,"interim_bonus": 0}')
        data=json_input
        df = pd.DataFrame(data)

        df_cleaned= df.dropna(axis=1, how='all')
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

        def extract_date(cell):
            pattern=r'\b\d{2}/\d{2}/d{4}\b'
            matches = re.findall(pattern,str(cell))
            return' '.join(matches) if matches else cell       


        def check_Accounted(accounted_On):
            if pd.isnull(accounted_On):  # Check for NULL values
                return ' '
            else:
                try:
                    doc_date1 = pd.to_datetime(accounted_On)  # Convert string to datetime object
                    
                    if doc_date1.year == 0 or doc_date1.month == 0 or doc_date1.day == 0:  # Check for '00-00-000' dates
                        return ' '
                    else:
                        return doc_date1.strftime('%m-%d-%Y')  # Convert datetime object to string in 'dd-mm-yyyy' format
                        #return doc_date1
                except ValueError:
                    return ' '
                
            # Data quality checks for Policy Term column
        def check_policy_term(policy_term):
            if pd.isnull(policy_term):  # Check for NULL values
                return ' '
            else:
                if policy_term < 1 or policy_term > 100:  # Check if policy term is out of range
                    return ' '
                else:
                    return policy_term
                
        def format_master_fup(fup_mas_yyyy):
            if fup_mas_yyyy <= 0:
                return ' '
            else:
                #master_fup_str = str(master_fup)
                
                # year_pattern= r'^\d{4}$'
                # selected_cols['fup_mas_yyyy']= selected_cols['fup_mas_yyyy'].match(year_pattern)
                return fup_mas_yyyy
            
        # Function to preprocess Master FUP month
        def format_master_fup_mm(month):
            if month < 1 or month > 12:
                return ' '
            else:
                #master_fup_str = str(master_fup)
                return month
            
        # Function to preprocess
        def preprocess_numeric(value):
            if value < 0 or value> 999999999:  # Check for negative values (assuming negative values are errors)
                #print('numeric_columns Data Error')  # Replace negative values with None for data error
                return ' ' # Replace invalid dates with None for data error
            else:
                return value
            
        def preprocess_premium(premium):
            if premium < 10.00 or premium> 999999999.99:  # Check for negative values (assuming negative values are errors)
               
                return ' ' # Replace invalid dates with None for data error
            else:
                return premium
            
        # Function to preprocess Mode
        def preprocess_mode(cm_mode):
            if cm_mode not in [0, 1, 3, 6, 12]:  # Check for invalid mode values
                return ' '  # Replace invalid mode values with None for data error
            else:
                return cm_mode
            
        def convert_to_nonobject1(selected_cols1, object_columns1):
            for column in object_columns1:
                    selected_cols1[column]=selected_cols1[column].astype('float64')
            return selected_cols1
        
        # extracting categorical columns
        def convert_to_object(selected_cols1, non_object_columns):
            for column in non_object_columns:
                selected_cols1[column]=selected_cols1[column].astype(str)
            return selected_cols1
        
            #top feature coefficients
        def get_top_features(ridge_classifier, feature_names,meaningful_cols_names,num_features=5):
            coefs=np.abs(ridge_classifier.coef_.flatten())
            feature_importance=zip(feature_names,coefs)
            sorted_features=sorted(feature_importance,key=lambda x: x[1],reverse=True)
            top_features=[feature[0] for feature in sorted_features[:len(feature_names)]]
            # print("topfeature:............",top_features)
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
            # print('top_features',new_meaningful_features)    
            return new_meaningful_features    
            #return top_features

        def anomaly_message(model, X, feature_names,meaningful_cols_names):
                results=[]
                prediction = model.predict(X)# Get prediction for the data point
                proba = model.decision_function(X) # Get probability of each class
                for i in range(len(prediction)):
                    confidence_score=expit(proba[i])
                    #print("confidence_score prob:",confidence_score,proba[i])
                    important_features = get_top_features(model, feature_names, meaningful_cols_names) # Extract top featur
                    anomaly_label= "Anomaly" if prediction[i] == 1 else "Non Anomaly"
                    anomaly_pointer = "Yes" if prediction[i] == 1 else "No"

                    confidence_score = confidence_score if prediction[i] == 1 else (1-confidence_score)
                    top_features_str = ", ".join(important_features) # Join top features with commas
                    message = f"""The record is classified as {anomaly_label} based on a confidence score of {round(confidence_score*100)} and the decision is made based on the important variables above"""
                    #return 
                    result={
                    "1. Anomaly": anomaly_pointer,
                    "2. Confidence Score(%)": round(confidence_score*100),
                    "3. Important Variables": top_features_str,
                    "4. Full Message": message
                    }
                    
                    results.append(result)
                    #print(result)
                return results  
        
       
       
        selected_cols=df.copy()
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
        'total_deduction','claim_amount', 'vested_bonus', 'interim_bonus'
        ]]

        selected_cols['paid_date']= pd.to_datetime(selected_cols['paid_date'], errors='coerce')
        selected_cols['accounted_on']=pd.to_datetime(selected_cols['accounted_on'], errors='coerce')
        # Apply data quality check function to DOC column
        selected_cols['commencement_date'] = selected_cols['commencement_date'].apply(check_doc)
        # Apply data quality check function to DOC column
        selected_cols['accounted_on'] = selected_cols['accounted_on'].apply(check_Accounted)
        selected_cols['accounted_on']= selected_cols['accounted_on'].apply(extract_date)  
        selected_cols['accounted_on'] =pd.to_datetime(selected_cols['accounted_on'],errors='coerce')
        selected_cols['accounted_on']=pd.to_datetime(selected_cols['accounted_on']).dt.date

        # Apply data quality check function to Policy Term column
        selected_cols['policy_term'] = selected_cols['policy_term'].apply(check_policy_term)
        selected_cols['fup_mas_yyyy'] = selected_cols['fup_mas_yyyy'].apply(format_master_fup)
        selected_cols['fup_mas_mm'] = selected_cols['fup_mas_mm'].apply(format_master_fup_mm)



        numeric_columns = ['claim_amount', 'vested_bonus', 'interim_bonus']
        for column in numeric_columns:
            selected_cols[column] = selected_cols[column].apply(preprocess_numeric)

        selected_cols['premium'] = selected_cols['premium'].apply(preprocess_premium)
        selected_cols['cm_mode'] = selected_cols['cm_mode'].apply(preprocess_mode)
        #numerical corelation
        num_df = selected_cols.select_dtypes(include = ['int64','float64','int32'])

        # Scaling the numeric values in the dataset
        num_df_cols=num_df.columns

        #createb month and year cols from date cols


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
        selected_cols1=selected_cols1.replace(' ',pd.NaT).dropna()
   
        selected_cols1['policy_term']=selected_cols1['policy_term'].astype('int64')

        object_columns1 =['premium']


        selected_cols1=convert_to_nonobject1(selected_cols1,object_columns1)
        non_object_columns =['life_death_ind','ord_prem_flag','cm_mode']
        selected_cols1=convert_to_object(selected_cols1,non_object_columns)

        unseen_data=selected_cols1.copy()
        catagorical_columns=unseen_data.select_dtypes(include=['object']).columns
        # print(catagorical_columns)

        numerical_columns=unseen_data.select_dtypes(include=['int64','int32','float64'])
        # print(numerical_columns.columns)
        loaded_model=self.loaded_model
        cat_one_hot_encoded=self.cat_one_hot_encoded

        encoded_unseen_data=pd.get_dummies(unseen_data,columns=catagorical_columns, drop_first=True).astype(int)
        missing_cols=set(cat_one_hot_encoded.columns)-set(encoded_unseen_data.columns)
        if missing_cols:
            for column in missing_cols:
                encoded_unseen_data[column]=0
        encoded_unseen_data= encoded_unseen_data[cat_one_hot_encoded.columns]  
        encoded_unseen_data=encoded_unseen_data.drop(columns=['Anomaly Flag'])

        pred=loaded_model.predict(encoded_unseen_data)
        #print('predicted result',pred)
        X1=encoded_unseen_data

        model=loaded_model
        X=X1.copy()
        feature_names=X.columns
        meaningful_cols_names=unseen_data.columns
        message_dict=anomaly_message(model, X, feature_names, meaningful_cols_names)
        #print(message_dict)

        # result = self.loaded_model.predict(data)  # Use the DataFrame directly for prediction
        # prediction_dict = {'result': result.tolist()}  # Convert result to list for JSON compatibility

        return message_dict

if __name__ == '__main__':

    try:
        # Get JSON input from the user
        json_input = json.loads(input().strip())
    

        # Create an instance of Main class
        m = Main()

        # Get prediction
        prediction = m.predict(json_input)

    except Exception as e:
        print(json.dumps({'error': str(e)}))
