from src.travel.exception import TravelException
import sys, os  
from src.travel.logger import logging
from src.travel.pipeline.training_pipeline import TrainPipeline
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.travel.ml.model.estimator import ModelResolver
from src.travel.constant.training_pipeline import SAVED_MODEL_DIR
from src.travel.utils.main_utils import load_object


# if __name__=="__main__":
#     try:
#         training_pipeline = TrainPipeline()
#         training_pipeline.run_pipeline()

#     except Exception as e: 
#         raise TravelException(e, sys)





# -------------------------------------------------------------------------------------------
st.set_page_config(page_title='Travel Package Purchase Prediction App',
    layout='wide')

st.write("""
# The Travel Package Purchase Prediction App
""")

st.write("## About the Travel Package Purchase Prediction")
st.write("""

Tourism is one of the most rapidly growing global industries and tourism forecasting is becoming 
an increasingly important activity in planning and managing the industry. Because of high fluctuations
 of tourism demand, accurate predictions of purchase of travel packages are of high importance for
  tourism organizations.

The goal is to predict whether the customer will purchase the travel package or not.
""")



add_sidebar = st.sidebar.selectbox('Select Train route or Predict route', ('NONE', 'TRAIN','PREDICT',"FEATURE IMPORTANCE"))

if add_sidebar == 'TRAIN':
    st.write("In this robust system , Model gets trained as soon as you Go in Train of sidebar and hit training button")
    st.write("## Press the below button")
    if st.button("Train-Model"):
        try:
            
            train_pipeline = TrainPipeline()
            if train_pipeline.is_pipeline_running:
                st.write("Currently training pipeline is running please wait....")
            train_pipeline.run_pipeline()

            from src.travel.logger import LOG_FILE_PATH

            with open(file=LOG_FILE_PATH, mode="r") as txt:
                logs = txt.read()

        # from sensor.logger import LOG_FILE_PATH
            st.write(f"""Training successful !!\n\n
                            {logs}""")
        except Exception as e: 
            st.write("Trained model is not better than the best model which is already exists, either add more data or do some better model tune or better split the data or etc.")
            raise TravelException(e, sys)

if  add_sidebar == 'PREDICT':

    model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
    if not model_resolver.is_model_exists():
        st.write("## Model doesn't exist, please Train the Model")

    st.write("### Fill the details below")
    st.write("if you are writing a number then after writing please Press Enter")
    age = st.number_input("Age")
    st.write("-"*50)

    type_of_contact = st.selectbox("Type of Contact", ("Self Enquiry", "Company Invited"))
    st.write("-"*50)
    
    city_tier = int(st.selectbox("City_Tier", (1, 2, 3)))
    st.write("-"*50)
    
    duration_of_pitch = st.number_input("DurationOfPitch")
    st.write("-"*50)
    
    occupation = st.selectbox("Occupation", ('Salaried' ,'Free Lancer', 'Small Business', 'Large Business'))
    st.write("-"*50)

    gender =  st.selectbox("Gender", ('Female' ,'Male'))
    st.write("-"*50)

    num_of_person_visiting = st.selectbox("NumberOfPersonVisiting", (1, 2, 3, 4, 5))
    st.write("-"*50)

    num_of_followups = st.selectbox("NumberOfFollowups", (1, 2, 3, 4, 5, 6))
    st.write("-"*50)
 
    product_pitched = st.selectbox("ProductPitched", ('Deluxe' ,'Basic' ,'Standard' ,'Super Deluxe', 'King'))
    st.write("-"*50)

    preferred_property_star = st.selectbox("PreferredPropertyStar", [3,4,5])
    st.write("-"*50)

    marital_status = st.selectbox("MaritalStatus", ['Single', 'Divorced' ,'Married' ,'Unmarried'])
    st.write("-"*50)

    number_of_trips = st.number_input("NumberOfTrips")
    st.write("-"*50)

    passport = st.selectbox("Passport : {no:0, yes:1}", [0,1])
    st.write("-"*50)

    PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", [1,2,3,4,5])
    st.write("-"*50)

    OwnCar = st.selectbox("OwnCar : {no:0, yes:1}", [0,1])
    st.write("-"*50)

    NumberOfChildrenVisiting = st.selectbox("NumberOfChildrenVisiting", [1,2,3,4])
    st.write("-"*50)

    Designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
    st.write("-"*50)

    MonthlyIncome = st.number_input("MonthlyIncome")
    st.write("-"*50)

    if st.button("Predict"):
        data = {'Age' : age ,
        'TypeofContact' 	:  type_of_contact,
        'CityTier' : city_tier ,
        'DurationOfPitch' 	:  duration_of_pitch,
        'Occupation' 	:  occupation,
        'Gender' 	: gender ,
        'NumberOfPersonVisiting' 	: num_of_person_visiting ,
        'NumberOfFollowups' 	: num_of_followups ,
        'ProductPitched' 	:  product_pitched,
        'PreferredPropertyStar' 	:  preferred_property_star,
        'MaritalStatus' 	:  marital_status,
        'NumberOfTrips' 	:  number_of_trips,
        'Passport' 	: passport ,
        'PitchSatisfactionScore' 	: PitchSatisfactionScore ,
        'OwnCar' 	:  OwnCar,
        'NumberOfChildrenVisiting' 	: NumberOfChildrenVisiting ,
        'Designation' 	:Designation,
        'MonthlyIncome':MonthlyIncome}

        df = pd.DataFrame(data, index=[0])
        st.table(df)
        out = 0

        try:

            best_model_path = model_resolver.get_best_model_path()
            model = load_object(file_path=best_model_path)
            y_pred, y_pred_prob = model.predict(df)

        except Exception as e:
           raise TravelException(e,sys)

        st.write(f"""### Probability of a customer whether he/she will purchase Travel Package is ->\n
         for YES -> {y_pred_prob[0][1]}\n
         for NO -> {y_pred_prob[0][0]}""")


if  add_sidebar == 'FEATURE IMPORTANCE':
    
    columns = [
                    'Age'
                    ,'DurationOfPitch'
                    ,'NumberOfTrips'
                    ,'MonthlyIncome'
                    ,'CityTier'
                    ,'NumberOfPersonVisiting'
                    ,'NumberOfFollowups'
                    ,'PreferredPropertyStar'
                    ,'Passport'
                    ,'PitchSatisfactionScore'
                    ,'OwnCar'
                    ,'NumberOfChildrenVisiting'
                    ,'TypeofContact'
                    ,'Occupation'
                    ,'Gender'
                    ,'ProductPitched'
                    ,'MaritalStatus'
                    ,'Designation'
                    ]
                        
    try:

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            st.write("## Model doesn't exist, please Train the Model")
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)

        if st.button("Get Feature Importance"):
            importances = model.get_feature_importances()
            indices = np.argsort(importances)

            plt.figure(figsize=(12, 12))
            plt.title("Feature Importances")
            plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
            plt.yticks(range(len(indices)), [columns[i] for i in indices])
            plt.xlabel("Relative Importance")


            st.pyplot(plt)

    except Exception as e:
        raise TravelException(e,sys)