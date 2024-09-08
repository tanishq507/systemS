import pickle
from xgboost import XGBClassifier
import numpy as np

def predErr(machineID,volt,rotate,pressure,vibration):
    result = [machineID,volt,rotate,pressure,vibration]
    #Passing data to model & loading the model from disks

    model_path = 'ml_model/xgb_model.pkl'
    classifier = pickle.load(open(model_path, 'rb'))
    prediction = classifier.predict([result])[0]

    if prediction == 0:
        prediction = 'Error 1'
    elif prediction == 1:
        prediction = 'Error 2'
    elif prediction == 2:
        prediction = 'Error 3'
    elif prediction == 3:
        prediction = 'Error 4'
    elif prediction == 4:
        prediction = 'Error 5'
    else:
        prediction = 'no Error'
    
    
    return prediction

def predHel(s1,s2,s3,s4,s5,s6,s7,s8):
    s1 = float(s1)
    s2 = float(s2)
    s3 = float(s3)
    s4 = float(s4)
    s5 = float(s5)
    s6 = float(s6)
    s7 = float(s7)
    s8 = float(s8)
    
    new_data = np.array([s1,s2,s3,s4,s5,s6,s7,s8])
    
    model_path = 'ml_model/rf_hel_model.pkl'
    loaded_model = pickle.load(open(model_path, 'rb'))
    predictions = loaded_model.predict(new_data.reshape(1, 8))
    
    return predictions[0]

def predRUL(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15):
    s1 = float(s1)
    s2 = float(s2)
    s3 = float(s3)
    s4 = float(s4)
    s5 = float(s5)
    s6 = float(s6)
    s7 = float(s7)
    s8 = float(s8)
    s9 = float(s9)
    s10 = float(s10)
    s11= float(s11)
    s12 = float(s12)
    s13= float(s13)
    s14= float(s14)
    s15= float(s15)
   
    
    new_data = np.array([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15])
    
    model_path = 'ml_model/knn_rul_model.pkl'
    loaded_model = pickle.load(open(model_path, 'rb'))
    predictions = loaded_model.predict(new_data.reshape(1, 15))
    
    return predictions[0]