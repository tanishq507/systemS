import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime

from dbFunc import newDoc, update_existing_document, get_all_docs, get_document, delete_document, get_documents_with_status, get_different_status
from errorPred import predErr, predHel, predRUL

i = 0
j = 0

# DO NOT REMOVE! IT'S A TRIBUTE TO OUR STAR MEMBER!
data = {
    'name'  :   'tanishq',
    'orientation'   :   'gay',
    'profile'   :   'hacker'
}

# newDoc(data, "userCollection", _id = 'newnew')


# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

@api_view(['GET', 'POST'])
def hello(request):
    # implement returen codes according to tasks required
    # read from firebase? and write into returncode
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    p_id = 0
    rcode = 0
    if request.method == 'POST':
        device = request.data.get('device')
        d = {
            'Ping id'   :   p_id,
            'device'    :   device,
            'time'      :   current_time 
        }
        newDoc(d,"pingLog", current_time)
        return Response({
                'Ping id'   :   p_id,
                'device'    :   device,
                'time'      :   current_time ,   
                'return code':  rcode      
            })
    return_data = {
        "device" : "unknown",
        "message" : "please don't trouble me with GETs. I'm already halndling a lot requests.",
    }
    return Response(return_data)



    ''' we have to:     get the data
                        store in firebase
                        trigger 2-3 models and append the data to firebase
                        
                        '''
@api_view(['GET', 'POST'])
def data(request):
    global i
    current_time = i
    i = i + 1
    if request.method == 'POST':
        device = request.data.get('device')
        MID = request.data.get('MID')
        volt = request.data.get('volt')
        press = request.data.get('press')
        rot = request.data.get('rot')
        vib = request.data.get('vib')
        s1 = request.data.get('s1')
        s2 = request.data.get('s2')
        s3 = request.data.get('s3')
        s4 = request.data.get('s4')
        s5 = request.data.get('s5')
        s6 = request.data.get('s6')
        s7 = request.data.get('s7')
        s8 = request.data.get('s8')

        machineID = float(MID)
        volt = float(volt)
        rotate = float(rot)
        pressure = float(press)
        vibration = float(vib)
        # we have to use the models for predictions datas 
        try: 
            error = predErr(machineID,volt,rotate,pressure,vibration)
        except Exception as e:
            error = "noErrNull"
        
        try:
            helth = predHel(s1,s2,s3,s4,s5,s6,s7,s8)
        except Exception as e:
            
            helth = "noHElthdata"

        # after all data generetaion
        d = {
            'device'    :   device,
            'time'      :   current_time ,
            'MID'    :  MID,
            'volt'      :   volt ,
            'rot'    :   rot,
            'press'      :   press ,
            'Error' :   error,
            'Health'    :   helth
        }
        newDoc(d,"dataLog", current_time)
        return Response({
                'device'    :   device,
                'time'      :   current_time ,   
                # 'return code':  rcode,
                'MID'    :   MID,
                'volt'      :   volt ,
                'rot'    :   rot,
                'press'      :   press ,
                'Error' :   error,
                'Health'    :   helth     
            })
    

@api_view(["POST"])
def predict_diabetictype(request):
    try:
        machineID = request.data.get('machineID',None)
        volt = request.data.get('volt',None)
        rotate = request.data.get('rotate',None)
        pressure = request.data.get('pressure',None)
        vibration = request.data.get('vibration',None)
        fields = [machineID,volt,rotate,pressure,vibration]
        
        if not None in fields:
            #Datapreprocessing Convert the values to float
            machineID = float(machineID)
            volt = float(volt)
            rotate = float(rotate)
            pressure = float(pressure)
            vibration = float(vibration)
            
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
            # conf_score =  np.max(classifier.predict_proba([result]))*100
            predictions = {
                'error' : '0',
                'message' : 'Successfull',
                'prediction' : prediction,
                # 'confidence_score' : conf_score
            }
        else:
            predictions = {
                'error' : '1',
                'message': 'Invalid Parameters'                
            }
    except Exception as e:
        predictions = {
            'error' : '2',
            "message": str(e)
        }
    
    return Response(predictions)

#RUL 
@api_view(['GET', 'POST'])
def RUL(request):
    global j
    current_time = j
    j+=1
    if request.method == 'POST':
        device = request.data.get('device')
        ID = request.data.get('ID')
        s1 = request.data.get('s1')
        s2 = request.data.get('s2')
        s3 = request.data.get('s3')
        s4 = request.data.get('s4')
        s5 = request.data.get('s5')
        s6 = request.data.get('s6')
        s7 = request.data.get('s7')
        s8 = request.data.get('s8')
        s9=  request.data.get('s9')
        s10= request.data.get('s10')
        s11= request.data.get('s11')
        s12= request.data.get('s12')
        s13= request.data.get('s13')
        s14= request.data.get('s14')
        s15 = request.data.get('s15')
        
        machineID = float(ID)
        # device = float(device)
        
        
         # we have to use the models for predictions datas 
        # try: 
        #     error = predErr(machineID,device)
        # except Exception as e:
        #     error = "noErrNull"
        
        try:
            helth = predRUL(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15)
        except Exception as e:
            
            helth = "noHElthdata"
        
        d = {
            'device'    :   device,
            'time'      :   current_time ,
            's1' :  s1,
            's2' :  s2,
            's3' :  s3,
            's4' :  s4,
            's5' :  s5,
            's6' :  s6,
            's7' :  s7,
            's8' :  s8,
            's9' :  s9,
            's10' :  s10,
            's11':  s11,
            's12' :  s12,
            's13':  s13,
            's14':  s14,
            's15':  s15,
            'RUL'    :   helth
        }
           
        newDoc(d,"RULLog", current_time)
        predictions = {
                'error' : '0',
                'message' : 'Successfull',
                'prediction' : helth,
                # 'confidence_score' : conf_score
            }
    return Response(predictions)