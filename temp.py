import numpy as np
import pickle 

loaded_model =pickle.load(open('trained_model.sav','rb'))

input_data =(1,2,3,4,5,3,4,5,5)
input_data_as_np=np.asarray(input_data)
input_data_reshaped = input_data_as_np.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print('Not Satisfied')
else:
    print('Sastified')