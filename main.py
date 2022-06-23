from fastapi import FastAPI
import uvicorn
import pickle


pickle_model = open("model.pkl","rb")
model=pickle.load(pickle_model)

pickle_scaler = open("scaler.pkl","rb")
scaler=pickle.load(pickle_scaler)

app=FastAPI()
@app.get("/prediction")
async def get_prediction(gender,hypertension,heartdisease,married,employeed,glucose,bmi,smoking,age016,age1632,age3248,age4864,age64p):
    Glucose_level=float(glucose)
    Bmi=float(bmi)

    scale=scaler.transform([[Glucose_level,Bmi]]).ravel()
    Glucose_level=scale[0]
    Bmi=scale[1]

    result=model.predict([[int(gender),int(hypertension),int(heartdisease),int(married),int(employeed),Glucose_level,Bmi,int(smoking),int(age016),int(age1632),int(age3248),int(age4864),int(age64p)]])

    if result[0]==1:
        result='Yes'
    else:
        result='No'
    return {"Predicted":result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)