from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel , Field , computed_field 
from typing import Literal ,Annotated
import pickle
import pandas as pd 

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://insurance-premium-frontend-hazel.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# import the ml model 

import joblib

model = joblib.load("model.pkl")


tier1_cities = [
    'Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad',
    'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'
]

tier2_cities = [ 'Gwalior','Jabalpur','Ujjain','Sagar','Kota',
                'Ajmer','Udaipur', 'Alwar','Meerut',
                'Prayagraj','Kanpur','Varanasi', 'Bareilly','Gorakhpur','Moradabad',
    'Amritsar','Jalandhar','Patiala','Jammu','Mangalore','Mysuru', 'Hubballi','Tiruchirappalli','Madurai',
    'Salem', 'Tiruppur','Thiruvananthapuram','Thrissur', 'Kozhikode','Vijayawada',
    'Guntur','Warangal','Tirupati', 'Cuttack', 'Rourkela', 'Siliguri', 'Durgapur', 'Jamshedpur', 'Dhanbad',
    'Bilaspur','Ambala','Hisar','Panipat','Rohtak', 'Shillong', 'Guwahati',
    'Agartala',  'Imphal','Aizawl','Itanagar'
]




# pydantic model to validate incoming data 
class UserInput(BaseModel):
    age:Annotated[int ,Field(..., gt=0 , lt=120 , description='Age of the user')]
    weight:Annotated[float ,Field(..., gt=0 , description='weight of the user')]
    height:Annotated[float ,Field(..., gt=0 , lt=2.5 , description='Height of the user')]
    income_lpa:Annotated[float,Field(..., gt=0, description='Annual salary of the user')]
    smoker:Annotated[bool ,Field(..., description='Is user a smoker')]
    city:Annotated[str ,Field(..., description='the city that the user belongs to')]
    occupation:Annotated[Literal['Engineer','Business','Doctor','Farmer','Driver','Teacher', 'Private Job','Student',
    'Government Job','Freelancer'] ,Field(..., description='Occupation of the user')]


    @computed_field
    @property
    def bmi(self)->float:
        return self.weight/(self.height**2)


    @computed_field
    @property
    def lifestyle_risk(self)->str:
        if self.smoker and self.bmi> 30 :
            return 'hight'
        elif self.smoker or self.bmi > 27 :
            return "medium"
        else :
            return 'low'


    @computed_field
    @property    
    def age_group(self)->str:
        if self.age <25 :
            return 'young'
        elif self.age<45:
            return 'adult'
        elif self.age<60 :
            return 'middle_aged'
        return 'senior'

    @computed_field
    @property 
    def city_tier(self)->int:
        if self.city in tier1_cities:
            return 1
        elif  self.city in tier2_cities:
            return 2
        else :
            return 3



@app.post("/predict")
def predict_premium(data:UserInput):

    input_df=pd.DataFrame([{
        'bmi':data.bmi , 
        'age_group':data.age_group,
        'lifestyle_risk':data.lifestyle_risk,
        'city_tier':data.city_tier,
        'income_lpa':data.income_lpa,
        'occupation':data.occupation
    }])    


    prediction=model.predict(input_df)[0]

    return JSONResponse(status_code=200 ,content={'predicted_category': prediction})
