import pandas as pd
import numpy as np
dataset=pd.read_csv("SalarayData.csv")
y=dataset["Salary"]
x=dataset["YearsExperience"].values.reshape(30,1)
from sklearn.linear_model import LinearRegression
mind=LinearRegression()
mind.fit(x,y)
print("weight: ",mind.coef_)
print("bias: ",mind.intercept_)
import joblib
joblib.dump(mind,"salary_predictor.pk1")
