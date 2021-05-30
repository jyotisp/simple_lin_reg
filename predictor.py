import joblib
experience=float(input("Enter your Experience: "))
mind=joblib.load("salary_predictor.pk1")
salary=mind.predict([[experience]])
print("Your expected salary is: ",salary)
