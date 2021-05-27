import pandas
import numpy
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv('Salary_Data.csv')
x = dataset['YearExperience'].values.reshape(30,1)
y = dataset['Salary']

model=LinearRegression()
model.fit(x,y)
user = int(input("Enter the years you want to pridict the salary of :")
output = model.predict([[user]])
print("Your salary be", output)