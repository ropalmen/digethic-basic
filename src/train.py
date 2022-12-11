import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

#Einlesen der Datei
df = pd.read_csv('data/auto-mpg-training-data.csv', sep = ";")

#Festlegen von Feautures und Target
x = df.drop(['mpg'], axis=1)
y = df['mpg']

#Aufteilen Testdaten
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Lineare Regression
reg = LinearRegression()

reg = reg.fit(x_train, y_train)

pred = reg.predict(x_test)

#Speichern der Datei
os.remove("data/models/linreg2.pickle")
os.rmdir("data/models")
os.mkdir("data/models")
file = open("data/models/linreg2.pickle", "wb")
pickle.dump(reg, file)
