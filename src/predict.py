import pandas as pd
import pickle

#Model einlesen
file_to_open = open("data/models/linreg2.pickle", 'rb')
model = pickle.load(file_to_open)
file_to_open.close()

pred_data = pd.read_csv("data/prediction-data.csv", sep = ";")

print(model.predict(pred_data))
