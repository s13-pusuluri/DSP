import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
#import os
#exit(os.getcwd())

Owner_model = pickle.load(open('Srikar_svm_winning_model.pkl', "rb"))

print("\n*****************************************************")
print("* Riding Mowner Prediction *")
print("*****************************************************\n")

Income = float(input("Enter the Income Amount: "))
Lot_Size = float(input("Enter Lot Size: "))
df = pd.DataFrame({'Income': [Income], 'Lot_Size': [Lot_Size]})
# df = pd.DataFrame({'Lot_Size' : [Lot_Size]})
result = Owner_model.predict(df)
probability = Owner_model.predict_proba(df)
result_pred = ('Nonowner', 'Owner')
print(f"\nRiding Mowner Prediction indicates probability of Ownership at {probability[0][1]:.4f}, therefore it's indicated that we should {result_pred[result[0]]}.\n")
