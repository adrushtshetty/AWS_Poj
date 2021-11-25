import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(r"C:\Users\Admin\Documents\.vscode\.vscode\Python\Crop_recommendation.csv")
df1 = df.copy()

Le = LabelEncoder()
df1['label'] = Le.fit_transform(df1['label'])
# print(df1.head())

# data_col = []
# for x in df1.columns:
#     data_col.append(x)

X=df1[['Nitrogen', 'Phosphorus', 'Potassium', 'temperature', 'humidity', 'ph', 'rainfall']]
Y=df1['label']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20)
clf = RandomForestClassifier()
clf.fit(X,Y)



# df2=df1.copy()
#
# for i in range(10):
#     x = df2.drop(['label'], axis = 1)
#     y = df2['label']
#     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20)
#     Model = RandomForestClassifier()
#     Model.fit(x_train,y_train)
#     y_prediction_for_test = Model.predict(x_test)
#     from sklearn.metrics import accuracy_score
#     result = accuracy_score(y_test,y_prediction_for_test)
#     print("Our model will predict correct flower type ",round(result*100,2),"% times")



# print("Welecome")
# print()
# print()
# print("-----------------------------------------------------------------------------------------")
# N = float(input("Enter your soils Nitrogen level: "))
# P = float(input("Enter your soils Phosphorus level: "))
# K = float(input("Enter your soils Potassium level: "))
# Temp = float(input("Enter the avg Temperature in your area: "))
# Humid = float(input("Enter the avg Humidity in your area: "))
# pH = float(input("Enter the pH level of the soil: "))
# Rain = float(input("Enter the avg Rainfall in your area: "))
# df_t = pd.read_csv(r"C:\Users\Admin\Python\AWS Proj\Crop_Recomendation\Crop_Recom_Test.csv")
# df_t.loc[0,"Nitrogen"]=N
# df_t.loc[0,"Phosphorus"]=P
# df_t.loc[0,"Potassium"]=K
# df_t.loc[0,"temperature"]=Temp
# df_t.loc[0,"humidity"]=Humid
# df_t.loc[0,"ph"]=pH
# df_t.loc[0,"rainfall"]=Rain
# df_t.to_csv(r"C:\Users\Admin\Python\AWS Proj\Crop_Recomendation\Crop_Recom_Test.csv", index = False)
test = pd.read_csv(r"C:\Users\Admin\Python\AWS_Proj\Crop_Recomendation\Crop_Recom_Test.csv")
prediction = clf.predict(test)
# if prediction[0] == 20:
#     print("Your soils meets the requirements to grow rice.")
# elif prediction[0] == 11:
#     print("Your soils meets the requirements to grow maize.")
# elif prediction[0] == 3:
#     print("Your soils meets the requirements to grow chickpeas.")
# elif prediction[0] == 9:
#     print("Your soils meets the requirements to grow kidneybeans.")
# elif prediction[0] == 18:
#     print("Your soils meets the requirements to grow pigeonpeas.")
# elif prediction[0] == 13:
#     print("Your soils meets the requirements to grow mothbeans.")
# elif prediction[0] == 14:
#     print("Your soils meets the requirements to grow mungbeans.")
# elif prediction[0] == 2:
#     print("Your soils meets the requirements to grow blackgram.")
# elif prediction[0] == 10:
#     print("Your soils meets the requirements to grow lentil.")
# elif prediction[0] == 19:
#     print("Your soils meets the requirements to grow pomegranate.")
# elif prediction[0] == 1:
#     print("Your soils meets the requirements to grow bananas.")
# elif prediction[0] == 12:
#     print("Your soils meets the requirements to grow mangoes.")
# elif prediction[0] == 7:
#     print("Your soils meets the requirements to grow grapes.")
# elif prediction[0] == 21:
#     print("Your soils meets the requirements to grow watermelons.")
# elif prediction[0] == 15:
#     print("Your soils meets the requirements to grow muskmelons.")
# elif prediction[0] == 0:
#     print("Your soils meets the requirements to grow apples.")
# elif prediction[0] == 16:
#     print("Your soils meets the requirements to grow oranges.")
# elif prediction[0] == 17:
#     print("Your soils meets the requirements to grow papayas.")
# elif prediction[0] == 4:
#     print("Your soils meets the requirements to grow coconuts.")
# elif prediction[0] == 6:
#     print("Your soils meets the requirements to grow cotton.")
# elif prediction[0] == 8:
#     print("Your soils meets the requirements to grow jute.")
# elif prediction[0] == 5:
#     print("Your soils meets the requirements to grow coffee.")



# print("Welecome")
# print()
# print()
# print("-----------------------------------------------------------------------------------------")
# N = int(input("Enter your soils Nitrogen level: "))
# P = int(input("Enter your soils Phosphorus level: "))
# K = int(input("Enter your soils Potassium level: "))
# Temp = int(input("Enter the avg Temperature in your area: "))
# Humid = int(input("Enter the avg Humidity in your area: "))
# pH = int(input("Enter the pH level of the soil: "))
# Rain = int(input("Enter the avg Rainfall in your area: "))
# df_t = pd.read_csv(r"C:\Users\Admin\Python\AWS Proj\Crop_Recomendation\Crop_Recom_Test.csv")
# df_t.loc[0,"Nitrogen"]=N
# df_t.loc[0,"Phosphorus"]=P
# df_t.loc[0,"Potassium"]=K
# df_t.loc[0,"temperature"]=Temp
# df_t.loc[0,"humidity"]=Humid
# df_t.loc[0,"ph"]=pH
# df_t.loc[0,"rainfall"]=Rain
# df_t.to_csv(("C:\Users\Admin\Python\AWS Proj\Crop_Recomendation\Crop_Recom_Test.csv"), index = False)


pickle.dump(clf,open("Model.pkl","wb"))