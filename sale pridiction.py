import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sc = StandardScaler()
data = pd.read_csv("C:\\Users\\sam\\Downloads\\Day_3\\Day_3\\DigitalAd_dataset.csv")
print(data.shape)
print(data.head(5))
x = data.iloc[:,:-1].values
y = data.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=0)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
comparison = np.concatenate((y_pred.reshape(-1, 1), y_test.to_numpy().reshape(-1, 1)), axis=1)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
age = int(input("enter the age"))
sal = int(input("enter the sal"))
cust = [[age,sal]]
res = model.predict(sc.transform(cust))
print(res)
if res[0] ==1 :
    print("custumer will buy")
else:
    print("custumer will not buy")