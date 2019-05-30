#import required packages
import pandas as pd
import matplotlib.pyplot as plt

#read the data
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df['DateTime'] = pd.to_datetime(df.DateTime , format = '%d/%m/%Y %H.%M',infer_datetime_format=True)
df_test['DateTime'] = pd.to_datetime(df_test.DateTime ,infer_datetime_format=True)
#check the dtypes

data = df[["PA","PB","PC","PD","PE","PF","PG"]]
data_test = df_test[["PA","PB","PC","PD","PE","PF","PG"]]
data.index = df.DateTime
print(df.dtypes)


from statsmodels.tsa.vector_ar.vecm import coint_johansen
#since the test works for only 12 variables,
#I would drop another and check the eigenvalues
coint_johansen(data,-1,1).eig

from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=df)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(df_test))


pred = pd.DataFrame(index=range(0,len(prediction)),columns=["PA","PB","PC","PD","PE","PF","PG"])
for j in range(0,7):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j].astype(int)

#for i in cols:
#    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))

model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat.values)
