import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
import yfinance as yf


from keras.models import load_model

start = '2010-01-01'
end = '2023-12-31'

st.title('Stock Trend prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')

df = yf.download(user_input,start,end)
#df = data.DataReader(user_input,'yahoo',start,end)
df.head()

#describing the data

st.subheader('DATA FROM 2010 TO 2023')
st.write(df.describe())


st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA ')
ma200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)



data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range =(0,1))

data_training_array = scaler.fit_transform(data_train)

# loading my model
model = load_model('kera_model.h5')

past_100_days = data_train.tail(100)

final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test =[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test,y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler.scale_
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor



st.subheader('predictions vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)


