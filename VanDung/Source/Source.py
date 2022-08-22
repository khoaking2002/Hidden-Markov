import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mean_squared_error
import csv
print('wait some mins for prepare')
#Read data
df_train = pd.read_csv("Google_Stock_Price_Train.csv", sep=",", header = 0, index_col = None)
df_test = pd.read_csv("Google_Stock_Price_Test.csv", sep=",", header = 0, index_col = None)
df_train = df_train.drop('Volume',axis=1)
df_test = df_test.drop('Volume',axis=1)
#Init_model
hmm = GaussianHMM(n_components=4)
fracChange_range = np.linspace(-0.1, 0.1, 50)
fracHigh_range = np.linspace(0, 0.1, 10)
fracLow_range = np.linspace(0, 0.1, 10)
possible_outcomes = np.array(list(itertools.product(fracChange_range, fracHigh_range, fracLow_range)))
#Train data
open_price_train = np.array(df_train["Open"]).astype(np.float)
close_price_train = np.array(df_train["Close"]).astype(np.float)
high_price_train = np.array(df_train["High"]).astype(np.float)
low_price_train = np.array(df_train["Low"]).astype(np.float)
fracChange = (close_price_train-open_price_train)/open_price_train
fracHigh = (high_price_train-open_price_train)/open_price_train
fracLow = (open_price_train-low_price_train)/open_price_train
obversation = np.column_stack((fracChange,fracHigh,fracLow))
hmm.fit(obversation)
#Predict close price
result_close=[]
close_real_value=[]
close_predict_value=[]
for i in range(0,df_test.shape[0]):
	date_current=df_test.iloc[i]["Date"]
	open_price_day=df_test.iloc[i]["Open"]
	close_price_day=df_test.iloc[i]["Close"]
	high_price_day=df_test.iloc[i]["High"]
	low_price_day=df_test.iloc[i]["Low"]
	fracChange_day = (close_price_day-open_price_day)/open_price_day
	fracHigh_day = (high_price_day-open_price_day)/open_price_day
	fracLow_day = (open_price_day-low_price_day)/open_price_day
	data_of_day = np.column_stack((fracChange_day,fracHigh_day,fracLow_day))
	outcome_score_list = []
	#predict
	for i in possible_outcomes:
		s = np.row_stack((data_of_day, i))
		outcome_score_list.append(hmm.score(s))
	predict_fracChange_day, predict_high_day, predict_low_day = possible_outcomes[np.argmax(outcome_score_list)]	
	result_close_predict=open_price_day * (1 + predict_fracChange_day)
	result_feature=(date_current,close_price_day,result_close_predict)
	result_close.append(result_feature)
	close_real_value.append(close_price_day)
	close_predict_value.append(result_close_predict)
result_df=pd.DataFrame(data = result_close, columns = ['Date','real_value', 'predict'])
print(result_df)
print('MSE value = ',mean_squared_error(close_real_value,close_predict_value))
#Data_visualazation 
ax=plt.gca()
result_df.plot(kind="line",x="Date", y="real_value", ax=ax)
result_df.plot(kind="line",x="Date", y="predict", ax=ax)
plt.ylabel("Google Stock Prices(in USD)")
plt.show()
