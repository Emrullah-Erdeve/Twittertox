
import pandas as pd
import numpy as np
df = pd.read_csv("train.csv")
df.drop(["id"],axis=1,inplace = True)

df.dropna()

df3=df

df3.dropna()

y = df3.target.values
x_ = df3.drop(["target","parent_id",
               "publication_id","article_id","asian","atheist","bisexual","bisexual","black",
               "hindu","homosexual_gay_or_lesbian" ,"intellectual_or_learning_disability",
               "jewish","latino","muslim","other_disability","other_gender","other_race_or_ethnicity",
               "other_religion","other_sexual_orientation","physical_disability","psychiatric_or_mental_illness"
               ,"transgender","identity_annotator_count","male","white",
               "buddhist","christian","female","heterosexual","parent_id","comment_text","created_date","rating"],axis=1)

x_ = x_.astype('float32')


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaled = scaler.fit_transform(x_) 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_,y,test_size = 0.5,random_state=42)

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(x_)


linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

# %%

y_head2 = linear_regression2.predict(x_polynomial)


