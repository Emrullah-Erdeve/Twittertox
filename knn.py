import pandas as pd
import numpy as np
df = pd.read_csv("train.csv")
df.drop(["id"],axis=1,inplace = True)

df.dropna()

df.target = [0 if each < 0.5 else 0 for each in df.target]
y = df.target.values
x_ = df.drop(["target","parent_id",
               "publication_id","article_id","asian","atheist","bisexual","bisexual","black",
               "hindu","homosexual_gay_or_lesbian" ,"intellectual_or_learning_disability",
               "jewish","latino","muslim","other_disability","other_gender","other_race_or_ethnicity",
               "other_religion","other_sexual_orientation","physical_disability","psychiatric_or_mental_illness"
               ,"transgender","identity_annotator_count","male","white",
               "buddhist","christian","female","heterosexual","parent_id","comment_text","created_date","rating"],axis=1)

x_ = x_.astype('float32')



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(x_) 


from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x_,y,test_size = 0.3,random_state = 42)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))

