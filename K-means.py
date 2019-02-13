#用k-means对亚洲足球队聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
#输入数据
data=pd.read_csv('data.csv',encoding='gbk')
train_x=data[['2019年国际排名','2018世界杯','2015亚洲杯']]
#数据标准化
scaler=preprocessing.StandardScaler()
train_x_scler=scaler.fit_transform(train_x)
#训练kmeans模型
kmeans=KMeans(n_clusters=5)
kmeans.fit(train_x_scler)
predict_y=kmeans.predict(train_x_scler)
#合并聚类结果
result=pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:'聚类'},axis=1,inplace=True)
print(result)
