import PIL.Image as image
import numpy as np
from sklearn.cluster import KMeans
#用k-means对图像进行分割
#加载图像，并对数据规范化
def load_data(filepath):
    f=open(filepath,'rb')
    data=[]
    #图像像素值
    img=image.open(f)
    #图像尺寸
    width,height=img.size
    for x in range(width):
        for y in range(height):
            #点（x，y）三个通道值
            c1,c2,c3=img.getpixel((x,y))
            data.append([(c1+1)/256,(c2+1)/256,(c3+1)/256])
    f.close()
    print(data)
    return np.mat(data),width,height
#加载图像，得到规范化结果以及图像尺寸
img,width,height=load_data('weixin.jpg')
#用kmeans对图像进行16聚类
kmeans=KMeans(n_clusters=16)
label=kmeans.fit_predict(img)
#将图像聚类结果，转化成图像尺寸的矩阵
label=label.reshape([width,height])
#创建新图像img，用来保存图像聚类压缩后的结果
img=image.new('RGB',(width,height))
for x in range(width):
    for y in range(height):
        c1=kmeans.cluster_centers_[label[x,y],0]
        c2=kmeans.cluster_centers_[label[x,y],1]
        c3=kmeans.cluster_centers_[label[x,y],2]
        img.putpixel((x,y),(int(c1*256)-1,int(c2*256)-1,int(c3*256)-1))
img.save('weixin_new.jpg')