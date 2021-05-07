import matplotlib.pyplot as plt

# 两个数组
x_data1 = ['1','2','3','4','5']
y_data1 = [0.2,0.38,0.6,0.801,1]

# k-NN
x_data2 = ['1','2','3','4','5']
y_data2 = [0.11,0.25,0.52,0.63,0.87]

# NN
x_data3 = ['1','2','3','4','5']
y_data3 = [0.13,0.31,0.57,0.69,0.95]

# Cascade R-CNN
x_data4 = ['1','2','3','4','5']
y_data4 = [0.102,0.15,0.51,0.64,0.88]

# plt.plot(ranks,cmc1,label='ranking',color='red',marker='o',markersize=5)



plt.plot(x_data1,y_data1,color='red',label='our method',marker='o',markersize=3)
plt.plot(x_data2,y_data2,color='blue',label='K-NN',marker='o',markersize=3)
plt.plot(x_data3,y_data3,color='green',label='NN',marker='o',markersize=3)
plt.plot(x_data4,y_data4,color='black',label='Cascade R-CNN',marker='o',markersize=3)
plt.legend(['our method','K-NN','NN','Cascade R-CNN'])#
plt.ylabel('Precision',fontsize=15)
plt.xlabel('Rank_num',fontsize=15)
plt.title('CMC Curve')

plt.show()