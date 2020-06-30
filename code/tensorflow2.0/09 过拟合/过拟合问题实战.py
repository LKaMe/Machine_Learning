#构建数据集
#导入数据集生成工具
from sklearn.datasets import make_moons
#从moon分布中随机采样1000个点，并切分为训练集-测试集
X,y = make_moons(n_samples = N_SAMPLES,noise=0.25,random_state=100)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=42)
#make_plot函数可以方便的根据样本的坐标X和样本的标签y绘制出数据的分布图
def make_plot(X,y,plot_name,file_name,XX=None,YY=None,preds=None):
    plt.figure()
    #sns.set_style("whitegrid")
    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min,y_max])
    axes.set(xlabel="$x_1$",ylabel="$x_2$")
    #根据网络输出绘制预测曲面
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX,YY,preds.reshape(XX.shape),25,alpha=0.08,cmap=cm.Spectral)
        plt.contour(XX,YY,preds.reshape(XX.shape),levels=[.5],cmap="Greys",vmin=0,vmax=.6)
        #绘制正负样本
        markers = ['o' if i == 1 else 's' for i in y.ravel()]
        mscatter(X[:,0],X[:,1],c=y.ravel(),s=20,cmap=plt.cm.Spectral,edgecolors='none',m=markers)
        #保存矢量图
        plt.savefig(OUTPUT_DIR + '/' + file_name)

#绘制数据集分布
make_plot(X,y,None,"dataset.svg")