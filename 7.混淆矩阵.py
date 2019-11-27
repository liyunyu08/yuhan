

["coast","forset","highway","insid","mount","openc","street","tallb"]

def my_confusion_matrix(y_true, y_pred):  
    from sklearn.metrics import confusion_matrix  
    label = ["coast","forset","highw","insid","mount","openc","street","tallb"]  
    labels = list(set(y_true)) 
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)  
    print "confusion_matrix(left labels: y_true, up labels: y_pred):"  
    print "labels\t",  
    for i in range(len(label)):  
        print label[i],"\t",  
    print   
    for i in range(len(conf_mat)):  
        print label[i],"\t",  
        for j in range(len(conf_mat[i])):  
            print int(conf_mat[i][j]) / 100.0,'\t',  
        print   
    print 


b = [0] * 100 + [1] * 100+ [2] * 100+ [3] * 100+ [4] * 100+ [5] * 100+ [6] * 100+ [7] * 100



import matplotlib.pyplot as plt
import pylab as pl
import seaborn
labels = ["coast","forset","highw","insid","mount","openc","street","tallb"]  
def plot_confusion_matrix(y_true, y_pred, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(50, 40), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)
    plt.show()