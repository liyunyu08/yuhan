#coding=utf-8
from svmutil import *
import multiprocessing


testfile = 'bottleneck_features_test_LM.dat'
trainfile = 'bottleneck_features_train_LM.dat'
scale = 'scale'
path0 = 'UIUC_'
def train(fanc):
    for k in range(2,15,3):
        f1= open(str("libsvm/libsvm-3.22/python/save2/saveout"+str(k)+".txt"),"w")
        #default
        f1.writelines('Native')
        f1.writelines('\n')
        f1.writelines(path0+str(k)) 
        f1.writelines('\n')
        y, x = svm_read_problem(path0+str(k)+"/"+trainfile)#读入训练数据
        yt, xt = svm_read_problem(path0+str(k)+"/"+testfile)#训练测试数据
        # m = svm_train(y, x )#训练
        # f1.writelines('\n')
        # f1.writelines('default:')
        # p_label,p_acc,p_vals=svm_predict(yt,xt,m)#测试
        # f1.writelines(str(p_acc[0]))
        # f1.writelines('\n')
        m = svm_train(y, x ,'-t 1')#训练
        f1.writelines('-t 1:')
        p_label,p_acc,p_vals=svm_predict(yt,xt,m)#测试
        f1.writelines('\n')
        f1.writelines(str(p_acc[0]))
        #scale
        f1.writelines('\n')    
        f1.writelines('Scale')
        f1.writelines('\n')
        f1.writelines(path0+str(k)) 
        f1.writelines('\n')
        y, x = svm_read_problem(path0+str(k)+"/"+scale+trainfile)#读入训练数据
        yt, xt = svm_read_problem(path0+str(k)+"/"+scale+testfile)#训练测试数据
        # m = svm_train(y, x )#训练
        # f1.writelines('\n')
        # f1.writelines('default:')
        # p_label,p_acc,p_vals=svm_predict(yt,xt,m)#测试
        # f1.writelines(str(p_acc[0]))
        # f1.writelines('\n')
        m = svm_train(y, x ,'-t 1')#训练
        f1.writelines('-t 1:')
        p_label,p_acc,p_vals=svm_predict(yt,xt,m)#测试
        f1.writelines(str(p_acc[0]))
        f1.writelines('\n')   
        f1.close()
for i in range(1):
    t = multiprocessing.Process(target=train,args=(i,))
    t.start()

print "The number of CPU is:" + str(multiprocessing.cpu_count())
for p in multiprocessing.active_children():
    print("child   p.name:" + p.name + "\tp.id" + str(p.pid))
print "END!!!!!!!!!!!!!!!!!"