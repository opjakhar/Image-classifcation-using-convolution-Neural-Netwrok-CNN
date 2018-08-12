
# coding: utf-8

# In[ ]:


#-----------importing package ----------------------------------------

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import argparse
np.random.seed(1234)
#---reading arguement ------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float,dest="abc")
parser.add_argument("--batch_size",type=int,dest="abc1")
parser.add_argument("--init",type=int,dest="abc2")
parser.add_argument("--save_dir", type=str,dest="abc3")
data_field= parser.parse_args()

#---------parameter information---------------------------------------

lr=data_field.abc                       
batch_size=data_field.abc1
init=data_field.abc2
model_file=data_field.abc3
flag=True

#-------------data load------------------------------------------------

train_data=pd.read_csv("train.csv")
train_label=train_data["label"]
train_labels=np.array(train_label)
train_data=np.array(train_data.iloc[:,1:785])
eval_data=pd.read_csv("val.csv")
eval_label=eval_data["label"]
eval_labels=np.array(eval_label)
eval_data=np.array(eval_data.iloc[:,1:785])
test_data=pd.read_csv("test.csv")
test_data=np.array(test_data.iloc[:,1:785])

#-----------one hot encoding------------------------------------------

one_hot_label=np.zeros(shape=(train_labels.shape[0],10))
for i in range(0,train_labels.shape[0]):
    one_hot_label[i][train_labels[i]]=1
#print(one_hot_label[0:5,:])

one_hot_val=np.zeros(shape=(eval_labels.shape[0],10))
for i in range(0,eval_labels.shape[0]):
    one_hot_val[i][eval_labels[i]]=1
#print(one_hot_label[0:5,:])

#-----------place holder-----------------------------------------------

input_data = tf.placeholder(tf.float32, shape=[None, 28,28,1], name='input_data')
label_onehot = tf.placeholder(tf.float32, shape=[None, 10], name='label_onehot')
actual_label = tf.argmax(label_onehot, dimension=1)

#-------session creation--------------------------------------------

session=tf.Session()
session.run(tf.global_variables_initializer())

#----------------convolution  layers-------------------------------------

if init ==1:    #-----xavier intializer--------------------------
    cnn_layer1 = tf.layers.conv2d(inputs=input_data,filters=64,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    cnn_layer1=tf.contrib.layers.batch_norm(
            inputs=cnn_layer1,
            is_training=flag)
    print(cnn_layer1.shape)
    print("before relu check point1 layer1")
    cnn_layer1=tf.nn.relu(cnn_layer1)
    print(cnn_layer1.shape)
    print("before relu check point1 layer1")
    cnn_pool1 = tf.layers.max_pooling2d(inputs=cnn_layer1, pool_size=[2, 2], strides=1)
    print("cnn_pool1 size")
    print(cnn_pool1.shape)
    
    cnn_layer2 = tf.layers.conv2d(      inputs=cnn_pool1,filters=128,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.xavier_initializer())
    cnn_layer2=tf.contrib.layers.batch_norm(
            inputs=cnn_layer2,
            is_training=flag)
    cnn_layer2=tf.nn.relu(cnn_layer2)

    print(cnn_layer2.shape)
    print("check point3")
    cnn_pool2 = tf.layers.max_pooling2d(inputs=cnn_layer2, pool_size=[2, 2], strides=1)
    print(cnn_pool2.shape)
    print("check point4")
    cnn_layer3 = tf.layers.conv2d(  inputs=cnn_pool2,filters=128,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.xavier_initializer())
    cnn_layer3=tf.contrib.layers.batch_norm(
            inputs=cnn_layer3,
            is_training=flag)
    cnn_layer3=tf.nn.relu(cnn_layer3)


    print(cnn_layer3.shape)
    print("check point5")
    cnn_layer4 = tf.layers.conv2d(inputs=cnn_layer3,filters=128,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.xavier_initializer())
    cnn_layer4=tf.contrib.layers.batch_norm(
            inputs=cnn_layer4,
            is_training=flag)
    cnn_layer4=tf.nn.relu(cnn_layer4)

    print(cnn_layer4.shape)
    print("check point6")
    cnn_pool3 = tf.layers.max_pooling2d(inputs=cnn_layer4, pool_size=[2, 2], strides=1)
    print(cnn_pool3.shape)
    print("check point7")
    pool3_flat = tf.reshape(cnn_pool3, [-1, 17 * 17 * 128])
    #     print(pool3_flat.shape)
    cnn_fc1 = tf.layers.dense(inputs=pool3_flat, units=1600, kernel_initializer=tf.contrib.layers.xavier_initializer())
    cnn_fc1=tf.contrib.layers.batch_norm(
            inputs=cnn_fc1,
            is_training=flag)
    cnn_fc1=tf.nn.relu(cnn_fc1)
    cnn_fc1 = tf.layers.dropout(inputs=cnn_fc1, rate=0.4, training=flag)
    print(cnn_fc1.shape)
    #     dropout = tf.layers.dropout(
    #      inputs=cnn_fc1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    print("check point8")
    cnn_fc2 = tf.layers.dense(inputs=cnn_fc1, units=1600, kernel_initializer=tf.contrib.layers.xavier_initializer())
    cnn_fc2=tf.contrib.layers.batch_norm(
            inputs=cnn_fc2,
            is_training=flag)
    cnn_fc2=tf.nn.relu(cnn_fc2)

    cnn_fc2 = tf.layers.dropout(inputs=cnn_fc2, rate=0.4, training=flag)
    print(cnn_fc2.shape)
    print("check point9")

    layer_fc3 = tf.layers.dense(inputs=cnn_fc2, units=10, kernel_initializer=tf.contrib.layers.xavier_initializer())

else: #-----he intializer-----------------------------------
    cnn_layer1 = tf.layers.conv2d(inputs=input_data,filters=64,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    
    cnn_layer1=tf.contrib.layers.batch_norm(
            inputs=cnn_layer1,
            is_training=flag)
    print(cnn_layer1.shape)
    print("before relu check point1 layer1")
    cnn_layer1=tf.nn.relu(cnn_layer1)
    print(cnn_layer1.shape)
    print("before relu check point1 layer1")
    cnn_pool1 = tf.layers.max_pooling2d(inputs=cnn_layer1, pool_size=[2, 2], strides=1)
    print("cnn_pool1 size")
    print(cnn_pool1.shape)
    
    cnn_layer2 = tf.layers.conv2d(      inputs=cnn_pool1,filters=128,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    cnn_layer2=tf.contrib.layers.batch_norm(
            inputs=cnn_layer2,
            is_training=flag)
    cnn_layer2=tf.nn.relu(cnn_layer2)

    print(cnn_layer2.shape)
    print("check point3")
    cnn_pool2 = tf.layers.max_pooling2d(inputs=cnn_layer2, pool_size=[2, 2], strides=1)
    print(cnn_pool2.shape)
    print("check point4")
    cnn_layer3 = tf.layers.conv2d(  inputs=cnn_pool2,filters=128,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    cnn_layer3=tf.contrib.layers.batch_norm(
            inputs=cnn_layer3,
            is_training=flag)
    cnn_layer3=tf.nn.relu(cnn_layer3)


    print(cnn_layer3.shape)
    print("check point5")
    cnn_layer4 = tf.layers.conv2d(inputs=cnn_layer3,filters=128,kernel_size=[3, 3],padding="valid",kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    cnn_layer4=tf.contrib.layers.batch_norm(
            inputs=cnn_layer4,
            is_training=flag)
    cnn_layer4=tf.nn.relu(cnn_layer4)

    print(cnn_layer4.shape)
    print("check point6")
    cnn_pool3 = tf.layers.max_pooling2d(inputs=cnn_layer4, pool_size=[2, 2], strides=1)
    print(cnn_pool3.shape)
    print("check point7")
    pool3_flat = tf.reshape(cnn_pool3, [-1, 17 * 17 * 128])
    #     print(pool3_flat.shape)
    cnn_fc1 = tf.layers.dense(inputs=pool3_flat, units=1024, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    cnn_fc1=tf.contrib.layers.batch_norm(
            inputs=cnn_fc1,
            is_training=flag)
    cnn_fc1=tf.nn.relu(cnn_fc1)

    print(cnn_fc1.shape)
    #     dropout = tf.layers.dropout(
    #      inputs=cnn_fc1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    cnn_fc1= tf.layers.dropout(inputs=cnn_fc1, rate=0.5, training=flag)
    
    print("check point8")
    cnn_fc2 = tf.layers.dense(inputs=cnn_fc1, units=1024, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    cnn_fc2=tf.contrib.layers.batch_norm(
            inputs=cnn_fc2,
            is_training=flag)
    cnn_fc2=tf.nn.relu(cnn_fc2)



    cnn_fc2 = tf.layers.dropout(inputs=cnn_fc2, rate=0.5, training=flag)
    print(cnn_fc2.shape)
    print("check point9")

    layer_fc3 = tf.layers.dense(inputs=cnn_fc2, units=10, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    
    

predict_prob = tf.nn.softmax(layer_fc3,name="predict_prob")
predict_class = tf.argmax(predict_prob, dimension=1)
loss_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,
                                                labels=label_onehot)
total_loss = tf.reduce_mean(loss_cross_entropy)
optimization_method = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
optimization_method2=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss)
correct_prediction_boolean = tf.equal(predict_class, actual_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction_boolean, tf.float32))

#-------------training model----------------------------------------------------------
session.run(tf.global_variables_initializer())
epochs=20
#batch_size=2
iterations=int(55000/batch_size)
iterations_valid=int(5000/batch_size)
early_stop=5
prev_loss=0
dict_train_loss=dict()
dict_val_loss=dict()
for i in range(0,epochs):
    train_loss=0
    for j in range(iterations):
        
        x_batch=train_data[j*batch_size:(j*batch_size)+batch_size,:]  
        x_batch=np.reshape(x_batch,(batch_size,28,28,1))
        y_true_batch=one_hot_label[j*batch_size:(j*batch_size)+batch_size,:]
        
        input_dict_tr = {input_data: x_batch,
                               label_onehot: y_true_batch}
        loss=session.run(total_loss,feed_dict=input_dict_tr)
        train_loss+=loss
        session.run(optimization_method, feed_dict=input_dict_tr)
        print("epoch is :{}".format(i))
        print("batch number is :{}".format(j))
    flag=False
    dict_train_loss[i]=train_loss/batch_size
    val_loss=0
    for j in range(iterations_valid):
        x_valid_batch=eval_data[j*batch_size:(j*batch_size)+batch_size,:]  
        x_valid_batch=np.reshape(x_valid_batch,(batch_size,28,28,1))
        y_valid_batch=one_hot_val[j*batch_size:(j*batch_size)+batch_size,:]

        input_dict_val = {input_data: x_valid_batch,
                      label_onehot: y_valid_batch}
        loss = session.run(total_loss, feed_dict=input_dict_val)
        val_loss+=loss
    dict_val_loss[i]=val_loss/batch_size
    if prev_loss<val_loss and early_stop==1:
        print("terminating because of early stop")
        break
    if prev_loss<val_loss:
        early_stop=early_stop-1	
    prev_loss=val_loss   
    correct_predi=0
    iterations11=int(5000/batch_size)
    for jj in range(iterations11):
        x_valid_batch=eval_data[jj*batch_size:(jj*batch_size)+batch_size,:]  
        x_valid_batch=np.reshape(x_valid_batch,(batch_size,28,28,1))
        y_valid_batch=one_hot_val[jj*batch_size:(jj*batch_size)+batch_size,:]
        input_dict_val = {input_data: x_valid_batch,
                  label_onehot: y_valid_batch}
        val_loss = session.run(total_loss, feed_dict=input_dict_val)
       # print("validation loss is  ",val_loss)
        temp=session.run(correct_prediction_boolean,feed_dict=input_dict_val)
        temp=tf.cast(temp,tf.int32)
        temp=tf.Session().run(temp)
        correct_predi=correct_predi+np.sum(temp)
    acc=correct_predi*100/5000
    print("validation accuracy{}".format(acc))
    
    flag=True



dic_parameters=(dict_train_loss,dict_val_loss)
with open('./'+model_file+'/val_train_loss.pickle','wb') as handle:
	pickle.dump(dic_parameters,handle)

saver = tf.train.Saver(tf.all_variables())
save = saver.save(session, './'+model_file+'/trained_model/model.ckpt')

flag=False    

#----------------------checking accuracy on validation data--------------
correct_predi=0
epochs=1
#batch_size=2
iterations=int(5000/batch_size)
for i in range(0,epochs):
    for j in range(iterations):
        x_valid_batch=eval_data[j*batch_size:(j*batch_size)+batch_size,:]  
        x_valid_batch=np.reshape(x_valid_batch,(batch_size,28,28,1))
        y_valid_batch=one_hot_val[j*batch_size:(j*batch_size)+batch_size,:]
        input_dict_val = {input_data: x_valid_batch,
                          label_onehot: y_valid_batch}
        val_loss = session.run(total_loss, feed_dict=input_dict_val)
        print("validation loss is  ",val_loss)
        temp=session.run(correct_prediction_boolean,feed_dict=input_dict_val)
        temp=tf.cast(temp,tf.int32)
        temp=tf.Session().run(temp)
        correct_predi=correct_predi+np.sum(temp)
    acc=correct_predi*100/5000
    print("validation accuracy{}".format(acc))

    
#----------------------generating label for test data------------------
    
correct_predi=0
#batch_size=2
iterations=int(10000/batch_size)
for j in range(iterations):
    print("test batch number {}".format(j))
    x_test_batch=test_data[j*batch_size:(j*batch_size)+batch_size,:]  
    x_test_batch=np.reshape(x_test_batch,(batch_size,28,28,1))
    #y_test_batch=one_hot_val[j*batch_size:(j*batch_size)+batch_size,:]
    feed_dict_test = {input_data: x_test_batch}
    #val_loss = session.run(cost, feed_dict=feed_dict_val)
    #print("validation loss is  ",val_loss)
    temp=session.run(predict_class,feed_dict=feed_dict_test)
    if j==0:
        test_label=list(temp)
    else:
        test_label.extend(list(temp))

f = open("./"+model_file+"/test_submission.csv","w+")
f.write("id,label\n")
for label_num in range(len(test_label)):
    f.write(str(label_num)+ "," + str(test_label[label_num]) + "\n")
f.close()        
session.close()
print("program successfully completed")

