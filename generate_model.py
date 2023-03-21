from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Conv1D,Conv2D,Flatten,BatchNormalization,MaxPooling1D,Dropout,Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam,SGD
import datetime
from tensorflow.keras.utils import plot_model
import os
import flask
import json



def load_dataset():
    df = pd.read_csv('./final_1/final.csv')
    # drop columns
    df.drop(['Timestamp'], axis=1, inplace=True)
    df.drop(['Protocol','Dst Port'],axis=1,inplace=True)

    # duplicated
    print(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    # drop inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # drop columns
    print(df.info())

    print(df['Label'].value_counts())

    train_dataset = df
    print(train_dataset['Label'].value_counts())

    # df.to_csv('./final_1/final_had_p.csv',index=False)

    # encode label
    labelE = LabelEncoder()
    train_dataset['Label'] = labelE.fit_transform(train_dataset['Label'])
    train_dataset['Label'].value_counts()
    y = train_dataset['Label']
    train_dataset = train_dataset.drop(['Label'],axis=1)

    # normalize data
    sc = StandardScaler()
    train_dataset = sc.fit_transform(train_dataset)
    train_dataset = pd.DataFrame(train_dataset,columns=df.columns[:-1])
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(train_dataset, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def re_shape(X_train,X_test,y_train,y_test):
    X_train = X_train.to_numpy().reshape(len(X_train),X_train.shape[1],1)
    X_test = X_test.to_numpy().reshape(len(X_test),X_test.shape[1],1)
    return X_train,X_test,y_train,y_test


def mutil_head_evaluate_model(time,Xtrain,ytrain,Xtest,ytest,logdir,epochs=10,batch_size=128,lr =0.001):
    input1s = Input(shape=(Xtrain.shape[1],1))
    conv1d = Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(input1s)
    batch = BatchNormalization()(conv1d)
    maxpool = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch)
    flat1 = Flatten()(maxpool)

    input2s = Input(shape=(Xtrain.shape[1],1))
    conv2d = Conv1D(filters=64,kernel_size=5,activation='relu',padding='same')(input2s)
    batch2 = BatchNormalization()(conv2d)
    maxpool2 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch2)
    flat2 = Flatten()(maxpool2)
    
    input3s = Input(shape=(Xtrain.shape[1],1))
    conv3d = Conv1D(filters=128,kernel_size=7,activation='relu',padding='same')(input3s)
    batch3 = BatchNormalization()(conv3d)
    maxpool3 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch3)
    flat3 = Flatten()(maxpool3)

    concat = concatenate([flat1,flat2,flat3])
    dense1 = Dense(256,activation='relu')(concat)
    drop = Dropout(0.2)(dense1)
    dense2 = Dense(4,activation='softmax')(drop)
    model = Model(inputs=[input1s,input2s,input3s],outputs=dense2)
    #plot model
    plot_model(model, to_file=f'models/{time}/model_{time}.png', show_shapes=True, show_layer_names=True)

    opt = SGD(lr=lr, momentum=0.9)  
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = logdir + time
    tensorboard_callback = TensorBoard(log_dir=logdir)
    
    model.fit([Xtrain,Xtrain,Xtrain],ytrain,epochs=epochs,batch_size=batch_size,validation_data=([Xtest,Xtest,Xtest],ytest),callbacks=[tensorboard_callback],verbose=1)
    _,acc = model.evaluate([Xtest,Xtest,Xtest],ytest,verbose=0)
    return acc

def mutil_head_double_layers_evaluate_model(time,Xtrain,ytrain,Xtest,ytest,logdir,epochs=10,batch_size=128,lr =0.001):
    input1s = Input(shape=(Xtrain.shape[1],1))
    conv1d_1s1 = Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(input1s)
    batch_1s1 = BatchNormalization()(conv1d_1s1)
    maxpool_1s1 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_1s1)
    conv1d_1s2 = Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(maxpool_1s1)
    batch_1s2 = BatchNormalization()(conv1d_1s2)
    maxpool_1s2 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_1s2)
    flat1 = Flatten()(maxpool_1s2)

    input2s = Input(shape=(Xtrain.shape[1],1))
    conv1d_2s1 = Conv1D(filters=64,kernel_size=5,activation='relu',padding='same')(input2s)
    batch_2s1 = BatchNormalization()(conv1d_2s1)
    maxpool_2s1 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_2s1)
    conv1d_2s2 = Conv1D(filters=64,kernel_size=5,activation='relu',padding='same')(maxpool_2s1)
    batch_2s2 = BatchNormalization()(conv1d_2s2)
    maxpool_2s2 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_2s2)
    flat2 = Flatten()(maxpool_2s2)
    
    input3s = Input(shape=(Xtrain.shape[1],1))
    conv1d_3s1 = Conv1D(filters=128,kernel_size=7,activation='relu',padding='same')(input3s)
    batch_3s1 = BatchNormalization()(conv1d_3s1)
    maxpool3 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_3s1)
    conv1d_3s2 = Conv1D(filters=128,kernel_size=7,activation='relu',padding='same')(maxpool3)
    batch_3s2 = BatchNormalization()(conv1d_3s2)
    maxpool_3s2 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_3s2)
    flat3 = Flatten()(maxpool_3s2)

    concat = concatenate([flat1,flat2,flat3])
    dense1 = Dense(256,activation='relu')(concat)
    drop = Dropout(0.2)(dense1)
    dense2 = Dense(4,activation='softmax')(drop)
    model = Model(inputs=[input1s,input2s,input3s],outputs=dense2)
    #plot model
    plot_model(model, to_file=f'models/{time}/model_{time}.png', show_shapes=True, show_layer_names=True)

    opt = SGD(lr=lr, momentum=0.9)  
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = logdir + time
    tensorboard_callback = TensorBoard(log_dir=logdir)
    calls = [tensorboard_callback,
             EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto'),
             ModelCheckpoint(filepath=f'models/{time}/model_{time}.h5',monitor='val_loss',save_best_only=True,mode='auto'),
             ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1,mode='auto',min_delta=0.0001,cooldown=0,min_lr=0)
            ]
    model.fit([Xtrain,Xtrain,Xtrain],ytrain,epochs=epochs,batch_size=batch_size,validation_data=([Xtest,Xtest,Xtest],ytest),callbacks=[tensorboard_callback],verbose=1)
    # _,acc = model.evaluate([Xtest,Xtest,Xtest],ytest,verbose=0)
    # print("Accuracy: %.2f%%" % (acc*100))
    return model

def multil_head_replace_unit(time,Xtrain,ytrain,Xtest,ytest,logdir,epochs=10,batch_size=128,lr=0.0001):
    input1s = Input(shape=(Xtrain.shape[1],1))
    conv1d_1s1 = Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(input1s)
    batch_1s1 = BatchNormalization()(conv1d_1s1)
    maxpool_1s1 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_1s1)
    flat1 = Flatten()(maxpool_1s1)

    input2s = Input(shape=(Xtrain.shape[1],1))
    conv1d_2s1 = Conv1D(filters=32,kernel_size=5,activation='relu',padding='same')(input2s)
    batch_2s1 = BatchNormalization()(conv1d_2s1)
    maxpool_2s1 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_2s1)
    flat2 = Flatten()(maxpool_2s1)

    input3s = Input(shape=(Xtrain.shape[1],1))
    conv1d_3s1 = Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(input3s)
    batch_3s1 = BatchNormalization()(conv1d_3s1)
    maxpool_3s1 = MaxPooling1D(pool_size=3,strides=2,padding='same')(batch_3s1)
    flat3 = Flatten()(maxpool_3s1)

    concat = concatenate([flat1,flat2,flat3])
    dense1 = Dense(256,activation='relu')(concat)
    drop = Dropout(0.2)(dense1)
    dense2 = Dense(4,activation='softmax')(drop)
    model = Model(inputs=[input1s,input2s,input3s],outputs=dense2)
    #plot model
    plot_model(model, to_file=f'models/{time}/model_{time}.png', show_shapes=True, show_layer_names=True)
    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    logdir = logdir + time
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model.fit([Xtrain,Xtrain,Xtrain],ytrain,epochs=epochs,batch_size=batch_size,validation_data=([Xtest,Xtest,Xtest],ytest),callbacks=[tensorboard_callback],verbose=1)
    _,acc = model.evaluate([Xtest,Xtest,Xtest],ytest,verbose=0)
    return acc

def model_CNN1(time,Xtrain,ytrain,Xtest,ytest,logdir,epochs=10,batch_size=128,lr = 0.0001):
    model = Sequential()
    model.add(Conv1D(filters=32,kernel_size=3,activation='relu',input_shape=Xtrain.shape[1:],padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3,strides=2,padding='same'))

    model.add(Conv1D(filters=64,kernel_size=3,activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3,strides=2,padding='same'))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4,activation='softmax'))

    plot_model(model, to_file=f'models/{time}/model_{time}.png', show_shapes=True, show_layer_names=True)
    
    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = logdir + time
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model.fit(Xtrain,ytrain,validation_data=(Xtest,ytest),epochs=epochs,batch_size=batch_size,callbacks=[tensorboard_callback],verbose=1)
    _, accuracy = model.evaluate(Xtest, ytest, batch_size=batch_size, verbose=0)
    
    return accuracy

def summarize_result(scores,n_filters,time):
    print(scores,n_filters)
    for fil in range(len(n_filters)):
        m,s = np.mean(scores[fil]), np.std(scores[fil])
        print('Batch_size #%d: %.3f%% (+/-%.3f)' % (n_filters[fil],m,s))
    plt.boxplot(scores,labels=n_filters)
    plt.savefig(f'models/{time}/exp_cnn_{time}.png')
    plt.close()
        
def run_experiment(n_filters,X_train,X_test,y_train,y_test,repeats=10):
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.mkdir(f'./models/{time}')
    # repeat experiment
    logdir="logs/multi/" + time +'/'
    all_scores = list()
    for fil in n_filters:
        scores = list()
        for r in range(repeats):
            score = mutil_head_double_layers_evaluate_model(time,X_train,y_train,X_test,y_test,logdir,10,fil)
            score = score * 100.0
            print('>#%d #%d: %.3f' % (fil,r+1,score))
            scores.append(score)
        all_scores.append(scores)
    # summarize results
    summarize_result(all_scores,n_filters,time)


if __name__ == '__main__':
    n_filters = [64,128,256]
    X_train,X_test,y_train,y_test = load_dataset()
    X_train,X_test,y_train,y_test = re_shape(X_train,X_test,y_train,y_test)
    # n_filters = [256]
    # %tensorboard --logdir logs/fit
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(f'./models/{time}')
    logdir="logs/multi/" + time +'/'
    model = mutil_head_double_layers_evaluate_model(time,X_train,y_train,X_test,y_test,logdir,10,128)
    model.save(f'models/{time}/model_{time}.h5')
    # run_experiment(n_filters,X_train,X_test,y_train,y_test,1)
