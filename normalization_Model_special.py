from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Conv1D,Conv2D,Flatten,BatchNormalization,MaxPooling1D,Dropout,Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import Activation,Add,Reshape,Permute,LeakyReLU,UpSampling2D,Conv2DTranspose,Concatenate
from tensorflow.keras.layers import Lambda,InputSpec,Layer,Input,Add,ZeroPadding2D,UpSampling2D,MaxPooling2D,Conv2D,Bidirectional,LSTM
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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

def preprocessing_predict(self,ddata):
    # ddata = pd.read_csv(path)
    df = ddata

    df.describe(include=[int, float])
    df.describe(include=[object]).transpose()
    df.isnull().sum().sum()
    df.isnull().sum() / df.shape[0]
    df.columns[df.isnull().any()]
    df.dropna(axis=0, inplace=True, how="any")

    np.all(np.isfinite(df.drop(['label'], axis=1)))
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    df[(df['flow_bytes_s'].isnull()) & (df['flow_packets_s'].isnull())].label.unique()
    df.dropna(axis=0, how='any', inplace=True)
    #  drop columns
    
    df[['flow_bytes_s', 'flow_packets_s']] = df[['flow_bytes_s', 'flow_packets_s']].apply(pd.to_numeric)

    cl_drop =['bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'cwr_flag_count']
    columns_drop = ['flow_duration', 'total_fwd_packet', 'total_bwd_packets', 
                    'total_length_of_fwd_packet', 'total_length_of_bwd_packet',
                    'fwd_packet_length_mean', 'bwd_packet_length_max', 'bwd_packet_length_mean', 
                    'flow_packets_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 
                    'bwd_header_length', 'packet_length_mean', 'bwd_bytes_bulk_avg', 'idle_mean', 'idle_max']
    # 22
    df.drop(labels=columns_drop, axis=1, inplace=True)
    df.drop(labels=cl_drop, axis=1, inplace=True)
    
    
    train_dataset = pd.read_csv('./normalization/normalization_data1M.csv')
    from pickle import load
    scaler = load(open('./normalization/preprocessor.pkl', 'rb'))
    # normalization
    categorical_features = train_dataset.select_dtypes(exclude=["int64", "float64"]).columns
    numeric_features = train_dataset.select_dtypes(exclude=[object]).columns

    preprocessor = ColumnTransformer(transformers=[
        ('categoricals', OneHotEncoder(drop='first', sparse=False, handle_unknown='error'), categorical_features),
        ('numericals', QuantileTransformer(), numeric_features)
    ])
    
    df = pd.DataFrame(scaler.transform(df), columns=preprocessor.get_feature_names())
    
    # 1. load model
    model = load_model('./normalization/model_2023-03-03-13-50-01.h5')
    # 2. predict
    y_pred = model.predict(df)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = pd.DataFrame(y_pred, columns=['label'])
    y_pred['label'].value_counts()
    y_pred['label'].value_counts().plot(kind='bar')
    
    return y_pred