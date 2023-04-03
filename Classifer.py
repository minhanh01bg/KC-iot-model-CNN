import os
import sys
import logging

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)

AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = (78, 4)
BATCH_SIZE = 1024
NUM_CLASSES = 6

# MODEL_PATH = glob('/home/app/server/model/multiclass_*.h5')[0]
# TEMP_PATH = '/home/app/server/tmp/flow_images'
# FEATURES = ['dst_port', 'protocol', 'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_len', 'bwd_header_len', 'fwd_pkts_s', 'bwd_pkts_s',
#             'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts', 'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts', 'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']
# LABELS = ['Benign', 'Botnet', 'Bruteforce', 'DoS', 'Tấn công xâm nhập']


class NetworkClassifier:
    def __init__(self,  batch_size=512):
        # self.model = self.load_model()
        self.batch_size = batch_size
        self.current_model_hash = ''

    # def load_model(self):
    #     model = keras.models.load_model(MODEL_PATH)
    #     return model

    # def create_dataset(self, paths):
    #     def decode_image(path):
    #         bits = tf.io.read_file(path)
    #         image = tf.image.decode_png(bits, channels=4)
    #         image = tf.cast(image, tf.float32) / 255.0
    #         image = tf.reshape(image, IMAGE_SIZE)
    #         return image

    #     dataset = (
    #         tf.data.TFRecordDataset
    #         .from_tensor_slices(paths)
    #         .map(decode_image, num_parallel_calls=AUTO)
    #         .batch(BATCH_SIZE)
    #     )
    #     return dataset

    def predict(self,path):
        ddata = pd.read_csv(path)
        namere = {
                'ack_flag_cnt':'ACK Flag Cnt',
                'active_max' :'Active Max',
                'active_mean' :'Active Mean',
                'active_min' :'Active Min',
                'active_std' :'Active Std',
                'bwd_blk_rate_avg' :'Bwd Blk Rate Avg',
                'bwd_byts_b_avg' :'Bwd Byts/b Avg',
                'bwd_header_len': 'Bwd Header Len',
                'bwd_iat_max' :'Bwd IAT Max',
                'bwd_iat_mean' :'Bwd IAT Mean',
                'bwd_iat_min' :'Bwd IAT Min',
                'bwd_iat_std' :'Bwd IAT Std',
                'bwd_iat_tot' :'Bwd IAT Tot',
                'bwd_pkt_len_max' :'Bwd Pkt Len Max',
                'bwd_pkt_len_mean' :'Bwd Pkt Len Mean',
                'bwd_pkt_len_min' :'Bwd Pkt Len Min',
                'bwd_pkt_len_std' :'Bwd Pkt Len Std',
                'bwd_pkts_b_avg' :'Bwd Pkts/b Avg',
                'bwd_pkts_s' :'Bwd Pkts/s',
                'bwd_psh_flags' :'Bwd PSH Flags',#Bwd PSH Flags
                'bwd_seg_size_avg' :'Bwd Seg Size Avg',
                'bwd_urg_flags' :'Bwd URG Flags',
                'cwe_flag_count' :'CWE Flag Count',
                'down_up_ratio' :'Down/Up Ratio',
                'dst_port' :'Dst Port',
                'ece_flag_cnt':'ECE Flag Cnt',
                'fin_flag_cnt':'FIN Flag Cnt',
                'flow_byts_s' :'Flow Byts/s',
                'flow_duration' :'Flow Duration',
                'flow_iat_max' :'Flow IAT Max',
                'flow_iat_mean' :'Flow IAT Mean',
                'flow_iat_min' :'Flow IAT Min',
                'flow_iat_std' :'Flow IAT Std',
                'flow_pkts_s' :'Flow Pkts/s',
                'fwd_act_data_pkts' :'Fwd Act Data Pkts',
                'fwd_blk_rate_avg' :'Fwd Blk Rate Avg',
                'fwd_byts_b_avg' :'Fwd Byts/b Avg',
                'fwd_header_len' :'Fwd Header Len',
                'fwd_iat_max' :'Fwd IAT Max',
                'fwd_iat_mean' :'Fwd IAT Mean',
                'fwd_iat_min' :'Fwd IAT Min',
                'fwd_iat_std' :'Fwd IAT Std',
                'fwd_iat_tot' :'Fwd IAT Tot',
                'fwd_pkt_len_max' :'Fwd Pkt Len Max',
                'fwd_pkt_len_mean' :'Fwd Pkt Len Mean',
                'fwd_pkt_len_min' :'Fwd Pkt Len Min',
                'fwd_pkt_len_std' :'Fwd Pkt Len Std',
                'fwd_pkts_b_avg' :'Fwd Pkts/b Avg',
                'fwd_pkts_s' :'Fwd Pkts/s',
                'fwd_psh_flags' :'Fwd PSH Flags', #
                'fwd_seg_size_avg' :'Fwd Seg Size Avg',
                'fwd_seg_size_min' :'Fwd Seg Size Min',
                'fwd_urg_flags' :'Fwd URG Flags',
                'idle_max' :'Idle Max',
                'idle_mean' :'Idle Mean',
                'idle_min' :'Idle Min',
                'idle_std' :'Idle Std',
                'init_bwd_win_byts' :'Init Bwd Win Byts',
                'init_fwd_win_byts' :'Init Fwd Win Byts',
                'pkt_len_max' :'Pkt Len Max',
                'pkt_len_mean' :'Pkt Len Mean',
                'pkt_len_min' :'Pkt Len Min',
                'pkt_len_std' :'Pkt Len Std',
                'pkt_len_var' :'Pkt Len Var',
                'pkt_size_avg' :'Pkt Size Avg',
                'protocol' :'Protocol',
                'psh_flag_cnt' :'PSH Flag Cnt',#
                'rst_flag_cnt' :'RST Flag Cnt',
                'subflow_bwd_byts' :'Subflow Bwd Byts',
                'subflow_bwd_pkts' :'Subflow Bwd Pkts',
                'subflow_fwd_byts' :'Subflow Fwd Byts',
                'subflow_fwd_pkts' :'Subflow Fwd Pkts',
                'syn_flag_cnt' :'SYN Flag Cnt',#
                'timestamp' :'Timestamp',
                'tot_bwd_pkts' :'Tot Bwd Pkts',
                'tot_fwd_pkts' :'Tot Fwd Pkts',
                'totlen_bwd_pkts' :'TotLen Bwd Pkts',
                'totlen_fwd_pkts' :'TotLen Fwd Pkts',
                'urg_flag_cnt' :'URG Flag Cnt'
        }
        ddata = ddata.rename(columns=namere)
        # ddata.drop(['Timestamp','Label'],axis=1,inplace=True)
        df = ddata
        df_ip = df[['src_ip', 'dst_ip']]
        # Remove missing values
        df.dropna(axis=0, inplace=True, how="any")
        # Replace infinite values to NaN
        df.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        df.dropna(axis=0, how='any', inplace=True)

        # drop columns 
        colu= ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg']
        df.drop(colu,axis=1,inplace=True)
        # convert to numeric
        df[['Flow Byts/s', 'Flow Pkts/s']] = df[['Flow Byts/s', 'Flow Pkts/s']].apply(pd.to_numeric)
        # drop columns 
        colu = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Bwd Header Len', 'Pkt Len Mean', 'Bwd Byts/b Avg', 'Subflow Bwd Pkts', 'Idle Mean', 'Idle Max']
        df.drop(colu,axis=1,inplace=True)
        df_ip = df[['src_ip', 'dst_ip']]
        from pickle import load
        scaler = load(open('./normalization_and_model/preprocessor.pkl', 'rb'))
        # normalization
        numeric_features = ['Dst Port', 'Protocol', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
        'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
        'Bwd Pkt Len Std', 'Flow Byts/s', 'Fwd IAT Tot', 'Fwd IAT Mean',
        'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Fwd PSH Flags', 'Fwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Std', 'Pkt Len Var',
        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
        'ACK Flag Cnt', 'URG Flag Cnt', 'ECE Flag Cnt', 'Down/Up Ratio',
        'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
        'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
        'Subflow Fwd Byts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
        'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Std',
        'Idle Min']
        df = pd.DataFrame(scaler.transform(df), columns=numeric_features)
        # print(df.columns)
        df = df.to_numpy().reshape(df.shape[0],df.shape[1],1)
        # 1. load model
        print(df.shape)
        model = load_model('./normalization_and_model/model_2023-03-03-18-19-11.h5')
        # 2. predict
        y_pred = model.predict([df,df,df])
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = pd.DataFrame(y_pred, columns=['label'])
        y_pred['label'].value_counts()
        y_pred['label'].value_counts().plot(kind='bar')
        ans = pd.concat([df_ip, y_pred], axis=1)
        return ans


    # def predict(self, csv_path):
    #     test_df = pd.read_csv(csv_path)
    #     feat_df = test_df.loc[:, FEATURES]

    #     # Convert network flow to RGBA image
    #     if os.path.exists(TEMP_PATH):
    #         rmtree(TEMP_PATH)
    #         os.mkdir(TEMP_PATH)
    #     else:
    #         os.mkdir(TEMP_PATH)
    #     for index, row in feat_df.iterrows():
    #         try:
    #             data = row.values.astype('float')
    #             image = Image.fromarray(data, mode='RGBA')
    #             image.save(f'{TEMP_PATH}/{index}.png')
    #         except Exception as e:
    #             print(e)

    #     # Create test dataset
    #     test_paths = glob(f'{TEMP_PATH}/*.png')
    #     test_dataset = self.create_dataset(test_paths)

    #     y_probs = self.model.predict(test_dataset)
    #     y_pred = np.argmax(y_probs, axis=1)

    #     ret_df = test_df.loc[:, ['src_ip', 'dst_ip']]
    #     ret_df.loc[:, 'pred'] = y_pred
    #     ret_df.drop(ret_df[ret_df['pred'] == 0].index, inplace=True)
    #     ret_df.loc[:, 'pred'] = ret_df.pred.apply(lambda x: LABELS[x])

    #     ret_df = ret_df.groupby(ret_df.columns.tolist(), as_index=False).size()
    #     ret_df.drop(ret_df[ret_df['size'] <= 5].index, inplace=True)
    #     ret_df.rename(columns={'size': 'N of flows'}, inplace=True)

    #     return ret_df


if __name__ == '__main__':
    csv_path = sys.argv[1]
    model = NetworkClassifier()
    ret = model.predict(csv_path)
    print(ret)
