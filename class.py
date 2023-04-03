# import os
# import sys
# import pandas as pd
# import numpy as np
# from joblib import load
# from tensorflow.keras.models import load_model

# MUL_PATH = 'server/model/multiclass.h5'
# BIN_PATH = 'server/model/binary.h5'
# # FT = ['bwd_pkts_s', 'fwd_pkts_s', 'flow_pkts_s', 'fwd_pkt_len_min', 'bwd_iat_min', 'flow_duration', 'flow_iat_mean', 'flow_iat_std', 'fwd_iat_max', 'fwd_iat_tot', 'fwd_iat_std', 'flow_iat_max', 'init_bwd_win_byts', 'fwd_iat_mean', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_tot', 'bwd_pkt_len_min', 'pkt_len_min', 'flow_byts_s', 'fwd_pkt_len_mean', 'fin_flag_cnt', 'idle_mean', 'idle_std', 'idle_max', 'idle_min', 'pkt_len_mean', 'pkt_size_avg', 'pkt_len_var', 'fwd_pkts_b_avg', 'fwd_header_len', 'totlen_fwd_pkts',
# #       'flow_iat_min', 'fwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'active_min', 'active_std', 'active_mean', 'active_max', 'fwd_byts_b_avg', 'down_up_ratio', 'bwd_pkts_b_avg', 'init_fwd_win_byts', 'bwd_byts_b_avg', 'bwd_pkt_len_mean', 'totlen_bwd_pkts', 'fwd_seg_size_min', 'tot_fwd_pkts', 'bwd_header_len', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'fwd_blk_rate_avg', 'fwd_pkt_len_max', 'pkt_len_std']
# REDUNDANT_FT = ['src_ip', 'dst_ip', 'src_port', 'timestamp']

# NETWORK_FLOW_THRESHOLD = 20


# class DeepLearningModel:
#     def __init__(self, model_type='multiclass', batch_size=512):
#         self.model_type = model_type
#         # self.model = self.get_model()
#         # self.batch_size = batch_size

#     # def get_model(self):
#     #     model = Sequential()
#     #     model.add(Dense(78, activation='relu', input_shape=(78,)))
#     #     model.add(BatchNormalization())
#     #     model.add(Dense(64, activation='relu'))
#     #     model.add(BatchNormalization())
#     #     model.add(Dense(32, activation='relu'))
#     #     model.add(BatchNormalization())
#     #     model.add(Dense(8, activation='relu'))
#     #     model.add(Dropout(0.5))
#     #     if self.model_type == 'multiclass':
#     #         model.add(Dense(3, activation='softmax'))
#     #         model.load_weights(MUL_PATH)
#     #     elif self.model_type == 'binary':
#     #         model.add(Dense(1, activation='sigmoid'))
#     #         model.load_weights(BIN_PATH)
#     #     return model

#     def predict(self, csv_path):
#         test_df = pd.read_pickle(csv_path)
#         if self.model_type == 'multiclass':
#             return self.preprocessing_predict(test_df)
#         elif self.model_type == 'binary':
#             return self.predict_binary(test_df)
        
#     def preprocessing_predict(self,ddata):
#         ddata.drop(['Timestamp'],axis=1,inplace=True)
#         df = ddata
#         # Remove missing values
#         df.dropna(axis=0, inplace=True, how="any")
#         # Replace infinite values to NaN
#         df.replace([-np.inf, np.inf], np.nan, inplace=True)
        
#         # Remove infinte values
#         df.dropna(axis=0, how='any', inplace=True)

#         # drop columns 
#         colu= ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg']
#         df.drop(colu,axis=1,inplace=True)
#         # convert to numeric
#         df[['Flow Byts/s', 'Flow Pkts/s']] = df[['Flow Byts/s', 'Flow Pkts/s']].apply(pd.to_numeric)
#         # drop columns 
#         colu = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Bwd Header Len', 'Pkt Len Mean', 'Bwd Byts/b Avg', 'Subflow Bwd Pkts', 'Idle Mean', 'Idle Max']
#         df.drop(colu,axis=1,inplace=True)

#         from pickle import load
#         scaler = load(open('./normalization/preprocessor.pkl', 'rb'))
#         # normalization
#         numeric_features = ['Dst Port', 'Protocol', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
#        'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
#        'Bwd Pkt Len Std', 'Flow Byts/s', 'Fwd IAT Tot', 'Fwd IAT Mean',
#        'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
#        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
#        'Fwd PSH Flags', 'Fwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
#        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Std', 'Pkt Len Var',
#        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
#        'ACK Flag Cnt', 'URG Flag Cnt', 'ECE Flag Cnt', 'Down/Up Ratio',
#        'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
#        'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
#        'Subflow Fwd Byts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
#        'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
#        'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Std',
#        'Idle Min']
#         df = pd.DataFrame(scaler.transform(df), columns=numeric_features)
#         df = df.to_numpy().re_shape(df.shape[0],df.shape[1],1)
#         # 1. load model
#         model = load_model('./normalization/model_2023-03-03-15-35-38.h5')
#         # 2. predict
#         y_pred = model.predict([df,df,df])
#         y_pred = np.argmax(y_pred, axis=1)
#         y_pred = pd.DataFrame(y_pred, columns=['label'])
#         y_pred['label'].value_counts()
#         y_pred['label'].value_counts().plot(kind='bar')
        
#         return y_pred

#     def predict_multiclass(self, test_df):
#         test = test_df.drop(REDUNDANT_FT, axis=1).values
#         scaler = load('server/model/multiclass-scaler.joblib')
#         test = scaler.transform(test)

#         y_probs = self.model.predict(test, batch_size=self.batch_size)
#         y_pred = np.argmax(y_probs, axis=1)
#         ret_df = test_df.loc[:, ['src_ip', 'dst_ip']]
#         ret_df.loc[:, 'pred'] = y_pred
#         ret_df.drop(ret_df[ret_df['pred'] == 0].index, inplace=True)

#         ret_df['pred'] = ret_df.pred.apply(
#             lambda v: 'DOS' if v == 1 else 'Tấn công thăm dò')

#         ret_df = ret_df.groupby(ret_df.columns.tolist(), as_index=False).size()
#         ret_df.rename(columns={'size': 'num of flows'}, inplace=True)
#         ret_df.drop(ret_df[ret_df['num of flows'] <=
#                            NETWORK_FLOW_THRESHOLD].index, inplace=True)
#         return ret_df



# if __name__ == '__main__':
#     csv_path = sys.argv[1]

#     # model = DeepLearningModel(model_type='binary')
#     # ret_bin = model.predict(csv_path)
#     model = DeepLearningModel(model_type='multiclass')
#     ret_mul = model.predict(csv_path)
#     # print(ret_bin)
#     print(ret_mul)
