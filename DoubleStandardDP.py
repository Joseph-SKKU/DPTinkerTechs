### (0) Load packages ###
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.optimize import minimize
from scipy.stats import laplace

# (1) data preprocessing
def preprocessing(path):
    ## load original dataset
    data = pd.read_csv(path)
    
    ## make block in same class, based on reverse time stamp appearing during rolling
    data_check = data['Time(s)'].diff().fillna(1) < 0
    data_check = data_check[data_check == True].index.to_list()
    
    data_index = []
    for i in range(len(data_check)+1):
        num_index = i+1   
        if i == 0:
            num_iter = data_check[i] 
        elif i == len(data_check):
            num_iter = len(data) - data_check[i-1]
        else:
            num_iter = data_check[i] - data_check[i-1]
        for _ in range(num_iter):
            data_index.append(num_index)
    
    data['data_split'] = data_index
    
    ## choose interest featuers
    select_cols = ['Accelerator_Pedal_value', 'Intake_air_pressure', 'Engine_soacking_time',
                   'Engine_speed', 'Engine_torque', 'Current_Gear', 'Gear_Selection', 'Vehicle_speed',
                   'Acceleration_speed_-_Longitudinal', 'Acceleration_speed_-_Lateral',
                   'data_split', 'Class']
    
    sorted_cols = []
    
    for col in select_cols[:-2]:
        sorted_cols.append(col)
        for i in range(1, 6):
            sorted_cols.append(col+f'-{i}')
    
    sorted_cols = sorted_cols + select_cols[-2:]
    
    class_dicts = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9}
    data = data[select_cols]
    data_split = []
    
    for i in range(data['data_split'].max()):
        num_split = i+1
        tmp_data = data[(data['data_split'] == num_split)]
        data_split.append(tmp_data)
        
    ## make time-series data (window = 5)
    data_time_series = []
    
    for item in data_split:
        item = item.copy()
        for feature in select_cols[:-2]:
            for i in range(1, 6):
                item[f'{feature}-{i}'] = item[f'{feature}'].shift(i)
        item = item.dropna()
        item = item[sorted_cols]
        item['Class'] = item['Class'].apply(lambda x: class_dicts[x])
        data_time_series.append(item)
    
    output = pd.concat(data_time_series)    
    return output

def addLapNoise(original, sensitivity, epsilon):
    scale = sensitivity / epsilon
    laplace_noise = np.random.laplace(loc=0, scale=scale)
    return original + laplace_noise

def addLapNoiseDirect(original, estimated_mu, estimated_scale):
    laplace_noise = np.random.laplace(loc=estimated_mu, scale=estimated_scale)
    return original + laplace_noise

def laplace_log_likelihood(params, data):
    mu, scale = params
    log_likelihood = -np.sum(laplace.logpdf(data, loc=mu, scale=scale))
    return log_likelihood

def getEstimatedParams(sensitivity, epsilon, X_test_noise):
    # 95% 105% 하한, 상한 제안
    initial_params_up, initial_params_dn = [0, 1.05*(sensitivity / epsilon)], [0, 0.95*(sensitivity / epsilon)]
    # 데이터의 10퍼센트만 추정에 활용
    recovery_params_up, recovery_params_dn, = minimize(laplace_log_likelihood, initial_params_up, args=(X_test_noise.iloc[:int(0.1*len(X_test_noise))],)), minimize(laplace_log_likelihood, initial_params_dn, args=(X_test_noise.iloc[:int(0.1*len(X_test_noise))],))
    estimated_mu_up, estimated_scale_up = recovery_params_up.x
    estimated_mu_dn, estimated_scale_dn = recovery_params_dn.x
    estimated_mu, estimated_scale = 0.5*(estimated_mu_up+estimated_mu_dn), 0.5*(estimated_scale_up+estimated_scale_dn)
    return estimated_mu, estimated_scale

def getTransformMatrix(X_train, X_train_noise):
    X_train_values, X_train_noise_values = X_train.values, X_train_noise.values
    X_train_values_T = np.transpose(X_train_values)
    transform_matrix = np.linalg.inv(X_train_values_T.dot(X_train_values)).dot(X_train_values_T).dot(X_train_noise_values)
    X_test_noise_transform = X_test.dot(transform_matrix)
    X_test_noise_transform = pd.DataFrame(X_test_noise_transform, columns=X_train.columns)
    return X_test_noise_transform
    
def getMetric(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    f1= list(f1_score(y_test, y_pred, average=None))
    precision = list(precision_score(y_test, y_pred, average=None))
    recall = list(recall_score(y_test, y_pred, average=None))
    accuracy = accuracy_score(y_test, y_pred)
    result = {'F1-score': sum(f1)/len(f1), 'Precision': sum(precision)/len(precision), 'Recall': sum(recall)/len(recall), 'Accuracy': accuracy}
    return result

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(60, activation='linear'),  
            tf.keras.layers.Dense(50, activation='linear'),
            tf.keras.layers.Dense(40, activation='linear')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='linear'),
            tf.keras.layers.Dense(60, activation='linear')
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def genAutoEncoderX(X_train, X_train_noise, X_test, colnames):
    autoencoder = Autoencoder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    autoencoder.fit(X_train, X_train_noise,
                    epochs=1000,
                    batch_size=1000)
    encoded_X_test = autoencoder.encoder(X_test.values)
    decoded_X_test = autoencoder.decoder(encoded_X_test)
    decoded_X_test = pd.DataFrame(decoded_X_test, columns=colnames)
    return decoded_X_test
    
# (2) main

## preprocesing
raw_path = "./Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv" # https://ocslab.hksecurity.net/Datasets/driving-dataset
data = preprocessing(raw_path)
data = data.sample(frac=1).reset_index(drop=True)
X, y = data.drop(['Class', 'data_split'], axis=1), data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Original dataset
pure_ML     = XGBClassifier().fit(X_train, y_train)
metric_pure = getMetric(pure_ML, X_test, y_test)

## Reference (1) : Noise (DP) dataset
sensitivity   = X_train.max().max() - X_train.min().min() # > 6000
epsilon       = 1000 # custom
X_train_noise = X_train.apply(lambda row: addLapNoise(row, sensitivity=sensitivity, epsilon=epsilon), axis=1)
noise_ML      = XGBClassifier().fit(X_train_noise, y_train) # reference for various attacks

## Reference (2) : Noise testset
X_test_noise = X_test.apply(lambda row: addLapNoise(row, sensitivity=sensitivity, epsilon=epsilon), axis=1)
metric_laplacian = getMetric(noise_ML, X_test_noise, y_test)

## beta-shifted technique (1) : AutoEncoder
X_test_autoencoder = genAutoEncoderX(X_train, X_train_noise, X_test, X.columns) # loss ~ 80.3196
metric_autoencoder = getMetric(noise_ML, X_test_autoencoder, y_test)

## beta-shifted technique (2) : Optimal transport
X_test_noise_transform = getTransformMatrix(X_train, X_train_noise)
metric_transform = getMetric(noise_ML, X_test_noise_transform, y_test)

## beta-shifted technique (3) : Statistical Recovery
estimated_mu, estimated_scale = getEstimatedParams(sensitivity, epsilon, X_test_noise)
X_test_noise_recovery = X_test.apply(lambda row: addLapNoiseDirect(row, estimated_mu, estimated_scale), axis=1)
metric_statistical_recovery = getMetric(noise_ML, X_test_noise_recovery, y_test)

## output summary
output = {'REF_PURE': metric_pure,
          'REF_NOISE': metric_laplacian,
          'DOE_AUTOENCODER': metric_autoencoder,
          'DOE_OPTIMALTRANSPORT': metric_transform,
          'DOE_STATISTICALRECOVERY': metric_statistical_recovery}

output = pd.DataFrame.from_dict(output, orient='index')


