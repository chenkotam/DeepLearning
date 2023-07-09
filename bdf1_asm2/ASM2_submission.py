import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import time
from tqdm import tqdm
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, LeakyReLU
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
#import statsmodels.tsa.stattools as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
import scipy.stats as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from bnn_vb import BNN_TaylorS, BNN_TaylorD, BNN_Reparam, Gaussian, Multinomial_Softmax, BNN_Base
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
np.random.seed(0)
tf.set_random_seed(0)

def x_small(returns, flows, stock_ID = 50286, retlag=3, flowlag=3):

    retlist=[]
    for i in range(retlag+1):
        retlist.append(returns[stock_ID].shift(i))

    flowlist=[]
    for i in range(flowlag):
        flowlist.append(flows[stock_ID].shift(i+1))
        
    return(pd.concat([pd.DataFrame(retlist).T, pd.DataFrame(flowlist).T], axis=1).dropna(axis=0).values)

def x_full(returns, flows, stock_target=50286, stock_list=[50286], retlag=3, flowlag=3):
    retlist=[]
    for i in range(retlag):
        retlist.append(returns[stock_list].shift(i+1))

    flowlist=[]
    for i in range(flowlag):
        flowlist.append(flows[stock_list].shift(i+1))
        
    for i in range(len(retlist)):
        if i==0:
            retdf = retlist[0]
        else:
            retdf = pd.concat([retdf, retlist[i]], axis=1)

    for i in range(len(flowlist)):
        if i==0:
            flowdf = flowlist[0]
        else:
            flowdf = pd.concat([flowdf, flowlist[i]], axis=1)
    
    if flowlag==0:
        X = retdf.dropna(axis=0)
    else:
        X = pd.concat([retdf, flowdf], axis=1).dropna(axis=0)
    lag = max(retlag, flowlag)
    return(pd.concat([returns[stock_target].iloc[lag:], X], axis=1).values)

#def x_full(returns, X, stock_ID=50286, lag=3):
#    return(pd.concat([returns[stock_ID].iloc[lag:], X], axis=1).values)

def x_small_MA(returns, flows, stock_ID = 50286, retlag=3, flowlag=3, ma_range=[3,5,10]):

    retlist=[]
    for i in range(retlag+1):
        retlist.append(returns[stock_ID].shift(i))

    flowlist=[]
    for i in range(flowlag):
        flowlist.append(flows[stock_ID].shift(i+1))
    
    data_bma = pd.concat([pd.DataFrame(retlist).T, pd.DataFrame(flowlist).T], axis=1)
    for i in range(len(ma_range)):
        data_bma['MA'+str(ma_range[i])] = returns[stock_ID].rolling(window=ma_range[i]).mean().shift(1)
    
    return(data_bma.dropna(axis=0).values)

def x_cf_quantile(data, quantile=[0.33, 0.66], window=30):
    y_actual=[]
    for i in range(len(data)-window):
        qt_threshold = np.quantile(data[i:i+window,0], quantile)
        if data[i+window,0] <= qt_threshold[0]:
            y_actual.append(0)
        elif data[i+window,0] > qt_threshold[1]:
            y_actual.append(2)
        else:
            y_actual.append(1)
    
    data = data[window:,:]
    data[:,0] = np.array(y_actual)
    return data

def x_cf_updown(data):
    y_actual=[]
    for i in range(len(data)):
        if data[i,0]>0:
            y_actual.append(1)
        else:
            y_actual.append(0)
    data[:,0] = np.array(y_actual)
    return data

def run_linreg(data, window=30):
    y_actual = data[window:,0]
    y_predict=[]
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
    #    y_test = data[i+window,0]

#        X_mean = np.mean(X_train,axis=0)
#        X_std = np.std(X_train,axis=0)
#        y_mean = np.mean(y_train)
#        y_std = np.std(y_train)
#        X_train = (X_train-X_mean)/X_std
#        y_train = (y_train - y_mean)/y_std
#        X_test = (X_test-X_mean)/X_std
    
        Beta = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)
        y_predict.append(float(X_test @ Beta))
#        y_predict.append(float(X_test @ Beta)*y_std + y_mean)
        
    y_predict = np.array(y_predict)
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return (y_predict-y_actual)**2

def run_histmean(data, window=30, ma=10):
    y_actual = data[window:,0]
    y_predict=[]
    for i in range(len(data)-window):
        y_predict.append(np.mean(data[i+window-ma:i+window,0]))
    
    y_predict = np.array(y_predict)
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return (y_predict-y_actual)**2

def run_GARCH(data, test_start=1000):
    y_actual = data[:,0]
    vn=[]
    hp=[]
    qs=[]
    pps=[]
    am = arch_model(y_actual, vol='GARCH', p=1, o=1, q=1, dist='Normal', mean='Zero')
    end_loc = test_start
    res = am.fit(disp='off', last_obs=end_loc)
    temp = res.forecast(horizon=1, start=end_loc).variance
    temp_ar = np.array(temp.dropna())
    fc_vol = np.reshape(temp_ar, len(temp_ar))
    yt = np.array(y_actual[end_loc:])
    fc_std = np.sqrt(fc_vol)
    
    #Computing PPS
    yt2 = np.array(y_actual[end_loc:]**2)
    test_size = len(y_actual[end_loc:])
    pps_pre = - (1/test_size) * (-0.5* test_size * np.log(2*np.pi) - 0.5*sum(np.log(fc_vol)) - 0.5*sum(yt2/fc_vol))
    pps.append(pps_pre)
    
    #Predictive Interval
    alpha=0.01
    var_up = 0 - st.norm.ppf(alpha/2)*fc_std #zero mean 99% forecast interval
    var_down = 0 + st.norm.ppf(alpha/2)*fc_std
    
    plt.figure(figsize=(10,5))
    plt.plot(yt, c='black')
    plt.plot(var_up, c='r')
    plt.plot(var_down, c='r')
    
    #violation number
    count_up=0
    count_down=0
    for i in range(len(yt)):
        if yt[i]<=var_down[i]:
            count_down += 1
        else:
            count_down += 0
        
        if yt[i]>=var_up[i]:
            count_up += 1
        else:
            count_up += 0
    
    violation_num = count_up + count_down
    vn.append(violation_num)
    
    #VaR
    VaR = st.norm.ppf(alpha)*fc_std
    
    #hit percentage
    count_VaR=0
    indicator=[]
    for i in range(len(yt)):
        if yt[i]<=VaR[i]:
            count_VaR += 1
            indicator.append(1)
        else:
            count_VaR += 0
            indicator.append(0)
            
    hit_percent = count_VaR/len(yt)
    hp.append(hit_percent)
    
    #QS
    diff = np.array([yt - VaR]).T
#    alpha = 0.01
    diff_ind = alpha - np.array([indicator])
    qs_pre = (diff_ind @ diff)/len(yt)
    qs.append(float(qs_pre))

    d = {'PPS': pps, 'Violation Number': vn, 'QS': qs, 'Hit Percentage': hp}
    df = pd.DataFrame(data=d)
    return df

def run_NN(data, window=30, skip=0):
    y_actual = data[window:,0]
    # define the keras model
    model = Sequential()
    model.add(Dense(10, input_dim=data.shape[1]-1, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    y_predict=[]
    skip_count=1e4
    for i in tqdm(range(len(data)-window)):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
    #    y_test = data[i+window,0]

        X_mean = np.mean(X_train,axis=0)
        X_std = np.std(X_train,axis=0)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        X_train = (X_train-X_mean)/X_std
        y_train = (y_train - y_mean)/y_std
        X_test = (X_test-X_mean)/X_std

        if skip_count>skip:
            model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
            skip_count=1
        else:
            skip_count+=1

        y_predict.append(float(model.predict(X_test))*y_std + y_mean)
#        y_predict.append(float(model.predict(X_test)))
    
    y_predict = np.array(y_predict)
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return (y_predict-y_actual)**2

def run_LSTM(data, window=30, skip=0):
    y_actual = data[window:,0]
    # define the keras model
    model = Sequential()
    model.add(LSTM(32, input_dim=data.shape[1]-1, return_sequences=False, activation='relu'))
#    model.add(SimpleRNN(10, activation='relu'))
#    model.add(LSTM(16, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
    #    y_test = data[i+window,0]

        X_mean = np.mean(X_train,axis=0)
        X_std = np.std(X_train,axis=0)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        X_train = (X_train-X_mean)/X_std
        y_train = (y_train - y_mean)/y_std
        X_test = (X_test-X_mean)/X_std

        # Transform 2D to 3D (For RNN and LSTM)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
        if skip_count>skip:
            model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
            skip_count=1
        else:
            skip_count+=1

        y_predict.append(float(model.predict(X_test))*y_std + y_mean)
#        y_predict.append(float(model.predict(X_test)))
        
    y_predict = np.array(y_predict)
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return (y_predict-y_actual)**2

def run_RNN(data, window=30, skip=0):
    y_actual = data[window:,0]
    # define the keras model
    model = Sequential()
    model.add(SimpleRNN(32, input_dim=data.shape[1]-1, return_sequences=False, activation='relu'))
#    model.add(SimpleRNN(32, activation='relu'))
#    model.add(SimpleRNN(16, activation='relu', return_sequences=False))
    model.add(Dense(1, activation='linear'))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    y_predict=[]
    skip_count=1e4
    for i in tqdm(range(len(data)-window)):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
    #    y_test = data[i+window,0]

        X_mean = np.mean(X_train,axis=0)
        X_std = np.std(X_train,axis=0)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        X_train = (X_train-X_mean)/X_std
        y_train = (y_train - y_mean)/y_std
        X_test = (X_test-X_mean)/X_std

        # Transform 2D to 3D (For RNN and LSTM)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
        if skip_count>skip:
            model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
            skip_count=1
        else:
            skip_count+=1

        y_predict.append(float(model.predict(X_test))*y_std + y_mean)
#        y_predict.append(float(model.predict(X_test)))
        
    y_predict = np.array(y_predict)
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return (y_predict-y_actual)**2

def run_onlineVI(method, data, natural=False, window=100, win_aft=30, skip=0):

    y_actual = data[window:,0]

    prior_mean = 0.0
    prior_var = 1.0
    hidden_size = [20]
    no_epochs = 2800
    batch_size = None
    act_fn=tf.nn.relu
    n_samples_train = 2
    likelihood = Gaussian(noise_var=0.01)
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):

        if i==0:
            x_train = data[i:i+window,1:]
            y_train = np.array([data[i:i+window,0]]).T
            x_test = np.array([data[i+window,1:]])
        else:
            x_train = data[i+window-win_aft:i+window,1:]
            y_train = np.array([data[i+window-win_aft:i+window,0]]).T
            x_test = np.array([data[i+window,1:]])
    
        x_mean = np.mean(x_train,axis=0)
        x_std = np.std(x_train,axis=0)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        x_train = (x_train-x_mean)/x_std
        y_train = (y_train - y_mean)/y_std
        x_test = (x_test-x_mean)/x_std
    
#        x_train, y_train, x_test, y_test, y_mean, y_std = get_data()
        N = x_train.shape[0]
        in_dim = x_train.shape[1]
        out_dim = y_train.shape[1]
        size = [in_dim] + hidden_size + [out_dim]
        np.random.seed(0)
        tf.set_random_seed(0)

        if skip_count>skip:
#            all_epoch_lb = []
#            all_raw_lb = []
#            all_time = []
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0015)
            if method == 'taylor_s':
                model = BNN_TaylorS(
                    size, N, likelihood=likelihood, 
                    n_samples_test=200, n_samples_train=n_samples_train, act_fn=act_fn,
                    optimizer=optimizer, prior_mean=prior_mean, prior_var=prior_var)
            elif method == 'taylor_d':
                model = BNN_TaylorD(
                    size, N, likelihood=likelihood, 
                    n_samples_test=200, n_samples_train=n_samples_train, act_fn=act_fn,
                    optimizer=optimizer, prior_mean=prior_mean, prior_var=prior_var)
            elif method == 'reparam':
                model = BNN_Reparam(
                    size, N, likelihood=likelihood, 
                    n_samples_test=200, n_samples_train=n_samples_train, act_fn=act_fn,
                    optimizer=optimizer, prior_mean=prior_mean, prior_var=prior_var)
#            init_lb = model.compute_energy(x_train, y_train, batch_size=batch_size)
#            print ('init energy: %.4f' % init_lb)
            if method == 'reparam':
                epoch_lb, raw_lb, train_time = model.train(x_train, y_train, 
                    no_epochs=no_epochs, batch_size=batch_size, nat_grad=natural)
            else:
        #            approx_epoch_lb, epoch_lb, approx_raw_lb, raw_lb, train_time = model.train(x_train, y_train, 
        #                no_epochs=no_epochs, batch_size=batch_size, nat_grad=natural, return_exact_bound=True)
                 epoch_lb, raw_lb, train_time = model.train(x_train, y_train, 
                     no_epochs=no_epochs, batch_size=batch_size, nat_grad=natural, return_exact_bound=False)
            
            prior_mean, prior_std = model.get_weights()
            prior_var = prior_std**2
        
#            epoch_lb.insert(0, init_lb)
#            raw_lb.insert(0, init_lb)
#            train_time.insert(0, 0)
#            all_epoch_lb.extend(epoch_lb)
#            all_raw_lb.extend(raw_lb)
#            all_time.extend(train_time)
            skip_count=1
        else:
            skip_count+=1

#    np.savetxt('/tmp/dataset_bnn_cost_%s_%s.txt'%(method, natural), np.array(all_epoch_lb), fmt='%.4f') 
#    np.savetxt('/tmp/dataset_bnn_time_%s_%s.txt'%(method, natural), np.array(all_time), fmt='%.4f') 

        pred = model.predict(x_test)
        mf = np.mean(pred, axis=0)
        mf_scale = mf*y_std + y_mean
        y_predict.append(float(mf_scale))
    
    y_predict = np.array(y_predict)
#    rmse = np.sqrt(np.mean((y_actual - y_predict)**2))
#    print (method, rmse)

    model.close_session()
    return (y_predict-y_actual)**2

def run_linlasso(data, window=30, alpha=0.01, skip=0):
    y_actual = data[window:,0]
    # define the Lasso model
    model = Lasso(alpha = alpha, normalize=True)
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
#        y_test = data[i+window,0]

        if skip_count>skip:
            model.fit(X_train, y_train[:,0])
            skip_count=1
        else:
            skip_count+=1

        y_predict.append(float(model.predict(X_test)))
        
    y_predict = np.array(y_predict)
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return (y_predict-y_actual)**2


def run_arima(data, window=30):
    y_actual = data[window:,0]
    y_predict=[]
    for i in range(len(data)-window):
        try:
            model = ARIMA(data[i:i+window,0], order=(1,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            y_predict.append(float(output[0]))
        except:
            y_predict.append(0)

    y_predict = np.array(y_predict)
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return (y_predict-y_actual)**2

def corr_stock(returns_df, top_num=1):
    corr_stock=[]
    cov_mat = np.cov(returns_df.iloc[:,1:].values.T)
    sID = returns_df.columns[1:]
    for i in range(len(returns_df.columns)-1):
        corr_top = sorted(cov_mat[:,i])[-1-top_num:]
        corr_top_list = [sID[int(np.where(cov_mat[:,i] == j)[0])] for j in corr_top]
        corr_stock.append(corr_top_list)
    return corr_stock

'''========================================================================='''
# Classification
def cf_accuracy(y_predict, y_actual):
    count=0
    for i in range(len(y_predict)):
        if y_predict[i]==y_actual[i]:
            count+=1
    return count/len(y_predict)

def run_LDA(data, window=30, skip=0):
    y_actual = data[window:,0]
    # define the LDA model
    LDA = LinearDiscriminantAnalysis()
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
    #    y_test = data[i+window,0]

#        X_mean = np.mean(X_train,axis=0)
#        X_std = np.std(X_train,axis=0)
#        X_train = (X_train-X_mean)/X_std
#        X_test = (X_test-X_mean)/X_std
    
        if skip_count>skip:
            LDA.fit(X_train, y_train[:,0])
            skip_count=1
        else:
            skip_count+=1
        y_predict.append(float(LDA.predict(X_test)))
        
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return acc

def run_QDA(data, window=30, skip=0):
    y_actual = data[window:,0]
    # define the LDA model
    QDA = QuadraticDiscriminantAnalysis()
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
#        y_test = data[i+window,0]

#        X_mean = np.mean(X_train,axis=0)
#        X_std = np.std(X_train,axis=0)
#        X_train = (X_train-X_mean)/X_std
#        X_test = (X_test-X_mean)/X_std

        if skip_count>skip:
            QDA.fit(X_train, y_train[:,0])
            skip_count=1
        else:
            skip_count+=1
        y_predict.append(float(QDA.predict(X_test)))
        
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return acc

def run_RF(data, window=30, RF_trees=100, skip=0):
    y_actual = data[window:,0]
    # define the RF model
    RF = RandomForestClassifier(n_estimators=RF_trees, random_state=0)
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
#        y_test = data[i+window,0]

#        X_mean = np.mean(X_train,axis=0)
#        X_std = np.std(X_train,axis=0)
#        X_train = (X_train-X_mean)/X_std
#        X_test = (X_test-X_mean)/X_std

        if skip_count>skip:
            RF.fit(X_train, y_train[:,0])
            skip_count=1
        else:
            skip_count+=1
        y_predict.append(float(RF.predict(X_test)))
        
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return acc

def run_LogitLasso(data, window=30, alpha=0.1, skip=0):
    y_actual = data[window:,0]
    # define the LR model
#    model = sm.Logit(y_train, X_train)
#    model.fit_regularized(alpha = sorted_scores[-1][0], disp=False)
    LR = LogisticRegression(random_state=0, penalty='l1', C=alpha)
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
#        y_test = data[i+window,0]

#        X_mean = np.mean(X_train,axis=0)
#        X_std = np.std(X_train,axis=0)
#        X_train = (X_train-X_mean)/X_std
#        X_test = (X_test-X_mean)/X_std

        if skip_count>skip:
            LR.fit(X_train, y_train[:,0])
            skip_count=1
        else:
            skip_count+=1

        y_predict.append(float(LR.predict(X_test)))
        
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return acc

def run_NN_cf(data, window=30, skip=0):
    y_actual = data[window:,0]
    # define the keras model
    model = Sequential()
    model.add(Dense(10, input_dim=data.shape[1]-1))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
    #    y_test = data[i+window,0]

        X_mean = np.mean(X_train,axis=0)
        X_std = np.std(X_train,axis=0)
        X_train = (X_train-X_mean)/X_std
        X_test = (X_test-X_mean)/X_std
    
        if skip_count>skip:
            model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=0)
            skip_count=1
        else:
            skip_count+=1
        
        y_predict.append(float(model.predict(X_test)))
        
    for j in range(len(y_predict)):
        if y_predict[j] >= 0.5:
            y_predict[j] = 1
        else:
            y_predict[j] = 0
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return acc

def run_NN_cf_3c(data, window=30, skip=0):
    y_actual = data[window:,0]
    # define the keras model
    model = Sequential()
    model.add(Dense(10, input_dim=data.shape[1]-1))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(3, activation='softmax'))
    # compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_predict=[]
    skip_count=1e4
    for i in range(len(data)-window):
        X_train = data[i:i+window,1:]
        y_train = np.array([data[i:i+window,0]]).T
        X_test = np.array([data[i+window,1:]])
    #    y_test = data[i+window,0]

        X_mean = np.mean(X_train,axis=0)
        X_std = np.std(X_train,axis=0)
        X_train = (X_train-X_mean)/X_std
        X_test = (X_test-X_mean)/X_std
    
        if skip_count>skip:
            model.fit(X_train, y=y_train, epochs=100, batch_size=50, verbose=0)
            skip_count=1
        else:
            skip_count+=1
        
        pred_prob = model.predict(X_test)[0,:]
        y_predict.append(int(np.where(pred_prob==max(pred_prob))[0]))
        
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
#    rmse = np.sqrt(mean_squared_error(y_predict, y_actual))
    return acc


def run_cf_benchmark(data, window=30):
    y_actual = data[window:,0]
    y_predict = data[window-1:-1,0]
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
    return acc

'''========================================================================='''
# Time Series Cross Validation (Regression)
def run_linear_lasso_cv(data, window=30, hold_out_ratio=1000):
    hold_out_test = data[hold_out_ratio:,:]
    training_set =  data[:hold_out_ratio,:]
   
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [{'lasso__alpha': np.logspace(1, -4, 50)}]
   
    scores = []
    for alpha in param_grid[0]['lasso__alpha']:
        cv_scores = []
        tra=[]
        tes=[]
        for train_index, test_index in tscv.split(training_set):
            tra.append(train_index)
            tes.append(test_index)
            #Define the training and validation sets
            df_train = training_set[train_index]
            df_validation = training_set[test_index]
            X_train, y_train = df_train[:,1:], df_train[:,0]
            X_validation, y_validation = df_validation[:,1:], df_validation[:,0]
   
            #Rescale the data by the training set
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_validation = scaler.transform(X_validation)
           
            #Traing the model with the penalty term  
            '''linear lasso'''
            model_type = Lasso(alpha = alpha)
            model = model_type.fit(X_train, y_train)
   
            #Record the score for that fold
            validation_score = model.score(X_validation,y_validation)
    #        validation_score = np.mean((model.predict(X_validation) - y_validation)**2)
            cv_scores.append(validation_score)
       
        #Average the CV scores and record
        scores.append([alpha, np.average(cv_scores)])
   
    sorted_scores = sorted(scores, key=lambda tup: tup[1])
    #print('Best Penalty Term: '+str(sorted_scores[-1][0])+'\nScore: '+str(sorted_scores[-1][1]))
   
    y_actual = hold_out_test[:,0]
    y_predict=[]
    for i in range(len(hold_out_test)):
        X_train = data[i+hold_out_ratio-window:i+hold_out_ratio,1:]
        y_train = np.array([data[i+hold_out_ratio-window:i+hold_out_ratio,0]]).T
        X_test = np.array([data[i+hold_out_ratio,1:]])
    #    y_test = data[i+window,0]
   
#        X_mean = np.mean(X_train,axis=0)
#        X_std = np.std(X_train,axis=0)
#        y_mean = np.mean(y_train)
#        y_std = np.std(y_train)
#        X_train = (X_train-X_mean)/X_std
#        y_train = (y_train - y_mean)/y_std
#        X_test = (X_test-X_mean)/X_std
       
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
           
        '''linear lasso'''
        model = Lasso(alpha = sorted_scores[-1][0])
        model.fit(X_train, y_train)
       
#        y_predict.append(float(model.predict(X_test))*y_std + y_mean)
        y_predict.append(float(model.predict(X_test)))
   
    y_predict = np.array(y_predict)

    return (y_predict-y_actual)**2

def run_RF_cv(data, window=30, hold_out_ratio=1000, skip=0):
    hold_out_test = data[hold_out_ratio:,:]
    training_set =  data[:hold_out_ratio,:]
   
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [{'lasso__alpha': np.linspace(1, 126, 30)}]
   
    scores = []
    for alpha in param_grid[0]['lasso__alpha']:
        cv_scores = []
        tra=[]
        tes=[]
        RF = RandomForestRegressor(n_estimators=20, random_state=0, min_samples_leaf=int(alpha))
        for train_index, test_index in tscv.split(training_set):
            tra.append(train_index)
            tes.append(test_index)
            #Define the training and validation sets
            df_train = training_set[train_index]
            df_validation = training_set[test_index]
            X_train, y_train = df_train[:,1:], df_train[:,0]
            X_validation, y_validation = df_validation[:,1:], df_validation[:,0]
   
            #Rescale the data by the training set
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_validation = scaler.transform(X_validation)
           
            #Traing the model with the penalty term
            '''RF'''
            model = RF.fit(X_train, y_train)
   
            #Record the score for that fold
            validation_score = model.score(X_validation,y_validation)
    #        validation_score = np.mean((model.predict(X_validation) - y_validation)**2)
            cv_scores.append(validation_score)
       
        #Average the CV scores and record
        scores.append([alpha, np.average(cv_scores)])
   
    sorted_scores = sorted(scores, key=lambda tup: tup[1])
    #print('Best Penalty Term: '+str(sorted_scores[-1][0])+'\nScore: '+str(sorted_scores[-1][1]))
   
    y_actual = hold_out_test[:,0]
    model_type = RandomForestRegressor(n_estimators=20, min_samples_leaf=int(sorted_scores[-1][0]), random_state=0)
    y_predict=[]
    skip_count=1e4
    for i in range(len(hold_out_test)):
        X_train = data[i+hold_out_ratio-window:i+hold_out_ratio,1:]
        y_train = np.array([data[i+hold_out_ratio-window:i+hold_out_ratio,0]]).T
        X_test = np.array([data[i+hold_out_ratio,1:]])
    #    y_test = data[i+window,0]
   
        #Rescale the data by the training set
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_test)
   
        '''RF'''
        if skip_count>skip:
            model = model_type.fit(X_train, y_train[:,0])
            skip_count=1
        else:
            skip_count+=1
       
        y_predict.append(float(model.predict(X_test)))
   
    y_predict = np.array(y_predict)
    return (y_predict-y_actual)**2
'''========================================================================='''

# Time Series Cross Validation (Classification)
def run_LogitLasso_cv(data, window=30, hold_out_ratio=1000, skip=0):
    hold_out_test = data[hold_out_ratio:,:]
    training_set =  data[:hold_out_ratio,:]
    
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [{'lasso__alpha': np.logspace(-2, 2, 50)}]
    
    scores = []
    for alpha in param_grid[0]['lasso__alpha']:
        cv_scores = []
        tra=[]
        tes=[]
        model_type = LogisticRegression(penalty='l1', C=1/alpha)
        for train_index, test_index in tscv.split(training_set):
            tra.append(train_index)
            tes.append(test_index)
            #Define the training and validation sets
            df_train = training_set[train_index]
            df_validation = training_set[test_index]
            X_train, y_train = df_train[:,1:], df_train[:,0]
            X_validation, y_validation = df_validation[:,1:], df_validation[:,0]
    
            #Rescale the data by the training set
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_validation = scaler.transform(X_validation)
            
            #Traing the model with the penalty term
            '''logistic lasso'''
    #        model_type = sm.Logit(y_train, X_train)
    #        model = model_type.fit_regularized(alpha = alpha, disp=False)
            model = model_type.fit(X_train, y_train)
    
            #Record the score for that fold
            validation_score = model.score(X_validation,y_validation)
    #        validation_score = np.mean((model.predict(X_validation) - y_validation)**2)
            cv_scores.append(validation_score)
        
        #Average the CV scores and record
        scores.append([alpha, np.average(cv_scores)])
    
    sorted_scores = sorted(scores, key=lambda tup: tup[1])
    #print('Best Penalty Term: '+str(sorted_scores[-1][0])+'\nScore: '+str(sorted_scores[-1][1]))
    
    y_actual = hold_out_test[:,0]
    model_type = LogisticRegression(random_state=0, penalty='l1', C=1/sorted_scores[-1][0])
    y_predict=[]
    skip_count=1e4
    for i in range(len(hold_out_test)):
        X_train = data[i+hold_out_ratio-window:i+hold_out_ratio,1:]
        y_train = np.array([data[i+hold_out_ratio-window:i+hold_out_ratio,0]]).T
        X_test = np.array([data[i+hold_out_ratio,1:]])
    #    y_test = data[i+window,0]
    
        #Rescale the data by the training set
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_test)
    
        '''logistic lasso'''
    #    model = sm.Logit(y_train, X_train)
    #    model.fit_regularized(alpha = sorted_scores[-1][0], disp=False)
        if skip_count>skip:
            model = model_type.fit(X_train, y_train)
            skip_count=1
        else:
            skip_count+=1
        
        y_predict.append(float(model.predict(X_test)))
    
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
    return acc

def run_RF_cv_clf(data, window=30, hold_out_ratio=1000, skip=0):
    hold_out_test = data[hold_out_ratio:,:]
    training_set =  data[:hold_out_ratio,:]
    
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [{'lasso__alpha': np.linspace(1, 126, 30)}]
    
    scores = []
    for alpha in param_grid[0]['lasso__alpha']:
        cv_scores = []
        tra=[]
        tes=[]
        RF = RandomForestClassifier(n_estimators=20, random_state=0, min_samples_leaf=int(alpha))
        for train_index, test_index in tscv.split(training_set):
            tra.append(train_index)
            tes.append(test_index)
            #Define the training and validation sets
            df_train = training_set[train_index]
            df_validation = training_set[test_index]
            X_train, y_train = df_train[:,1:], df_train[:,0]
            X_validation, y_validation = df_validation[:,1:], df_validation[:,0]
    
            #Rescale the data by the training set
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_validation = scaler.transform(X_validation)
            
            #Traing the model with the penalty term
            '''RF'''
            model = RF.fit(X_train, y_train)
    
            #Record the score for that fold
            validation_score = model.score(X_validation,y_validation)
    #        validation_score = np.mean((model.predict(X_validation) - y_validation)**2)
            cv_scores.append(validation_score)
        
        #Average the CV scores and record
        scores.append([alpha, np.average(cv_scores)])
    
    sorted_scores = sorted(scores, key=lambda tup: tup[1])
    #print('Best Penalty Term: '+str(sorted_scores[-1][0])+'\nScore: '+str(sorted_scores[-1][1]))
    
    y_actual = hold_out_test[:,0]
    model_type = RandomForestClassifier(n_estimators=20, min_samples_leaf=int(sorted_scores[-1][0]), random_state=0)
    y_predict=[]
    skip_count=1e4
    for i in range(len(hold_out_test)):
        X_train = data[i+hold_out_ratio-window:i+hold_out_ratio,1:]
        y_train = np.array([data[i+hold_out_ratio-window:i+hold_out_ratio,0]]).T
        X_test = np.array([data[i+hold_out_ratio,1:]])
    #    y_test = data[i+window,0]
    
        #Rescale the data by the training set
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = scaler.transform(X_test)
    
        '''RF'''
        if skip_count>skip:
            model = model_type.fit(X_train, y_train[:,0])
            skip_count=1
        else:
            skip_count+=1
        
        y_predict.append(float(model.predict(X_test)))
    
    y_predict = np.array(y_predict)
    acc = [0 if y_predict[i]==y_actual[i] else 1 for i in range(len(y_predict))]
    return acc

'''========================================================================='''
# Import datasets
returns_df = pd.read_excel('src/Returns_Clean.xlsx')
flows_df = pd.read_excel('src/Flows_Clean.xlsx')

'''========================================================================='''
'''Regression type 100 stocks'''
w=252
return_lag = 1
flow_lag = 0
MA_list = [5,10,20,30]
hist_ma = w
skip_num = 100
corr_top = 99
hold_out_ratio = 1200 #set as window size for no hold-out

corr_sID = corr_stock(returns_df, top_num=corr_top)
loop_num = len(returns_df.columns)-1
sID = returns_df.columns[1:]
err_linreg=[]
err_histmean=[]
err_linlasso=[]
err_arima=[]
err_LSTM=[]
err_RF=[]
err_onlineVI=[]
for i in tqdm(range(10)):
    try:
        data = x_small(returns_df, flows_df, stock_ID = sID[i], retlag=return_lag, flowlag=flow_lag)
#        data = x_small_MA(returns_df, flows_df, stock_ID = sID[i], retlag=return_lag, flowlag=flow_lag, ma_range=MA_list)
#        data = x_full(returns_df, flows_df, stock_target=sID[i], stock_list=corr_sID[i], retlag=return_lag, flowlag=flow_lag)
        err_linreg.append(run_linreg(data[hold_out_ratio-w:,:], window=w))
        err_histmean.append(run_histmean(data[hold_out_ratio-w:,:], window=w, ma=hist_ma))
#        err_linlasso.append(run_linlasso(data[hold_out_ratio-w:,:], window=w, skip=0))
        err_LSTM.append(run_LSTM(data[hold_out_ratio-w:,:], window=w, skip=skip_num))
        err_onlineVI.append(run_onlineVI("reparam", data[hold_out_ratio-w:,:], natural=True, window=w, win_aft=150, skip=skip_num))
        err_arima.append(run_arima(data[hold_out_ratio-w:,:], window=w))
        '''cross validation'''
        err_linlasso.append(run_linear_lasso_cv(data, window=w, hold_out_ratio=hold_out_ratio))
        err_RF.append(run_RF_cv(data, window=w, hold_out_ratio=hold_out_ratio, skip=skip_num))
    except:
        pass

err_linreg = np.array(err_linreg).T
err_histmean = np.array(err_histmean).T
err_linlasso = np.array(err_linlasso).T
err_LSTM = np.array(err_LSTM).T
err_onlineVI = np.array(err_onlineVI).T
err_arima = np.array(err_arima).T
err_RF = np.array(err_RF).T

# save as txt
fname = 'esttab/Regression'
np.savetxt(fname+'/mse_%s_%s.txt'%("linreg", "cstock"), np.mean(err_linreg,axis=1), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("histmean", "cstock"), np.mean(err_histmean,axis=1), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("linlasso", "cstock"), np.mean(err_linlasso,axis=1), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("LSTM", "cstock"), np.mean(err_LSTM,axis=1), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("onlineVI", "cstock"), np.mean(err_onlineVI,axis=1), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("arima", "cstock"), np.mean(err_arima,axis=1), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("RF", "cstock"), np.mean(err_RF,axis=1), fmt='%.4f') 

np.savetxt(fname+'/mse_%s_%s.txt'%("linreg", "ctime"), np.mean(err_linreg,axis=0), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("histmean", "ctime"), np.mean(err_histmean,axis=0), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("linlasso", "ctime"), np.mean(err_linlasso,axis=0), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("LSTM", "ctime"), np.mean(err_LSTM,axis=0), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("onlineVI", "ctime"), np.mean(err_onlineVI,axis=0), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("arima", "ctime"), np.mean(err_arima,axis=0), fmt='%.4f') 
np.savetxt(fname+'/mse_%s_%s.txt'%("RF", "ctime"), np.mean(err_RF,axis=0), fmt='%.4f') 

# read the results from txt
rname='esttab/Regression/LSTM_OLVI'
mse_linreg_cs = np.loadtxt(rname+'/mse_%s_%s.txt'%("linreg", "cstock")) 
mse_histmean_cs = np.loadtxt(rname+'/mse_%s_%s.txt'%("histmean", "cstock")) 
mse_linlasso_cs = np.loadtxt(rname+'/mse_%s_%s.txt'%("linlasso", "cstock")) 
mse_LSTM_cs = np.loadtxt(rname+'/mse_%s_%s.txt'%("LSTM", "cstock")) 
mse_onlineVI_cs = np.loadtxt(rname+'/mse_%s_%s.txt'%("onlineVI", "cstock")) 
mse_arima_cs = np.loadtxt(rname+'/mse_%s_%s.txt'%("arima", "cstock")) 
mse_RF_cs = np.loadtxt(rname+'/mse_%s_%s.txt'%("RF", "cstock")) 

mse_linreg_ct = np.loadtxt(rname+'/mse_%s_%s.txt'%("linreg", "ctime")) 
mse_histmean_ct = np.loadtxt(rname+'/mse_%s_%s.txt'%("histmean", "ctime")) 
mse_linlasso_ct = np.loadtxt(rname+'/mse_%s_%s.txt'%("linlasso", "ctime")) 
mse_LSTM_ct = np.loadtxt(rname+'/mse_%s_%s.txt'%("LSTM", "ctime")) 
mse_onlineVI_ct = np.loadtxt(rname+'/mse_%s_%s.txt'%("onlineVI", "ctime")) 
mse_arima_ct = np.loadtxt(rname+'/mse_%s_%s.txt'%("arima", "ctime")) 
mse_RF_ct = np.loadtxt(rname+'/mse_%s_%s.txt'%("RF", "ctime")) 

cum_rmse_linreg = np.cumsum(np.sqrt(mse_linreg_cs))
cum_rmse_histmean = np.cumsum(np.sqrt(mse_histmean_cs))
cum_rmse_linlasso = np.cumsum(np.sqrt(mse_linlasso_cs))
cum_rmse_LSTM = np.cumsum(np.sqrt(mse_LSTM_cs))
cum_rmse_onlineVI = np.cumsum(np.sqrt(mse_onlineVI_cs))
cum_rmse_arima = np.cumsum(np.sqrt(mse_arima_cs))
cum_rmse_RF = np.cumsum(np.sqrt(mse_RF_cs))

plt.figure()
plt.plot(cum_rmse_linreg, label="linreg")
plt.plot(cum_rmse_histmean, label="histmean")
plt.plot(cum_rmse_linlasso, label="linlasso")
plt.plot(cum_rmse_LSTM, label="LSTM")
plt.plot(cum_rmse_onlineVI, label="onlineVI")
plt.plot(cum_rmse_arima, label="arima")
plt.plot(cum_rmse_RF, label="RF")
plt.legend()
#
## Plotting the cumulative RMSE difference
plt.figure()
plt.plot(cum_rmse_histmean-cum_rmse_linreg)
plt.plot(cum_rmse_histmean-cum_rmse_linlasso)
plt.plot(cum_rmse_LSTM-cum_rmse_onlineVI)
'''========================================================================='''
'''Classification type 100 stocks'''
w=252
return_lag = 1
flow_lag = 1
MA_list = [5,10,20,30]
cl_quantile = [0.33, 0.66]
hist_ma = w
skip_num = 100
corr_top = 1
alpha=0.25
hold_out_ratio = 1200 #set as window size for no hold-out

corr_sID = corr_stock(returns_df, top_num=corr_top)
loop_num = len(returns_df.columns)-1
sID = returns_df.columns[1:]
err_benchmark=[]
err_LDA=[]
err_LogitLasso=[]
err_RF=[]
err_QDA=[]
err_NN=[]
for i in tqdm(range(loop_num)):
    try:
        data = x_small(returns_df, flows_df, stock_ID = sID[i], retlag=return_lag, flowlag=flow_lag)
#        data = x_small_MA(returns_df, flows_df, stock_ID = sID[i], retlag=return_lag, flowlag=flow_lag, ma_range=MA_list)
#        data = x_full(returns_df, flows_df, stock_target=sID[i], stock_list=corr_sID[i], retlag=return_lag, flowlag=flow_lag)
        data_clf = x_cf_quantile(data, quantile=cl_quantile, window=w)
#        data_clf = x_cf_updown(data)
        err_benchmark.append(run_cf_benchmark(data_clf[hold_out_ratio-w:,:], window=w))
        err_LDA.append(run_LDA(data_clf[hold_out_ratio-w:,:], window=w, skip=skip_num))
#        err_LogitLasso.append(run_LogitLasso(data_clf[hold_out_ratio-w:,:], window=w, skip=skip_num, alpha=alpha))
#        err_RF.append(run_RF(data_clf[hold_out_ratio-w:,:], window=w, skip=skip_num))
        err_QDA.append(run_QDA(data_clf[hold_out_ratio-w:,:], window=w, skip=skip_num))
#        err_NN.append(run_NN_cf(data_clf[hold_out_ratio-w:,:], window=w, skip=skip_num))
        '''quantile'''
        err_NN.append(run_NN_cf_3c(data_clf[hold_out_ratio-w:,:], window=w, skip=skip_num))
        '''cross validation'''
        err_LogitLasso.append(run_LogitLasso_cv(data_clf, window=w, hold_out_ratio=hold_out_ratio, skip=skip_num))
        err_RF.append(run_RF_cv_clf(data_clf, window=w, hold_out_ratio=hold_out_ratio, skip=skip_num))
    except:
        pass

err_benchmark = np.array(err_benchmark).T
err_LDA = np.array(err_LDA).T
err_LogitLasso = np.array(err_LogitLasso).T
err_RF = np.array(err_RF).T
err_QDA = np.array(err_QDA).T
err_NN = np.array(err_NN).T

# save as txt
fname = 'esttab/Classification'
np.savetxt(fname+'/acc_%s_%s.txt'%("benchmark", "cstock"), np.mean(err_benchmark,axis=1), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("LDA", "cstock"), np.mean(err_LDA,axis=1), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s_%s.txt'%("LogitLasso", "cstock", str(alpha)), np.mean(err_LogitLasso,axis=1), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("RF", "cstock"), np.mean(err_RF,axis=1), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("QDA", "cstock"), np.mean(err_QDA,axis=1), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("NN", "cstock"), np.mean(err_NN,axis=1), fmt='%.4f') 

np.savetxt(fname+'/acc_%s_%s.txt'%("benchmark", "ctime"), np.mean(err_benchmark,axis=0), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("LDA", "ctime"), np.mean(err_LDA,axis=0), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s_%s.txt'%("LogitLasso", "ctime", str(alpha)), np.mean(err_LogitLasso,axis=0), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("RF", "ctime"), np.mean(err_RF,axis=0), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("QDA", "ctime"), np.mean(err_QDA,axis=0), fmt='%.4f') 
np.savetxt(fname+'/acc_%s_%s.txt'%("NN", "ctime"), np.mean(err_NN,axis=0), fmt='%.4f') 

# read the results from txt
rname = 'esttab/Classification/3cluster_xsmall_lag1'
acc_benchmark_cs = np.loadtxt(rname+'/acc_%s_%s.txt'%("benchmark", "cstock")) 
acc_LDA_cs = np.loadtxt(rname+'/acc_%s_%s.txt'%("LDA", "cstock")) 
acc_LogitLasso_cs = np.loadtxt(rname+'/acc_%s_%s_%s.txt'%("LogitLasso", "cstock", str(alpha))) 
acc_RF_cs = np.loadtxt(rname+'/acc_%s_%s.txt'%("RF", "cstock")) 
acc_QDA_cs = np.loadtxt(rname+'/acc_%s_%s.txt'%("QDA", "cstock"))
acc_NN_cs = np.loadtxt(rname+'/acc_%s_%s.txt'%("NN", "cstock")) 

acc_benchmark_ct = np.loadtxt(rname+'/acc_%s_%s.txt'%("benchmark", "ctime")) 
acc_LDA_ct = np.loadtxt(rname+'/acc_%s_%s.txt'%("LDA", "ctime")) 
acc_LogitLasso_ct = np.loadtxt(rname+'/acc_%s_%s_%s.txt'%("LogitLasso", "ctime", str(alpha))) 
acc_RF_ct = np.loadtxt(rname+'/acc_%s_%s.txt'%("RF", "ctime")) 
acc_QDA_ct = np.loadtxt(rname+'/acc_%s_%s.txt'%("QDA", "ctime")) 
acc_NN_ct = np.loadtxt(rname+'/acc_%s_%s.txt'%("NN", "ctime"))


cum_acc_benchmark = np.cumsum(acc_benchmark_cs)
cum_acc_LDA = np.cumsum(acc_LDA_cs)
cum_acc_LogitLasso = np.cumsum(acc_LogitLasso_cs)
cum_acc_RF = np.cumsum(acc_RF_cs)
cum_acc_QDA = np.cumsum(acc_QDA_cs)
cum_acc_NN = np.cumsum(acc_NN_cs)

avg_scale = 1/np.linspace(1,len(cum_acc_benchmark), num=len(cum_acc_benchmark))
avg_acc_benchmark = cum_acc_benchmark*avg_scale
avg_acc_LDA = cum_acc_LDA*avg_scale
avg_acc_LogitLasso = cum_acc_LogitLasso*avg_scale
avg_acc_RF = cum_acc_RF*avg_scale
avg_acc_QDA = cum_acc_QDA*avg_scale
avg_acc_NN = cum_acc_NN*avg_scale


times = pd.to_datetime(returns_df['Dates'].iloc[-len(cum_acc_benchmark):], format='%Y%m%d')

plt.figure()
plt.plot(cum_acc_benchmark, label="benchmark")
plt.plot(cum_acc_LDA, label="LDA")
plt.plot(cum_acc_LogitLasso, label="LogitLasso")
plt.plot(cum_acc_RF, label="RF")
plt.plot(cum_acc_QDA, label="QDA")
plt.legend()

plt.figure()
plt.plot(times,1-avg_acc_benchmark, label="benchmark")
plt.plot(times,1-avg_acc_LDA, label="LDA")
plt.plot(times,1-avg_acc_LogitLasso, label="LogitLasso")
plt.plot(times,1-avg_acc_RF, label="RF")
plt.plot(times,1-avg_acc_QDA, label="QDA")
plt.plot(times,1-avg_acc_NN, label="NN")
plt.legend()

# Plotting the cumulative RMSE difference
plt.figure()
plt.plot(cum_acc_benchmark-cum_acc_LDA)

'''========================================================================='''

