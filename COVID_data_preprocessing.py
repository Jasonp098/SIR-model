import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import mean_squared_error


e = 1e-16
R_ = pd.read_csv("COVID-19-master/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
C_ = pd.read_csv("COVID-19-master/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
D_ = pd.read_csv("COVID-19-master/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
N = pd.read_csv("N.csv")


def region_cleaner(df1,df2) :
    Valid_regions = []
    for i in np.arange(df1.shape[0]):
        Valid_regions.append((df1["Province/State"][i], df1["Country/Region"][i]))
    df_blanks = []
    for i in range(df2.shape[0]):
        tmp = (df2["Province/State"][i],df2["Country/Region"][i])
        if tmp not in Valid_regions:
            df_blanks.append(i)
    df_indexes_to_keep = set(range(df2.shape[0])) - set(df_blanks)
    df_sliced = df2.take(list(df_indexes_to_keep))
    return df_sliced


C = region_cleaner(R_, C_)
D = region_cleaner(R_, D_)
R = region_cleaner(C_, R_)
C = C.sort_values(by=['Province/State'], ascending = True)
C = C.sort_values(by=['Country/Region'], ascending = True)
D = D.sort_values(by=['Province/State'], ascending = True)
D = D.sort_values(by=['Country/Region'], ascending = True)
R = R.sort_values(by=['Province/State'], ascending = True)
R = R.sort_values(by=['Country/Region'], ascending = True)
N = N.sort_values(by=['Province/State'], ascending = True)
N = N.sort_values(by=['Country/Region'], ascending = True)

num_dates = C.shape[1]-4
num_regions = C.shape[0]

N = N.to_numpy()
x = np.tile(N[:,2], (num_dates,1)).T
N = x.astype(np.float)
C = C.to_numpy()
C = C[:, 4:]
R = R.to_numpy()
R = R[:, 4:]
I = C-R
S = N-I-R

blank = np.zeros((num_regions, 1))
S_ = np.hstack((blank, S))
I_ = np.hstack((blank, I))
R_ = np.hstack((blank, R))
S__ = np.hstack((S, blank))
I__ = np.hstack((I, blank))
R__ = np.hstack((R, blank))
N__ = np.hstack((N, N[:,1:2]))

B = -(S__-S_)*(N__)/ ((S_+e)*(I_+e)) +e
G = (R__ - R_) / (I_+e) +e

B_smooth = np.zeros((B.shape[0], B.shape[1]+6), dtype = 'float64')

for i in range(7):
    B_smooth[:, i:B.shape[1]+i] = B_smooth[:, i:B.shape[1]+i] + B

B_smooth = B_smooth[:,3:B.shape[1]-3]
B_smooth /= 7.0
B = B_smooth

G_smooth = np.zeros((G.shape[0], G.shape[1]+14), dtype = 'float64')
for i in range(15):
    G_smooth[:, i:G.shape[1]+i] = G_smooth[:, i:G.shape[1]+i] + G
G_smooth = G_smooth[:,7:G.shape[1]-7]
G_smooth /= 15.0
G = G_smooth


def blank_appender(df,x):
    if x!=7 :
        pre = np.tile(blank,x-1)
        post = np.tile(blank,7-x)
        a, b, c =df
        a_ = np.hstack((pre,a,post))
        b_ = np.hstack((pre,b,post))
        c_ = np.hstack((pre,c,post))
    else :
        pre = np.tile(blank,6)
        a, b, c =df
        a_ = np.hstack((pre,a))
        b_ = np.hstack((pre,b))
        c_ = np.hstack((pre,c))
    return a_, b_, c_


S1, I1, R1 = blank_appender((S,I,R),1)
S2, I2, R2 = blank_appender((S,I,R),2)
S3, I3, R3 = blank_appender((S,I,R),3)
S4, I4, R4 = blank_appender((S,I,R),4)
S5, I5, R5 = blank_appender((S,I,R),5)
S6, I6, R6 = blank_appender((S,I,R),6)
S7, I7, R7 = blank_appender((S,I,R),7)
training_data = np.hstack((S1,S2,S3,S4,S5,S6,S7, I1,I2,I3,I4,I5,I6,I7, R1,R2-R1,R3-R2,R4-R3,R5-R4,R6-R5,R7-R6))

np.savetxt("training_Data.csv", training_data,delimiter = ',')

def Regionwide(key):
    num_pre = 15
    training_cut = training_data[key].reshape((21, num_dates+6))
    cache = (training_cut[0,0], training_cut[6,-1], training_cut[20,-1])
    training_cut[0:7] = (training_cut[0:7]-training_cut[6,-1])/(training_cut[0,0]-training_cut[6,-1])
    training_cut[7:14] = training_cut[7:14]/10000
    training_cut[14:21] = training_cut[14:21]/training_cut[20,-1]
    training_cut = training_cut[:,6+num_pre:-6]
    B_cut = B[key,num_pre:-1]
    G_cut = G[key,num_pre:-1]

    return training_cut.T, B_cut, G_cut, cache


def retrieve_original(df_r, cache):
    t0, t6, t20 = cache
    S_0 = df_r[0] * (t0-t6) + t6
    I_0 = df_r[7]*10000
    R_0 = df_r[14] * t20
    return (S_0, I_0, R_0)


def predict_future(df_r, model1, model2, cache):
    predicts = np.zeros((3,100))
    today = df_r[-1].reshape(1,-1)

    t0, t6, t20 = cache
    for i in range(100):
        print (i)
        b = model1.predict(today)
        b = (b+np.abs(b))/2.0
        g = model2.predict(today)
        g = (g+np.abs(g))/2.0
        s_p,i_p,r_p = retrieve_original(today.T, cache)
        print (b,g, s_p, i_p, r_p)
        s_f = s_p - b * s_p * i_p /(s_p+i_p+r_p)
        i_f = i_p + b * s_p * i_p /(s_p+i_p+r_p) - g * i_p
        r_f = r_p + g * i_p
        predicts[0,i] = s_f
        predicts[1,i] = i_f
        predicts[2,i] = r_f
        today_ = np.zeros((1,21))
        today_[:, 1:7] = today[:, 0:6]
        today_[:, 8:14] = today[:, 7:13]
        today_[:, 15:21] = today[:, 14:20]
        today_[:,15] = today[:,14]-r_f/t20
        today_[0,0] = (s_f - t6) / (t0-t6)
        today_[0,7] = i_f / 10000
        today_[0,14] = r_f / t20
        today = today_
        print (today)
    return predicts



def training_model(X,y, max_iters = 1000, mode=1):
    if mode == 1 :
        model = MLPRegressor(hidden_layer_sizes=(15, 10, 5), activation='tanh', solver='adam', alpha=0,
                             learning_rate='invscaling', learning_rate_init=1e-4, verbose=0, tol=1e-6, warm_start=True,
                             max_iter=1)
        X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=3213)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.25, random_state=5241)

    if mode == 2 :
        X = X[8:,:]
        y = y
        model = MLPRegressor(hidden_layer_sizes=(10,8,6,4,2), activation='tanh', solver='adam', alpha=0,
                             learning_rate='invscaling', learning_rate_init=3e-5, verbose=0, tol=1e-6, warm_start=True,
                             max_iter=1)
        X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.01, random_state=2311)
        X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.01, random_state=541515)

    is_overfit = False
    best_c_val_loss = 2020
    train_losses = []                                                                                   # Training_loss가 얼마인지 기록
    c_val_losses = []                                                                                   # C-val loss가 얼마인지 기록

    for i in range(max_iters):
        if i == 1 : tic = time.time()
        if i == 10 :
            toc = time.time()
            print ("Expected remaining time :" + str((round((toc-tic)*max_iters/10.0,2))) + "seconds") # TIME IS GOLD

        model.fit(X_train, y_train)                                              # Forward propagation 1번, Backward propagation 1번
        train_loss_ = mean_squared_error(y_train, model.predict(X_train))            # Train loss 얼마?
        c_val_loss = mean_squared_error(y_val, model.predict(X_val))                                       # C-val loss 얼마?
        train_losses.append(train_loss_)                                                               # 기록
        c_val_losses.append(c_val_loss)   # 기록
        if (i%1000 == 0):
            print ("Iteration #" + str(i) + " Training loss : "+ str(round(train_loss_,8))                 # 얼마인지 확인
                   +", Validation loss : " + str(round(c_val_loss,8)))

        if i > 50 and c_val_loss < best_c_val_loss :                                                   # Cross validation loss가 잘 감소하다가 증가한다면 training 중단
            model_to_keep = model
            is_overfit = True
            best_c_val_loss = c_val_loss

    if not is_overfit :
        print ("Underfit")
        model_to_keep = model

    acc = mean_squared_error(y_test, model_to_keep.predict(X_test))
    print (acc)
    return model_to_keep, train_losses, c_val_losses, i


def plot_learning_curve(Loss1, Loss2, num) :                                                           # Plot Learning curve
    x = list(range(1, num+2))
    plt.plot(x, Loss1, label="Train loss")
    plt.plot(x, Loss2, label="C-val loss")
    plt.xlabel("#Iters")
    plt.ylabel("Losses")
    plt.legend()
    plt.show(10)


key_r = 206                                           #141 korea 106 france 228 US
df_r, B_r, G_r, cache = Regionwide(key_r)
np.savetxt("B.csv",B_r, delimiter = ',')
np.savetxt("G.csv",G_r,delimiter = ',')
np.savetxt("df_r.csv",df_r,delimiter = ',')

model_B, train_losses, c_val_losses, num_iters = training_model(df_r, B_r, max_iters = 100000, mode = 1)
#plot_learning_curve (train_losses, c_val_losses, num_iters)
model_G, train_losses, c_val_losses, num_iters = training_model(df_r, G_r, max_iters = 100000, mode = 2)
#plot_learning_curve (train_losses, c_val_losses, num_iters)


b = model_B.predict(df_r)
b = (b+np.absolute(b)) /2.0
b_pred = np.vstack((B_r, b)).T
np.savetxt("Japan/"+str(key_r)+"B_pred.csv",b_pred, delimiter = ',')

g = model_G.predict(df_r)
g = (g+np.absolute(g))/2.0
g_pred = np.vstack((G_r, g[8:])).T

predicts = predict_future(df_r, model_B, model_G, cache)
np.savetxt("Japan/"+str(key_r)+"G_pred.csv",g_pred, delimiter = ',')
np.savetxt("Japan/"+str(time.time())+str(key_r)+"SIRs.csv",predicts, delimiter = ',')