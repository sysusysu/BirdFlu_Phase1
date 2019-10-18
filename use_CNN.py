### --------- make CNN ---------- ###

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


from sklearn.metrics import matthews_corrcoef, confusion_matrix, log_loss, mutual_info_score, f1_score
from sklearn.model_selection import StratifiedKFold



### Calculate TN, FN, TP, FP, accuracy, sensitivity, specificity and MCC
def manual_mcc(tp, fp, tn, fn):
    return((tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
def PNmetrics2(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    # print(m)
    TN = m[0][0]
    FP = m[0][1]
    FN = m[1][0]
    TP = m[1][1]
    accuracy = (TN+TP)/(TN+FP+FN+TP)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    # mcc = matthews_corrcoef(y_true, y_pred)
    mcc = manual_mcc(TP, FP, TN, FN)
    mi = mutual_info_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # print(mcc)
    return(dict(zip(["TN","FP","FN","TP","acc","sen","spe","MCC","MI","FScore"],[TN, FP, FN, TP, accuracy, sensitivity, specificity, mcc, mi, f1])))

def score_to_binary(y_pred):
    pred = np.zeros((len(y_pred)))
    pred[y_pred>0.5]=1
    return(pred)
### get the matrics of validating
### make prediction
### compare result
def DL_mcc(model, x_test,y_test):
    pred_score = model.predict(x_test)
    pred_score = pred_score.reshape(len(pred_score))
    pred = np.zeros((len(pred_score)))
    pred[pred_score>0.5]=1
    result = PNmetrics2(y_test,pred)
    return(pred, result)

### input the vector of structure of CNN, input shape; return compiled model. 
# s: vector to define the cnn
# shape: input shape
# n_dense: number of dense layers
def make_cnn(s, shape, regression = False, verbose = False):
    ### first # (20) numbers specify CNN
    ### the last # (6) specify dense layer
    n_dense = int((len(s) - 20)/2)
    cnn = np.array(s[0:20]).reshape(4,5)
    dense = np.array(s[20:]).reshape(n_dense,2)
    if verbose:
        print(cnn)
        print(dense)
    model = Sequential()
    for i in range(cnn.shape[0]):
        if int(cnn[i][0]) == 0:
            continue

        if len(model.layers) == 0:
            model.add(Conv2D(filters=int(cnn[i][0]), kernel_size=(int(cnn[i][1]),10), strides=(int(cnn[i][2]), 1), padding='same', input_shape=(shape)))
        else:
            model.add(Conv2D(filters=int(cnn[i][0]), kernel_size=(int(cnn[i][1]),10), strides=(int(cnn[i][2]), 1), padding='same'))
            
        #print("-- I : ",i, "; stride: ", cnn[i][4])
        model.add(LeakyReLU(alpha=0.2))
        #print(" -- cnn check: ", int(cnn[i][3]))
        model.add(MaxPooling2D(pool_size=(int(cnn[i][3]), 1), strides=(int(cnn[i][3]),1), padding='same'))
        model.add(Dropout(cnn[i][4]))
    model.add(Flatten())
    for i in range(dense.shape[0]):
        if dense[i][0] == 0:
            continue
        model.add(Dense(int(dense[i][0])))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dense[i][1]))
    ### last layer
    if regression:
        model.add(Dense(1, activation='relu'))
    else:
        model.add(Dense(1, activation='sigmoid'))

    if verbose:
        print(model.summary())
    return(model)

# Number of layers of CNN 1 4
# 0. Number of filters 32 256
# 1. Kernel size  2 5
# 2. Stride 1 3
# 3. Pooling size 1 4
# 4. Dropout probability 0.1 0.5
# Number of dense layers 1 3
# Number of neurons in dense layers 72 256

## n_struc: number of structures
## n_cnn: number of cnn_layer; p_cnn: number of parameters in cnn
## n_dense: number of dense layer; p_dense: number of parameters in dense
## shape: shape of input
def random_generate_cnn(seed, n_struc, n_cnn, n_dense, p_cnn, p_dense, shape, fix_dense=False):
    if seed is None:
        seed = np.random.randint(999999999)
    np.random.seed(seed)
    result = np.zeros((n_struc, n_cnn*p_cnn+n_dense*p_dense))
    print("-- Seed: ", seed)
    print("-- N of structure: ", n_struc, "; N of CNN: ", n_cnn, "; CNN p: ", p_cnn,"; N of dense: ", n_dense, "; Dense p: ", p_dense)
    ## decide number of CNN layers, decide number of dense layers
    n_cnn_layer = []
    n_dense_layers = []
    for i in range(n_struc):
        n_cnn_layer.append(np.random.randint(1, n_cnn+1))  ## upper bound is exclusive
        if not fix_dense:
            n_dense_layers.append(np.random.randint(1, n_dense+1)) 
        else: 
            n_dense_layers.append(n_dense)
        
    ## fill the vector: result
    i = 0 #index of structure
    while(True):
        for j in range(n_cnn_layer[i]):
            # filter 32-256
            result[i][0+j*5] = np.random.randint(32, 257) 
            # kernel size 2-5
            result[i][1+j*5] = np.random.randint(2, 6)  
            # stride 1-3
            result[i][2+j*5] = np.random.randint(1, 4)
            # pooling size 1-4
            result[i][3+j*5] = np.random.randint(1, 5)  
            # dropout 0.1-0.5
            result[i][4+j*5] = np.random.uniform(0.1, 0.5)
            
        for j in range(n_dense_layers[i]):
            ## neurons 72 256
            result[i][20+0+j*2] = np.random.randint(72, 257)  ## neurons
            ## dropout 0.1 0.5
            result[i][20+1+j*2] = np.random.uniform(0.1, 0.5) ## dropout
        dup = False    ## check duplication
          
        for ii in range(0, i):
            if np.array_equal(result[i], result[ii]):
                dup = True
                break
        ## check validation: check if shrink to zero or not. 
        
        if not check_cnn_valid(result[i], shape):
            continue
        if dup:
            continue
        else:
            i+=1
        if i == n_struc:
            break
    #print(result.shape)
    #print(n_cnn_layer)
    #print(n_dense_layers)
    return(result)

## check if a structure is valid:
## shape should not contain axis for samples
def check_cnn_valid(s, shape, fix_dense = True):
    a = shape[0]
    sum_a = 0
    
    for j in range(4):
        sum_a+=s[5*j]
        if s[5*j]>0:
            # stride
            if s[j*5 + 2] == 0 or s[j*5 + 3] == 0:
                print("-- Warning: /0", s[5*j:5*j+4])
            a = np.round(a/s[j*5 + 2])
            # pooling
            a = np.round(a/s[j*5 + 3])
    if fix_dense:
        sum_d = 0
        for j in range(20, len(s), 2):
            sum_d += s[j]
        if int(sum_d) == 0:
            return False
    if int(sum_a) == 0:
        return False
    if a<=1:
        return False
    else:
        return True
### evaluate CNN structures
def read_cnn_struc(struc_file, shape, regression = False, return_model=True):
    
    cnn_struc = []
    with open(struc_file, "r") as f:
        for line in f:
            line = line.rstrip()
            cnn_struc = list(map(float, line.split("\t")))
    f.close()
    print(len(cnn_struc))
    if return_model:
        model = make_cnn(cnn_struc, shape, regression = regression)
        return(model)
    else:
        return(cnn_struc)
## r: NN structure
## minimize the objective function
def cnn_objective_function(r, trainx, trainy, testx, testy, p=0.5):
    print("obj check: ",r)
    model = make_cnn(r, trainx.shape[1:], (len(r)-20)/2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trainx, trainy, epochs=10, batch_size=600)
    result = DL_mcc(model, testx, testy)
    ### pred on trainx
    pred_score = model.predict(trainx)
    pred_score = pred_score.reshape(len(pred_score))
    pred = np.zeros((len(pred_score)))
    pred[pred_score>0.5]=1
    train_mcc = matthews_corrcoef(trainy, pred)
    ### pred on testx
    pred_score = model.predict(testx)
    pred_score = pred_score.reshape(len(pred_score))
    pred = np.zeros((len(pred_score)))
    pred[pred_score>0.5]=1
    test_mcc = matthews_corrcoef(testy, pred)
    
    cost = -p*train_mcc - (1-p)*test_mcc
    return(cost)

def test_make_cnn():
    x = np.array([70, 4, 1, 0.4, 3, 210, 5, 2, 0.45, 3, 210, 5, 2, 0.45, 3, 0, 0, 0, 0, 3, 100, 0.5, 112, 0.2])
    print(len(x))
    model = make_cnn(x, [100,10,1])
    print(model.summary())

#####################################
# KFold model 
#
#
#####################################


## average the result from nfolds
# def merge_several_folds_mean(data, nfolds):
#     a = np.array(data[0])
#     for i in range(1, nfolds):
#         a += np.array(data[i])
#     a /= nfolds
#     return a

## train models with nfold cross validation
def run_cross_validation_train_models(train_data, train_target, model_struc, nfolds=10, nb_epoch = 200):
    # input image dimensions
    batch_size = 600
    
    yfull_train = dict()
    # kf = KFold(, n_folds=nfolds, shuffle=True, random_state=random_state)
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in skf.split(train_data, train_target):
        
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('-- Start KFold train number {} from {}'.format(num_fold, nfolds))
        print('---- Split train: ', len(X_train), len(Y_train))
        print('---- Split valid: ', len(X_valid), len(Y_valid))

        model = make_cnn(model_struc, train_data.shape[1:], verbose = False)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid))

        predictions_valid, summary = DL_mcc(model, X_valid, Y_valid)

        score = log_loss(Y_valid, predictions_valid)
        print('-- Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("-- Train: Log_loss independent avg: ", score)
    ytrue = train_target[list(yfull_train.keys())]
    pred = np.array(list(yfull_train.values()))
    binary_y = score_to_binary(pred.reshape(-1))
    # print(ytrue)
    summary = PNmetrics2(ytrue, binary_y)

    info_string = '-- loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return(info_string, models, summary)

def run_cross_validation_process_test(test_data, test_target, info_string, models):
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('-- Start KFold test number {} from {}'.format(num_fold, nfolds))
        test_prediction = model.predict(test_data, verbose=0)
        yfull_test.append(test_prediction)

    test_res = np.mean(np.array(yfull_test), axis = 0)
    # test_res = merge_several_folds_mean(yfull_test, nfolds)
    binary_y = score_to_binary(test_res.reshape(-1))
    # print(test_target)
    # print(binary_y)
    summary = PNmetrics2(test_target, binary_y)

    info_string = '-- Test: loss_' + info_string + '_folds_' + str(nfolds)
    return(summary)
def form_summary_string(train_summary, test_summary, keys = ["TP","FP","TN","FN",'acc','sen','spe','MCC']):
    s = ""
    ## output format:
    ## train size,train positive,train negative,test size,test positive,
    #test negative,train acc,train sen,train spe,train MCC,test acc,test sen,test spe,test MCC
    train_size = train_summary["TP"] + train_summary["FP"]+ train_summary["TN"]+ train_summary["FN"]
    train_pos = train_summary["TP"] + train_summary["FN"]
    train_neg = train_summary["TN"] + train_summary["FP"]

    test_size = test_summary["TP"]+test_summary["FP"]+test_summary["TN"]+test_summary["FN"]
    test_pos = test_summary["TP"] + test_summary["FN"]
    test_neg = test_summary["TN"] + test_summary["FP"]
    s += (str(train_size)+","+str(train_pos)+","+str(train_neg)+",")
    s += (str(test_size)+","+str(test_pos)+","+str(test_neg)+",")
    s += (str(train_summary['acc'])+","+str(train_summary['sen'])+","+str(train_summary['spe'])+","+str(train_summary['MCC'])+",")
    s += (str(test_summary['acc'])+","+str(test_summary['sen'])+","+str(test_summary['spe'])+","+str(test_summary['MCC'])+"\n")
    return(s)

def test_nfold_locally():
    X = np.random.rand(1000, 116, 10, 1)
    Y = np.random.randint(0,2,size=1000)
    xtrain = X[0:800]
    ytrain = Y[0:800]
    xtest = X[800:]
    ytest = Y[800:]
    #print(xtrain)
    #print(ytrain)
    ### toy model which runs fast
    s = [2,2.0,3.0,5.0,0.275294654758, 0.0,0.0,0.0,0.0,0.0, 256.0,2.0,2.0,1.0,0.1, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0]
    info_string, models, train_summary = run_cross_validation_train_models(xtrain, ytrain, s, nfolds=10, nb_epoch = 2)
    test_summary = run_cross_validation_process_test(xtest, ytest, info_string, models)
    # print("sklskllks--",train_summary)
    # print("sklskllks--",test_summary)
    s = form_summary_string(train_summary, test_summary)
    print(s)
if __name__ == "__main__":
    # test_make_cnn()
    test_nfold_locally()



