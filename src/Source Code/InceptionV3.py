import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
from platform import python_version
import keras
from keras import backend as K
from tensorflow.keras.metrics import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, auc 
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
from tensorflow.python.platform import build_info
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.initializers import glorot_uniform, Zeros, RandomUniform
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Lambda, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D,add, Dropout, Activation, Conv2DTranspose, multiply
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]="0"   # select a specific gpu if you have more than one


"""## Variable initialization """

input_size = 299
batch_size = 32
split = 1
epoch_num = 50
random_state = 77
foldpath = "/Datasets/BUS_Combined/folds/"             # path to the 5-folds where each fold contains train, test, and validation 
parent = os.path.dirname(os.path.dirname(os.getcwd()))  #  grand parent path
input_shape = (input_size, input_size, 3)
img_row = img_col = input_size
Dropout_seed = 1
opt_SGD = SGD(learning_rate= 0.002,  momentum=0.9)
opt_Adam = Adam(learning_rate= 0.00001)
opt_Nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

"""# Reading image files"""

def read_images(paths, labels, input_size = 224):
    data = []
    GT = []
    # Path to images
    Dbpath ='/Datasets/BUS_Combined/imgs/'
    for i in range(len(paths)):
        image = cv2.resize(cv2.imread(os.path.join(parent+Dbpath,paths[i]+".png")), (input_size, input_size))
        data.append(image)
        GT.append(labels[i])
    data = np.array(data)
    GT = np.array(GT).astype(np.float32)
    print(data.dtype, data.shape)
    print(GT.dtype, GT.shape)
    return data, GT


"""# Evaluations """

def auc_roc_fun(Y_test,Y_pred,ROC_PLOT_FILE):
    folder = "plots"
    roc_log = roc_auc_score(Y_test, Y_pred)
    false_positive_rate, true_positive_rate, threshold = roc_curve(Y_test, Y_pred)
    area_under_curve = auc(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('{}/{}'.format(folder, ROC_PLOT_FILE), dpi=300, pad_inches=0.1)
    #plt.show()
    plt.close()
    
def evaluate(df_pred,ROC_PLOT_FILE):
    cm = confusion_matrix(df_pred.Actual, df_pred.Pred)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total                          # Malignant  = 1    , Benign = 0
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) #recall
    specificity = cm[0, 0] / (cm[0, 1] + cm[0, 0]) 
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    f1scoreT_B = 2 * (precision * sensitivity) / (precision + sensitivity)
    rocAuc = roc_auc_score(df_pred.Actual, df_pred.Pred)    #TN, FP, FN, TP = cm.ravel()
    auc_roc_fun(df_pred.Actual, df_pred.Pred,ROC_PLOT_FILE)
    FPR = cm[0, 1]/(cm[0, 1] + cm[0, 0])
    FNR = cm[1, 0]/(cm[1, 0] + cm[1, 1])
    print("Acc: ", acc, "confusion mat =", cm)
    print("Recall(TP): ", sensitivity)
    print("Specificity(TN): ", specificity)
    
    return acc, sensitivity, specificity, f1scoreT_B, rocAuc, FPR, FNR

""" # prediction method  """

def predictions(model, X, Y, threshold = 0.5):
    probs = model.predict(X, verbose = 1)
    df_pred = pd.DataFrame()
    df_pred['Prob'] = probs.flatten()
    Y_pred = (df_pred['Prob'] >= threshold).astype(int)
    df_pred['Pred'] = Y_pred
    df_pred['Actual'] = Y.values
    return df_pred
    
"""# Augmentation"""

def my_generator(X_train, Y_train, batch_size):
    data_generator = ImageDataGenerator(rotation_range=20,
				width_shift_range=0.2,
		height_shift_range=0.2,
		 horizontal_flip=True ).flow(X_train, Y_train, batch_size = batch_size, seed=1)
    while True:
        x_batch, y_batch = data_generator.next()
        yield x_batch, y_batch

"""# Loss Functions"""

def wBCE(y_true, y_pred, pos = 3):
    epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    pos = tf.constant(pos, tf.float32)
    cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred,pos)
    return K.mean(cost, axis=-1)    
def focal_loss(gamma=2., alpha=.5):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed 

def tversky_loss(beta):
  def loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

  return loss

"""# Creating the model """

def create_encoder():
    kernel_init = glorot_uniform(seed=1)
    conv =InceptionV3(input_shape= input_shape, include_top = False, weights = 'imagenet')
    x=conv.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu', kernel_initializer='he_normal' )(x)
    x=Dropout(0.3, seed = Dropout_seed)(x)
    x=Dense(64,activation='relu', kernel_initializer='he_normal' )(x)
    #x=Dropout(0.3, seed = Dropout_seed)(x)  
    preds=Dense(1,activation='sigmoid')(x)

    model =Model(inputs=conv.input, outputs=preds, name="BusBench")
    return model

"""# Training and Testing"""

epBest_metrics_split=[]

for i in range(5):

    train1 = pd.read_csv(os.path.join(parent+foldpath, 'folder'+str(i+1), 'train.csv'))
    valid1 = pd.read_csv(os.path.join(parent+foldpath, 'folder'+str(i+1), 'valid.csv'))
    test1 = pd.read_csv(os.path.join(parent+foldpath, 'folder'+str(i+1), 'test.csv'))
    # Reading training from pre splitted folder
    images_paths_Train = list(train1['img name'])
    images_labels_Train = list(train1['tumor types'])
    # Reading test set from pre splitted folder
    images_paths_Test = list(test1['img name'])
    images_labels_Test = list(test1['tumor types'])
    # Reading validation from pre splitted folder
    images_paths_Valid = list(valid1['img name'])
    images_labels_Valid = list(valid1['tumor types'])
    
    print( len(images_paths_Train), len(images_labels_Train),len(images_paths_Test), len(images_labels_Test))
    
    X_Train, Y_Train = read_images(images_paths_Train, images_labels_Train, input_size)
    #print("X test......................")
    X_Test, Y_Test = read_images(images_paths_Test, images_labels_Test, input_size)
    X_Valid, Y_Valid = read_images(images_paths_Valid, images_labels_Valid, input_size)
    
    #X_processed = preprocess_input(X) 
    Y_Tn = pd.Series(Y_Train)
    Y_Tst = pd.Series(Y_Test)
    Y_V = pd.Series(Y_Valid)

    print("Fold", split)
    
    gen = my_generator(X_Train, Y_Tn, batch_size)

    # create the model 
    model = create_encoder()
    
    model.compile(optimizer=opt_SGD,loss=wBCE,metrics=[tf.keras.metrics.Recall(), 'acc'])
    
    
    # save models based on min loss
    callbacks = [ModelCheckpoint("InceptionV3_Fold%d.h5"%(split), 
                                 monitor='val_loss', mode='min',  save_best_only=True, verbose=1)] 
    
    history = model.fit(gen, steps_per_epoch=len(X_Train)/batch_size, epochs=epoch_num, callbacks=callbacks, 
                        validation_data=(X_Valid ,Y_V), verbose = 1)
    
    print('---Load Model--- ')
    model_load = load_model("InceptionV3_Fold%d.h5"%(split),compile=False)
   
   # Roc auc curve names
    ROC_testnameL =  "InceptionV3_testL_Fold%d"%(split)
    ROC_testnameB = "InceptionV3_testB_Fold%d"%(split)
   
   # Test prediction on the last iteration in each epoch
    df_pred_last = predictions(model, X_Test, Y_Tst)

    # Test prediction on the best saved models in each fold
    df_pred_test = predictions(model_load, X_Test, Y_Tst)

    accT_B, recallT_B, speT_B, f1scoreT_B, rocT_B, FPRT_B, FNRT_B = evaluate(df_pred_test,ROC_testnameB)
    accL, recallL, speL, f1scoreL, rocL, FPRL, FNRL= evaluate(df_pred_last,ROC_testnameL)

    epBest_metrics_split += [[ accT_B, recallT_B, speT_B, f1scoreT_B, rocT_B, FPRT_B, FNRT_B,  accL, recallL, speL, f1scoreL, rocL, FPRL, FNRL]]
    
    split += 1


"""# Saving results to a csv file"""

df_epBest = pd.DataFrame(epBest_metrics_split, columns= ['accT_B', 'recallT_B', 'speT_B', 'f1scoreT_B', 'rocT_B', 'FPRT_B', 'FNRT_B', 'accL', 'recallL', 'speL', 'f1scoreL', 'rocL', 'FPRL', 'FNRL'])



AvgMetrics=pd.DataFrame([[df_epBest["accT_B"].mean(), df_epBest["recallT_B"].mean(), df_epBest["speT_B"].mean(), df_epBest["f1scoreT_B"].mean(), df_epBest["rocT_B"].mean(), df_epBest["FPRT_B"].mean(),df_epBest["FNRT_B"].mean(),
                            df_epBest["accL"].mean(), df_epBest["recallL"].mean(), df_epBest["speL"].mean(),df_epBest["f1scoreL"].mean(),df_epBest["rocL"].mean(),df_epBest["FPRL"].mean(),df_epBest["FNRL"].mean()]], 
                           columns= ['accT_Best',  'recallT_Best', 'speT_Best', 'f1score_Best', 'roc_Best',  'FPR_Best','FNR_Best','acc_Last','recall_Last','spe_Last','f1score_Last', 'roc_Last', 'FPR_Last','FNR_Last' ],
                           index = ['epBest_metrics_split'])
                           
# Save evaluation results to a csv file
AvgMetrics.to_csv('InceptionV3.csv', index=False, header=True)