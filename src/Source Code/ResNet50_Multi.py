import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import pandas as pd
import random
import keras
from keras import backend as K
from tensorflow.keras.metrics import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix , auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.platform import build_info
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.initializers import glorot_uniform, Zeros, RandomUniform
from skimage import io
from PIL import Image
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, concatenate, Lambda, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D,add, Dropout, Activation, Conv2DTranspose, multiply
from tensorflow.keras.layers import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"   # select a specific gpu if you have more than one


"""## Variable initialization """

input_size = 224
input_shape = (input_size, input_size, 3)
batchnum = 32
parent = os.path.dirname(os.path.dirname(os.getcwd()))  #  grand parent path

results_fold=[]
results_fold2=[]
results_fold3=[]
df_results5 = []
split = 1
n_epoch = 50
split_SEED = 42  
random_state = 77
img_row = img_col = input_size
Dropout_seed = 1
nfolds=5

runfold = 0

""" Optimizers """
opt_SGD = SGD(learning_rate= 0.002,  momentum=0.9)
opt_Adam = Adam(learning_rate= 0.00001)
opt_Nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

""" # Define cross validation function """
kfold = KFold(nfolds, shuffle=True, random_state=1)



"""# Reading image files"""

def read_images(images_names, path_imgs, labels, path_masks, input_size = 224):
    X = []
    Y_labels = []
    Y_masks = []
    
    for i in range(len(images_names)):
        
        image = cv2.imread(os.path.join(parent + path_imgs, images_names[i])+'.png')
        
        image = cv2.resize(image, (input_size, input_size)) 
        image = image/255
        X.append(image)
        Y_labels.append(labels[i])
        
        mask = cv2.imread(parent+path_masks + images_names[i]+'.png', 0)
        mask = cv2.resize(mask, (input_size, input_size), cv2.INTER_NEAREST)
        if len(np.unique(mask)) > 2:
            _,mask = cv2.threshold(mask,50,255,cv2.THRESH_BINARY)
        mask = mask/255
        Y_masks.append(mask)
    X = np.array(X).astype(np.float32)
    Y_labels = np.array(Y_labels).astype(np.float32)
    Y_masks = np.expand_dims(np.array(Y_masks), axis=3).astype(np.float32)
    
    print(X.dtype, X.shape)
    print(Y_labels.dtype, Y_labels.shape)
    print(Y_masks.dtype, Y_masks.shape)

    return X, Y_labels, Y_masks


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
    

"""# Evaluation Metrics for Classification task    """
def evaluate(df_pred,ROC_PLOT_FILE):
    cm = confusion_matrix(df_pred.Actual, df_pred.Pred)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total                          # Malignant  = 1    , Benign = 0
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) #recall
    specificity = cm[0, 0] / (cm[0, 1] + cm[0, 0]) 
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    f1scoreT_B = 2 * (precision * sensitivity) / (precision + sensitivity)
    rocAuc = roc_auc_score(df_pred.Actual, df_pred.Pred)            #TN, FP, FN, TP = cm.ravel()
    auc_roc_fun(df_pred.Actual, df_pred.Pred, ROC_PLOT_FILE)
    FPR = cm[0, 1]/(cm[0, 1] + cm[0, 0])
    FNR = cm[1, 0]/(cm[1, 0] + cm[1, 1])
    print("Acc: ", acc, "confusion mat =", cm)
    print("Recall(TP): ", sensitivity)
    print("Specificity(TN): ", specificity)
    
    return acc, sensitivity, specificity, f1scoreT_B, rocAuc, FPR, FNR 
    
    
"""# Evaluation Metrics for Segmentation task"""

def seg_metrics(y_true, y_pred): # for single image & pred # mask
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    recall = (tp + K.epsilon()) / (tp + fn + K.epsilon())  # tpr: recall(1-??? FN_rate)
    spe = (tn + K.epsilon()) / (tn + fp + K.epsilon())     # tnr: specificity (1-??? FP_rate)
    prec = (tp + K.epsilon()) / (tp + fp + K.epsilon())    # precision
    iou = (tp + K.epsilon()) / (tp + fn + fp + K.epsilon()) # intersection FNR union
    dsc = (2*tp + K.epsilon()) / (2*tp + fn + fp + K.epsilon()) # dice score
    
    return dsc, iou, recall, spe
    

"""# Loss Functions"""

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) 
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

    
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

def my_metric_fn(y_true, y_pred): # To monitor loss during training "mean squared difference"
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

# plotting the segmentation prediction    
def show_img_mask_pred(img, gt, pred, dsc, iou, rec, spe):
    fig=plt.figure(figsize=(11, 11))
    
    fig.add_subplot(1, 3, 1)
    plt.imshow(img, 'gray')
    plt.title('Image')
    plt.axis('off')
    
    fig.add_subplot(1, 3, 2)
    plt.imshow(gt,'gray')
    plt.title('GT')
    plt.axis('off')

    fig.add_subplot(1, 3, 3)
    plt.imshow(np.round(pred), 'gray') # threhold by 0.5
    plt.title('DSC=%0.2f,IOU=%0.2f,Rec=%0.2f,Spe=%0.2f' %(dsc,iou, rec, spe))
    plt.axis('off')
    
  
def pred_df(model, X, Y_label, Y_gt, ROC_testname, index = "epbest_Test", threshold = 0.5, show = False):
    
    probs, preds_seg = model.predict(X,verbose = 1)
    n_test = len(X)
    
    ################ 1) classification
    df_pred = pd.DataFrame()
    df_pred['Prob'] = probs.flatten()
    preds_label = (df_pred['Prob'] >= threshold).astype(int)
    df_pred['Pred'] = preds_label
    df_pred['Actual'] = Y_label
    
    acc, recall, spe, f1score, rocAuc ,  FPR, FNR= evaluate(df_pred, ROC_testname)
    
    ################ 2) segmentation
    dsc_list = np.zeros((n_test,1))
    iou_list = np.zeros_like(dsc_list)
    rec_list = np.zeros_like(dsc_list)
    spe_list = np.zeros_like(dsc_list)

    for i in range(n_test):
        dsc_list[i], iou_list[i], rec_list[i], spe_list[i] = seg_metrics(Y_gt[i], preds_seg[i] > threshold)
        if show:
            show_img_mask_pred(X[i][:,:,0] , Y_gt[i][:,:,0], preds_seg[i][:,:,0], dsc_list[i], iou_list[i], rec_list[i], spe_list[i])
    dsc_avg = round(np.mean(dsc_list), 3)*100 
    iou_avg = round(np.mean(iou_list), 3)*100  
    rec_avg = round(np.mean(rec_list), 3)*100 
    spe_avg = round(np.mean(spe_list), 3)*100
    #print('>>DSC \t\t{0:^.3f} \n>>IOU \t\t{1:^.3f} \n>>Rec \t{2:^.3f} \n>>Spe\t{3:^.3f}'.format(dsc_avg,iou_avg,rec_avg,spe_avg)) 

    confusion = confusion_matrix(Y_gt.ravel().astype(int), preds_seg.round().ravel().astype(int))
    acc_avg = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    acc_avg = round(acc_avg, 3)*100
    auc_avg = roc_auc_score(preds_seg.ravel()>threshold, Y_gt.ravel())
    auc_avg = round(auc_avg, 3)*100
    
    ################ merge into dataframe
    df_metrics = pd.DataFrame([[acc, recall, spe, dsc_avg, iou_avg, rec_avg, spe_avg, acc_avg, auc_avg]], 
                              columns= ['clf_acc', 'clf_rec', 'clf_spe','dsc','iou','rec', 'spe', 'acc', 'auc'], 
                              index = [index])
    return acc, recall, spe, f1score, rocAuc, FPR, FNR,  dsc_avg, iou_avg, rec_avg, spe_avg, acc_avg, auc_avg
    
""" ###   Custom Augmentation approach. """
gen = ImageDataGenerator()
def gen_flow_for_two_inputs(X1, y, X2):
    genX1 = gen.flow(X1,y,  batch_size=32, seed=666)
    genX2 = gen.flow(X2,y, batch_size=32, seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()

            yield X1i[0], [ X1i[1],X2i[0]]       

            
"""# Creating ResNet50 model"""

def create_encoder():
    inputs = Input(shape=input_shape)
    base_model = ResNet50(input_shape= input_shape, include_top = False, weights = "imagenet")
    
    ########## Classification
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.3)(x)
    x=Dense(64,activation='relu')(x) 
    clf_out =Dense(1,activation='sigmoid', name='C')(x)
    print(clf_out.name)
    #print("Summary = ", base_model.summary())  

    ######## Segmentation Decoder
    x = base_model.output
    up6a = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')( base_model.get_layer("pool4_relu").output), base_model.get_layer("pool3_relu").output], axis=3)
    conv6a = ConvBlock(up6a, 128, name='block6a_upsmaple')     

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6a), base_model.get_layer("pool2_relu").output], axis=3)
    conv6 = ConvBlock(up6, 64, name='block6_upsmaple')

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), base_model.get_layer("conv1/relu").output], axis=3)
    conv7 = ConvBlock(up7, 32, name='block7_upsmaple')

    up8 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7), base_model.get_layer("input_2").output], axis=3)
    conv8 = ConvBlock(up8, 16, name='block8_upsmaple')
    seg_out = Conv2D(1, (1, 1), activation='sigmoid', name ='S')(conv8)
    
    model = Model(inputs=base_model.inputs, outputs=[clf_out, seg_out])

    return model

def ConvBlock(in_fmaps, num_fmaps, name):
    # Inputs: feature maps, number of output feature maps
    conv_1 = Conv2D(num_fmaps, (3, 3), padding='same', name=name + '_conv1')(in_fmaps)
    conv_1 = BatchNormalization(name=name + '_bn1')(conv_1)
    conv_1 = Activation('relu', name=name + '_relu1')(conv_1)

    conv_2 = Conv2D(num_fmaps, (3, 3), padding='same', name=name + '_conv2')(conv_1)
    conv_2 = BatchNormalization(name=name + '_bn2')(conv_2)
    conv_out = Activation('relu', name=name + '_relu2')(conv_2)

    return conv_out
    
# loading and reading dataset
df_images = pd.read_csv(parent+'/Datasets/Multitask/Multitask.csv')
images_names = list(df_images['img name'])
images_labels = list(df_images['tumor type'])

path_imgs = '/Datasets/Multitask/imgs/'
path_masks ='/Datasets/Multitask/mask/'
print(df_images.shape, len(path_imgs), len(path_masks))


X, Y_labels, Y_masks = read_images(path_imgs, images_labels, path_masks, input_size)
Y = []

for i in range(len(X)):
    Y.append([Y_labels[i], Y_masks[i]])
print(len(Y))

"""# Training and Testing"""

for train_ix, test_ix in kfold.split(X):
    
    # Display the run number
    print('Run #', runfold+1)
    X_train, gt_train, label_train, X_test, gt_test, label_test = \
        X[train_ix], Y_masks[train_ix], Y_labels[train_ix], X[test_ix], Y_masks[test_ix], Y_labels[test_ix]
    
    X_train1, X_val, label_train1, label_val, gt_train1, gt_val = train_test_split(X_train, label_train, gt_train, 
                                                                                           test_size=0.2,random_state=split_SEED)
    # create the model 
    model = create_encoder()
    loss_list = {'C': wBCE, 'S': dice_loss}
    loss_weights = {'C': 3, 'S': 1}
    metrics_list = {'C': 'acc', 'S': dsc}   # Metric list to monitor while training, to view 
    
    callbacks = [ReduceLROnPlateau(factor=0.1, patience=15, min_lr=1e-06, verbose=1),
             
             ]     
    model.compile(optimizer=opt_Adam, loss=loss_list, loss_weights = loss_weights, metrics=metrics_list)


    gen_flow = gen_flow_for_two_inputs(X_train1, label_train1, gt_train1)

    history = model.fit(gen_flow,  epochs=n_epoch, callbacks=callbacks, steps_per_epoch = 200, verbose = 1, validation_data=(X_val,  [label_val, gt_val]),)
   
     # Roc auc curve names
    ROC_testnameL =  "ResNet50_testL_Fold%d"%(split)
    ROC_testnameB = "ResNet50_testB_Fold%d"%(split)
    
    print(" Printing the model at the last epoch")
    acc, recall, spe, f1score,  rocAuc, FPR, FNR,dsc_avg, iou_avg, rec_avg, spe_avg, acc_avg, auc_avg = pred_df(model, X=X_test, Y_label=label_test, Y_gt=gt_test, ROC_testname=ROC_testnameL, 
                                              index ="All_Test", threshold = 0.5, show = False)
    #####################
    # The results of the last epoch
    results_fold += [[acc, recall, spe, f1score,  rocAuc, FPR, FNR,dsc_avg, iou_avg, rec_avg, spe_avg, acc_avg, auc_avg]]
   
    # clear the session and delete the model
    tf.keras.backend.clear_session()
    del model
    runfold +=1 
    
df_results = pd.DataFrame(results_fold, columns= [ 'acc', 'recall', 'spe', 'f1score',  'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'])


# Average of five epochs 
df_results_avg=pd.DataFrame([[ df_results["acc"].mean(), df_results["recall"].mean(), df_results["spe"].mean(), df_results["f1score"].mean(), df_results["rocAuc"].mean(), df_results["FPR"].mean(), df_results["FNR"].mean(), df_results["dsc_avg"].mean(),df_results["iou_avg"].mean(),df_results["rec_avg"].mean(), df_results["spe_avg"].mean(), df_results["acc_avg"].mean(), df_results["auc_avg"].mean()]], 
                           columns= ['acc',  'recall', 'spe',  'f1score', 'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'], index = ['df_results_avg'])


# save to a csv file 
df_results_avg.to_csv('ResNet50.csv', index=False, header=True)