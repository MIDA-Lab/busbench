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

os.environ["CUDA_VISIBLE_DEVICES"]="0"   # select a specific gpu if you have more than one


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

""" Optimizers """
opt_SGD = SGD(learning_rate= 0.002,  momentum=0.9)
opt_Adam = Adam(learning_rate= 0.00001)
opt_Nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



nfolds=5
# Prepare cross validation
kfold = KFold(nfolds, shuffle=True, random_state=1)
runfold = 0


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
    
""" ###   Custom Augmentation for our multitask (ResESTAN) approach. """
gen = ImageDataGenerator(rotation_range=20,
				width_shift_range=0.2,
		height_shift_range=0.2,
		 horizontal_flip=True)
def gen_flow_for_two_inputs(X1, y, X2):
    genX1 = gen.flow(X1,y,  batch_size=32, seed=666)
    genX2 = gen.flow(X2,y, batch_size=32, seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()

            yield X1i[0], [ X1i[1],X2i[0]]    
            
            
"""# Creating ResESTAN model"""

def create_encoder():
    inputs = Input(shape=input_shape)
    base_model = ResNet50( input_shape= input_shape, include_top = False, weights = "imagenet") 
    #print("Summary = ", base_model.summary())
    ############
    #Block1
    #input from ResNet50
    xinput = base_model.get_layer("input_2").output               
    conv11 = base_model.get_layer("conv1_conv").output
    conv1N1 = Conv2D(32, (1, 1), activation='relu', padding='same')(xinput)
    conv1N1 = BatchNormalization()(conv1N1)
    conv1N2 = Conv2D(32, (1, 15), activation='relu', padding='same')(xinput )
    conv1N2 = Conv2D(32, (15, 1), activation='relu', padding='same')(conv1N2)
    conv1N2 = BatchNormalization()(conv1N2)
    conv1N1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1N1)
    conv1N2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1N2)
    conv1NM = concatenate([conv1N1, conv1N2], axis=3)
    conv1NM = Conv2D(32, (5, 5), activation='relu', padding='same')(conv1NM)
    conv1 = base_model.get_layer("conv1_relu").output
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)##B
    pool1NM = MaxPooling2D(pool_size=(2, 2))(conv1NM)

    #Block2
    conv22 = base_model.get_layer("conv2_block1_1_bn").output
    conv2N1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1NM)
    conv2N1 = BatchNormalization()(conv2N1)
    conv2N2 = Conv2D(64, (1, 13), activation='relu', padding='same')(pool1NM)
    conv2N2 = Conv2D(64, (13, 1), activation='relu', padding='same')(conv2N2)
    conv2N2 = BatchNormalization()(conv2N2)
    conv2N1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2N1)
    conv2N2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2N2)
    conv2NM = concatenate([conv2N1, conv2N2], axis=3)
    conv2NM = Conv2D(64, (5, 5), activation='relu', padding='same')(conv2NM)
    conv2 = base_model.get_layer("conv2_block3_out").output
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2NM = MaxPooling2D(pool_size=(2, 2))(conv2NM)

    #Block3
    conv33 = base_model.get_layer("conv3_block1_1_bn").output
    conv3N1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2NM)
    conv3N1 = BatchNormalization()(conv3N1)
    conv3N2 = Conv2D(128, (1,11), activation='relu', padding='same')(pool2NM)
    conv3N2 = Conv2D(128, (11, 1), activation='relu', padding='same')(conv3N2)
    conv3N2 = BatchNormalization()(conv3N2)
    conv3N1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3N1)
    conv3N2 = Conv2D(128, (1, 1), activation='relu', padding='same')(conv3N2)
    conv3NM = concatenate([conv3N1, conv3N2], axis=3)
    conv3NM = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3NM)
    conv3 = base_model.get_layer("conv3_block4_out").output
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3NM = MaxPooling2D(pool_size=(2, 2))(conv3NM)

    #Block4
    conv41 = base_model.get_layer("conv4_block1_1_bn").output
    conv4N1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3NM)
    conv4N1 = BatchNormalization()(conv4N1)
    conv4N2 = Conv2D(256, (1, 9), activation='relu', padding='same')(pool3NM)
    conv4N2 = Conv2D(256, (9, 1), activation='relu', padding='same')(conv4N2)
    conv4N2 = BatchNormalization()(conv4N2)
    conv4N1 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4N1)
    conv4N2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4N2)
    conv4NM = concatenate([conv4N1, conv4N2], axis=3)
    conv4NM = Conv2D(256, (1, 1), activation='relu', padding='same')(conv4NM)
    conv4 = base_model.get_layer("conv4_block6_out").output
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4NM = MaxPooling2D(pool_size=(2, 2))(conv4NM)

    #Block5
    conv51 = base_model.get_layer("conv5_block1_1_bn").output
    conv5N1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4NM)
    conv5N2 = Conv2D(512, (3, 7), activation='relu', padding='same')(pool4NM)
    conv5N2 = Conv2D(512, (7, 3), activation='relu', padding='same')(conv5N2)

    conv5N1 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5N1)
    conv5N2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5N2)
    conv5 = base_model.get_layer("conv5_block3_out").output

    conv5 = concatenate([conv5, conv51], axis=3)
    conv5NM = concatenate([conv5N1, conv5N2], axis=3)
    conv5NM = Conv2D(512, (1, 1), activation='relu', padding='same')(conv5NM)
    conv5 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv5NM], axis=3)
# Up Block 1
    up6 = concatenate([conv5, conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = concatenate([(conv6), conv41], axis=3)
    conv6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv4NM], axis=3)
    conv6 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
# Up Block 2
    up7 = concatenate([(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = concatenate([(conv7), conv33], axis=3)
    conv7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv3NM], axis=3)
    conv7 = Conv2D(128, (1, 1), activation='relu', padding='same' )(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same' )(conv7)
    
# Up Block 3
    up8 = concatenate([(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = concatenate([(conv8), conv22], axis=3)
    conv8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv2NM], axis=3)
    conv8 = Conv2D(64, (1, 1), activation='relu', padding='same' )(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same' )(conv8)
       
# Up Block 4       
    up9 = concatenate([(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = concatenate([(conv9), conv11], axis=3)
    conv9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9), conv1NM], axis=3)
#    print("shape === ", up9.shape, conv9.shape, conv11.shape)
    conv9 = Conv2D(32, (5, 5), activation='relu', padding='same' )(conv9)
    conv9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv11), conv9], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same' )(conv9)
# Final prediction , segmentation branch
    seg_out = Conv2D(1, (1, 1), activation='sigmoid', name='S')(conv9) 

#Classification Task branch
#################
    x=GlobalAveragePooling2D()(up6)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(128,activation='relu')(x) 
    clf_out =Dense(1,activation='sigmoid', name='C')(x)

#######################
    model = Model(inputs=base_model.input, outputs=[clf_out, seg_out])
    return model
    
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
    loss_weights = {'C': 1.5, 'S': 1}
    metrics_list = {'C': my_metric_fn, 'S': dsc}   # Metric list to monitor while training, to view 
    
    callbacks = [ReduceLROnPlateau(factor=0.1, patience=15, min_lr=1e-06, verbose=1),
             
             ModelCheckpoint('MTestan_best_ovall.h5', 
                              monitor='val_loss', mode='min', save_best_only=True, verbose=1),
             
             ModelCheckpoint("MTestan_best_clf.h5",
                              monitor='val_C_my_metric_fn', mode='min',  save_best_only=True, verbose=1)]     
    model.compile(optimizer=opt_Adam, loss=loss_list, loss_weights = loss_weights, metrics=metrics_list)

    gen_flow = gen_flow_for_two_inputs(X_train1, label_train1, gt_train1)

    history = model.fit(gen_flow,  epochs=n_epoch, callbacks=callbacks, steps_per_epoch = 210, verbose = 1, validation_data=(X_val,  [label_val, gt_val]),)
   
     # Roc auc curve names
    ROC_testnameL =  "MTESTAN_testL_Fold%d"%(split)
    ROC_testnameB = "MTESTAN_testB_Fold%d"%(split)
    ROC_testnameCls = "MTESTAN_testS_Fold%d"%(split)
    
    print(" Printing the model at the final epoch")
    acc, recall, spe, f1score,  rocAuc, FPR, FNR,dsc_avg, iou_avg, rec_avg, spe_avg, acc_avg, auc_avg = pred_df(model, X=X_test, Y_label=label_test, Y_gt=gt_test, ROC_testname=ROC_testnameL, 
                                              index ="All_Test", threshold = 0.5, show = False)
   
#####################
    print(" Printing the best model")
    model_best = load_model('MTestan_best_ovall.h5',
                           custom_objects={'dice_loss': dice_loss, 'dsc': dsc,'wBCE': wBCE,  'my_metric_fn': my_metric_fn})
    acc2, recall2, spe2, f1score2,  rocAuc2, FPR2, FNR2,dsc_avg2, iou_avg2, rec_avg2, spe_avg2, acc_avg2, auc_avg2 = pred_df(model_best, X=X_test, Y_label=label_test, Y_gt=gt_test, ROC_testname=ROC_testnameB, 
                                              index ="All_Test", threshold = 0.5, show = False)
    
    print(" Printing the best Classification model")                              
    model_clf = load_model('MTestan_best_clf.h5',
                            custom_objects={'dice_loss': dice_loss, 'dsc': dsc, 'wBCE': wBCE,  'my_metric_fn': my_metric_fn})
    acc4, recall4, spe4, f1score4,  rocAuc4, FPR4, FNR4,dsc_avg4, iou_avg4, rec_avg4, spe_avg4, acc_avg4, auc_avg4 = pred_df(model_clf, X=X_test, Y_label=label_test, Y_gt=gt_test, ROC_testname=ROC_testnameB, 
                                              index ="All_Test", threshold = 0.5, show = False)

    #####################
    # Final epochs
    results_fold += [[acc, recall, spe, f1score,  rocAuc, FPR, FNR,dsc_avg, iou_avg, rec_avg, spe_avg, acc_avg, auc_avg]]
    # The result best overall model
    results_fold2 += [[ acc2, recall2, spe2, f1score2,  rocAuc2, FPR2, FNR2,dsc_avg2, iou_avg2, rec_avg2, spe_avg2, acc_avg2, auc_avg2]]
    # The result of best Classification model
    results_fold3 += [[ acc4, recall4, spe4, f1score4,  rocAuc4, FPR4, FNR4,dsc_avg4, iou_avg4, rec_avg4, spe_avg4, acc_avg4, auc_avg4]]
    
    # clear the session and delete the model
    tf.keras.backend.clear_session()
    del model
    runfold +=1 
    
df_results = pd.DataFrame(results_fold, columns= [ 'acc', 'recall', 'spe', 'f1score',  'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'])
df_results2 = pd.DataFrame(results_fold2, columns= [ 'acc', 'recall', 'spe', 'f1score',  'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'])
df_results3 = pd.DataFrame(results_fold3, columns= [ 'acc', 'recall', 'spe', 'f1score',  'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'])

# Average of five epochs 
df_results_avg=pd.DataFrame([[ df_results["acc"].mean(), df_results["recall"].mean(), df_results["spe"].mean(), df_results["f1score"].mean(), df_results["rocAuc"].mean(), df_results["FPR"].mean(), df_results["FNR"].mean(), df_results["dsc_avg"].mean(),df_results["iou_avg"].mean(),df_results["rec_avg"].mean(), df_results["spe_avg"].mean(), df_results["acc_avg"].mean(), df_results["auc_avg"].mean()]], 
                           columns= ['acc',  'recall', 'spe',  'f1score', 'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'], index = ['df_results_avg'])
df_results_avg2=pd.DataFrame([[ df_results2["acc"].mean(), df_results2["recall"].mean(), df_results2["spe"].mean(), df_results2["f1score"].mean(), df_results2["rocAuc"].mean(), df_results2["FPR"].mean(), df_results2["FNR"].mean(), df_results2["dsc_avg"].mean(),df_results2["iou_avg"].mean(),df_results2["rec_avg"].mean(), df_results2["spe_avg"].mean(), df_results2["acc_avg"].mean(), df_results2["auc_avg"].mean()]], 
                           columns= ['acc',  'recall', 'spe',  'f1score', 'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'], index = ['df_results_avg2'])
df_results_avg3=pd.DataFrame([[ df_results3["acc"].mean(), df_results3["recall"].mean(), df_results3["spe"].mean(), df_results3["f1score"].mean(), df_results3["rocAuc"].mean(), df_results3["FPR"].mean(), df_results3["FNR"].mean(), df_results3["dsc_avg"].mean(),df_results3["iou_avg"].mean(),df_results3["rec_avg"].mean(), df_results3["spe_avg"].mean(), df_results3["acc_avg"].mean(), df_results3["auc_avg"].mean()]], 
                           columns= ['acc',  'recall', 'spe',  'f1score', 'rocAuc', 'FPR', 'FNR',  'dsc_avg', 'iou_avg', 'rec_avg', 'spe_avg', 'acc_avg', 'auc_avg'], index = ['df_results_avg3'])
# Concate the three dataframe 

res = pd.concat([df_results_avg, df_results_avg2, df_results_avg3],axis=0)  

# save to a csv file 
res.to_csv('MTESTAN.csv', index=False, header=True)