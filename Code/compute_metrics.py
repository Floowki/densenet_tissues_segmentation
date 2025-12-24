import cv2 
import os 
from tqdm import tqdm 
import numpy as np 


def compute_metrics(path_semanticGT, path_semanticPRED, dim) :
    #| Inputs : 
    #|   # path_semanticGT : path to the groundtruth masks (manually segmented)
    #|   # path_semanticPRED : path to the predicted masks (applied trained classifier)
    #|   # dim : dimension of the tiles in pixels (e.g. 512)

    #| Outputs : 
    #|   # ACC : global and per-class accuracy metrics 
    #|   # PRE : global and per-class precision metrics 
    #|   # REC : per-class recall metrics 

    ## Steps ## 

    # Define metrics (global and per-class)
    glob_acc = 0
    TP_tiss, FP_tiss = 0, 0
    TP_back, TN_back, FP_back, FN_back = 0, 0, 0, 0
    TP_neo, TN_neo, FP_neo, FN_neo = 0, 0, 0, 0
    TP_nneo, TN_nneo, FP_nneo, FN_nneo = 0, 0, 0, 0


    # Loop over tiles in test dataset folder 
    filenames = os.listdir(path_semanticPRED)
    tot_pxl = len(filenames) * dim * dim 

    for filename in tqdm(filenames, desc="Manually segmented masks loop", unit="tile"):
        maskGT_path = os.path.join(path_semanticGT, filename)
        maskPRED_path = os.path.join(path_semanticPRED, filename)

        mask_GT = cv2.imread(maskGT_path, cv2.IMREAD_GRAYSCALE)
        mask_PRED = cv2.imread(maskPRED_path, cv2.IMREAD_GRAYSCALE)

        ## Global metrics ##  
        # >> Accuracy (concerns the correct classifications)
        cc = (mask_GT == mask_PRED)
        glob_acc += np.sum(cc)
        ## >> Precision (tissues against background)
        tp_tiss = (mask_GT != 0) & (mask_PRED != 0)
        TP_tiss += np.sum(tp_tiss)
        fp_tiss = (mask_GT == 0) & (mask_PRED != 0)
        FP_tiss += np.sum(fp_tiss)

        ## Per-class metrics ##

        # >>> Background 
        tp_back = (mask_PRED == 0) & (mask_GT == 0)
        TP_back += np.sum(tp_back)
        tn_back = tp_tiss
        TN_back += np.sum(tn_back)
        fp_back = (mask_PRED == 0) & (mask_GT != 0)
        FP_back += np.sum(fp_back)
        fn_back = (mask_PRED != 0) & (mask_GT == 0)
        FN_back += np.sum(fn_back)

        # >>> Neoplastic 
        tp_neo = (mask_PRED == 2) & (mask_GT == 2)
        TP_neo += np.sum(tp_neo)
        tn_neo = (mask_PRED != 2) & (mask_GT != 2)
        TN_neo += np.sum(tn_neo) 
        fp_neo = (mask_PRED == 2) & (mask_GT != 2)
        FP_neo += np.sum(fp_neo)
        fn_neo = (mask_PRED != 2) & (mask_GT == 2)
        FN_neo += np.sum(fn_neo)

        # Non neoplastic 
        tp_nneo = (mask_PRED == 1) & (mask_GT == 1)
        TP_nneo += np.sum(tp_nneo)
        tn_nneo = (mask_PRED != 1) & (mask_GT != 1)
        TN_nneo += np.sum(tn_nneo) 
        fp_nneo = (mask_PRED == 1) & (mask_GT != 1)
        FP_nneo += np.sum(fp_nneo)
        fn_nneo = (mask_PRED != 1) & (mask_GT == 1)
        FN_nneo += np.sum(fn_nneo)


    ## Final operations for computing the metrics ##  
    # >> Global metrics 
    glob_acc /= tot_pxl
    glob_pre = TP_tiss / (TP_tiss + FP_tiss)
    
    # >> Per-class metrics 
    back_acc = (TP_back + TN_back) / (TP_back + TN_back + FP_back + FN_back)
    neo_acc = (TP_neo + TN_neo) / (TP_neo + TN_neo + FP_neo + FN_neo)
    nneo_acc = (TP_nneo + TN_nneo) / (TP_nneo + TN_nneo + FP_nneo + FN_nneo) 

    back_pre = TP_back / (TP_back + FP_back)
    neo_pre = TP_neo / (TP_neo + FP_neo)
    nneo_pre = TP_nneo / (TP_nneo + FP_nneo)

    back_rec = TP_back / (TP_back + FN_back)
    neo_rec = TP_neo / (TP_neo + FN_neo)
    nneo_rec = TP_nneo / (TP_nneo + FN_nneo)
    

    ACC = [glob_acc, back_acc, neo_acc, nneo_acc]
    PRE = [glob_pre, back_pre, neo_pre, nneo_pre]
    REC = [back_rec, neo_rec, nneo_rec]

    return ACC, PRE, REC

