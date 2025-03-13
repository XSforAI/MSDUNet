import torch
import torch.nn.functional as F
import os, argparse
import seaborn as sns
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from sklearn.metrics import roc_curve, auc

from models.MSDUNet import MSDUFormer_Xnet_3463_FUESMS as model
from utils.dataloader import test_dataset

import numpy as np

def plot_roc_curve(Y, preds, save_path):
    fpr, tpr, thresholds = roc_curve(Y.ravel(), preds.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.show()


def calculate_metrics(Y_test, yp):
    jacard = 0
    dice = 0
    tanimoto = 0
    smooth = 1

    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        only_neg = y2 * (1 - yp_2)
        only_pos = (1 - y2) * yp_2

        if (np.sum(y2) == 0) and (np.sum(yp_2) == 0):
            tanimoto += 1.0
            jacard += 1.0
            dice += 1.0

        elif (np.sum(intersection) == 0):
            tanimoto += 0.0
            jacard += 0.0
            dice += 0.0

        else:
            tanimoto += ((np.sum(intersection) + smooth) / (
                        np.sum(intersection) + np.sum(only_neg) + np.sum(only_pos) + smooth))
            jacard += ((np.sum(intersection) + smooth) / (np.sum(union) + smooth))
            dice += (2. * np.sum(intersection) + smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)



    jacard /= len(Y_test)
    dice /= len(Y_test)
    tanimoto /= len(Y_test)

    return jacard, dice, tanimoto



def confusion_matrix_scorer(Y, Y_pred):
    Y = Y.astype(np.int8)
    Y_pred = Y_pred.astype(np.int8)
    P = len(np.where(Y == 1)[0])
    N = len(np.where(Y == 0)[0])

    # print([P, N])

    FP = len(np.where(Y - Y_pred == -1)[0])
    FN = len(np.where(Y - Y_pred == 1)[0])
    TP = len(np.where(Y + Y_pred == 2)[0])
    TN = len(np.where(Y + Y_pred == 0)[0])

    return P, N, TN, FP, FN, TP


def get_metrics(Y, pred):
    # Y = np.reshape(Y, pred.shape)
    smooth = 1
    P = 0
    N = 0
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    sensitivity = 0
    specificity = 0
    accuracy = 0
    precision = 0
    F1 = 0
    MCC = 0
    for i in range(len(Y)):
        _p, _n, _tn, _fp, _fn, _tp = confusion_matrix_scorer(Y[i], pred[i])
        P += _p
        N += _n
        TN += _tn
        FP += _fp
        FN += _fn
        TP += _tp
        if (np.sum(Y[i]) == 0) and (np.sum(pred[i]) == 0):
            sensitivity += 1
            specificity += 1
            precision += 1
            F1 += 1
            MCC += 1
        else:
            if (_tp == 0):
                sensitivity += 0
                precision += 0
                F1 += 0.0
            else:
                sensitivity += (_tp / (_tp + _fn))
                precision += (_tp / (_tp + _fp))
                F1 += (2 * ((_tp / (_tp + _fp)) * (_tp / (_tp + _fn))) / ((_tp / (_tp + _fp)) + (_tp / (_tp + _fn))))
            if (_tn == 0):
                specificity += 0
            else:
                specificity += (_tn / (_tn + _fp))
            MCC += (_tp * _tn - _fp * _fn + smooth) / (
                        np.power((_tp + _fp) * (_tp + _fn) * (_tn + _fp) * (_tn + _fn), 0.5) + smooth)
        accuracy += ((_tp + _tn) / (_tp + _fn + _fp + _tn))

    return P, N, TN, FP, FN, TP, sensitivity / len(Y), specificity / len(Y), accuracy / len(Y), precision / len(
        Y), F1 / len(Y), MCC / len(Y)


def get_metrics_and_print(Y, yp, method="PVT-GCASCADE", testset='test', threshold=0.5, show=False, write=False):
    rs = []
    # yp = preds_test >= threshold #np.round(preds_test,0)

    P, N, TN, FP, FN, TP, sensitivity, specificity, accuracy, precision, f1, mcc_cal = get_metrics(Y, yp)
    jacard, dice, tanimoto= calculate_metrics(Y, yp)

    cmat = [[TN, FP], [FN, TP]]
    cmat_score = [[TN / N, FP / N], [FN / P, TP / P]]
    print(cmat)
    if show:
        plt.figure(figsize=(6, 6))
        sns.heatmap(cmat_score, cmap="Reds", annot=True, fmt='.2%', square=1, linewidth=2.)  # cmat/np.sum(cmat)
        plt.xlabel("Predictions")
        plt.ylabel("True values")
        plt.show()

    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", sensitivity)
    print("F1 Score: ", f1)
    print("MCC: ", mcc_cal)
    print('Dice: ', dice)
    print('Jacard: ', jacard)
    print('Tanimoto: ', tanimoto)

    if (write):
        results = pd.DataFrame([[method, TN, FP, FN, TP, jacard,
                                 dice, sensitivity, specificity,
                                 accuracy, precision, f1, mcc_cal]],
                               columns=['Method', 'TN', 'FP', 'FN', 'TP', 'mIoU/Jacard', 'DICE',
                                        'Sensitivity/Recall', 'Specificity', 'Accuracy', 'Precision',
                                        'F-score', 'MCC'])
        results.to_csv('Repeat_results_' + testset + '.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    method_name = 'HAM10000_MSDUNet_352_2_B8_aug'
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='PVT', help='Name of encoder: PVT or MERIT')
    parser.add_argument("--num_classes", type=int, default=1, help="output channel of network")
    parser.add_argument('--skip_aggregation', type=str, default='additive',
                        help='Type of skip-aggregation: additive or concatenation')
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_out/HAM10000/' + method_name + '/' + method_name + '-best.pth')
    parser.add_argument('--img_size', type=int, default=224, help='training dataset size')
    args = parser.parse_args()

    # torch.cuda.set_device(0)  # set your gpu device

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = model(num_classes=args.num_classes).cuda(0)
    x = torch.load(args.pth_path)
    model.load_state_dict(x)

    model.eval()
    # dataset_name = 'PH2'
    dataset_name = 'HAM10000'
    testset = 'HAM10000_' + dataset_name
    for _data_name in ['test']:
        if dataset_name == 'HAM10000':
            ##### put data_path here #####
            data_path = 'E:/Workspace/Dr/data/HAM10000/{}'.format(_data_name)

            ##### save_path #####
            save_path = './model_out/HAM10000/' + method_name + '/HAM10000_{}/'.format(_data_name)
        elif dataset_name == 'PH2':
            data_path = 'E:/Workspace/Dr/data/PH2/{}'.format(_data_name)

            ##### save_path #####
            save_path = './model_out/HAM10000/' + method_name + '/PH2_{}/'.format(_data_name)


        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('Evaluating ' + data_path)

        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, args.testsize)
        DSC = 0.0
        JACARD = 0.0
        preds = []
        gts = []
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)  # forward

            # eval Dice
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res * 255)

            input = np.where(res >= 0.5, 1, 0)
            target = np.where(np.array(gt) >= 0.5, 1, 0)

            preds.append(input)
            gts.append(gt)

            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            union = input_flat + target_flat - intersection

            jacard = ((np.sum(intersection) + smooth) / (np.sum(union) + smooth))
            jacard = '{:.4f}'.format(jacard)
            jacard = float(jacard)
            JACARD += jacard

            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC += dice

        print('*****************************************************')
        print('Dice Score: ' + str(DSC / num1))
        print('Jacard Score: ' + str(JACARD / num1))
        get_metrics_and_print(gts, preds, method=method_name, testset=testset + _data_name, write=True)
        # plot_roc_curve(np.array(gts), np.array(preds), save_path + 'roc_curve.png')
        print(_data_name, 'Finish!')
        print('*****************************************************')
