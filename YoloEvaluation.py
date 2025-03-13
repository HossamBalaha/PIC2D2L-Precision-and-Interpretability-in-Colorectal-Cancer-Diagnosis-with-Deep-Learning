'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jan 1st, 2024
# Permissions and Citation: Refer to the README file.
'''

import cv2, os, tqdm
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import *

import warnings

warnings.filterwarnings('ignore')


def CalculateMetrics(references, predictions):
  cm = confusion_matrix(references, predictions)

  FP = cm.sum(axis=0) - np.diag(cm)
  FN = cm.sum(axis=1) - np.diag(cm)
  TP = np.diag(cm)
  TN = cm.sum() - (FP + FN + TP)

  if len(np.unique(references)) <= 2:
    FP = np.array(FP[0])
    FN = np.array(FN[0])
    TP = np.array(TP[0])
    TN = np.array(TN[0])

  FPStr = str(FP)
  FNStr = str(FN)
  TPStr = str(TP)
  TNStr = str(TN)

  eps = np.finfo(float).eps

  accuracy = (TP + TN + eps) / (TP + TN + FP + FN + eps)
  precision = (TP + eps) / (TP + FP + eps)
  recall = (TP + eps) / (TP + FN + eps)
  specificity = (TN + eps) / (TN + FP + eps)
  f1 = (2.0 * (precision * recall) + eps) / (precision + recall + eps)
  iou = (TP + eps) / (TP + FP + FN + eps)
  bac = (specificity + recall) / 2.0
  mcc = ((TP * TN) - (FP * FN) + eps) / np.sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
  )
  youden = recall + specificity - 1.0
  yule = (TP * TN - FP * FN + eps) / (TP * TN + FP * FN + eps)

  references = np.array(references)
  totalSamples = len(references)
  weights = np.array(
    [
      (np.sum(references == label) / totalSamples)
      for label in np.unique(references)
    ]
  )

  weightedAccuracy = np.sum(accuracy * weights)
  weightedPrecision = np.sum(precision * weights)
  weightedRecall = np.sum(recall * weights)
  weightedSpecificity = np.sum(specificity * weights)
  weightedF1 = np.sum(f1 * weights)
  weightedIoU = np.sum(iou * weights)
  weightedBAC = np.sum(bac * weights)
  weightedMCC = np.sum(mcc * weights)
  weightedYouden = np.sum(youden * weights)
  weightedYule = np.sum(yule * weights)

  avg = (
          weightedAccuracy
          + weightedPrecision
          + weightedRecall
          + weightedSpecificity
          + weightedF1
          + weightedBAC
          + weightedIoU
          + weightedYouden
          + weightedYule
          + weightedMCC
        ) / 10.0

  metrics = {
    "accuracy"    : weightedAccuracy,
    "precision"   : weightedPrecision,
    "recall"      : weightedRecall,
    "specificity" : weightedSpecificity,
    "f1"          : weightedF1,
    "iou"         : weightedIoU,
    "bac"         : weightedBAC,
    "mcc"         : weightedMCC,
    "youden"      : weightedYouden,
    "yule"        : weightedYule,
    "FP"          : FPStr,
    "FN"          : FNStr,
    "TP"          : TPStr,
    "TN"          : TNStr,
    "Sum"         : np.sum(cm),
    "Metrics_Mean": avg,
  }

  return metrics


if __name__ == "__main__":
  outputFolderName = "CK+48 - Output"
  datasetDirName = r"CK+48"
  runsFolder = r"runs-CK+48"
  baseDir = os.getcwd()
  extensions = ['tiff', 'tif', 'jpeg', 'jpg', 'png', 'bmp']
  inputShape = (100, 100)
  for cat in ["val", "test", "train"]:
    # modelKeyword = "yolov8s"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    for modelKeyword in ["yolov8n", "yolov8s", "yolov8m", "yolov8x", "yolov8l"]:  #
      weights = rf"{baseDir}\{runsFolder}\classify\{modelKeyword}-cls-{inputShape[0]}\weights\best.pt"
      model = YOLO(weights, task="classify", verbose=True).load(weights)
      print(model.names)

      history = []
      for k, v in tqdm.tqdm(model.names.items(), desc="Classes"):
        files = os.listdir(os.path.join(baseDir, outputFolderName, cat, v))
        for file in tqdm.tqdm(files[:], desc=v):
          fileExt = file.split(".")[-1]
          if (fileExt.lower() not in extensions):
            continue
          imgPath = os.path.join(baseDir, outputFolderName, cat, v, file)
          pred = model(imgPath, imgsz=inputShape[0], classes=model.names, verbose=False)
          toValues = pred[0].probs.data.cpu().numpy()
          history.append({
            "Image"          : imgPath,
            "Actual"         : v,
            "Predicted"      : model.names[pred[0].probs.top1],
            "Actual Index"   : k,
            "Predicted Index": pred[0].probs.top1,
            **{f"{model.names[i]} Prob": toValues[i] for i in range(len(toValues))}
          })

      df = pd.DataFrame(history)
      file = rf"{modelKeyword} Classification-{cat.capitalize()}.csv"
      df.to_csv(file, index=False)
      df = pd.read_csv(file)

      references = df["Actual Index"].values
      predictions = df["Predicted Index"].values

      metrics = CalculateMetrics(references, predictions)
      print(metrics)

# pred = model.predict(
#   [img],
#   save=False,
#   imgsz=inputShape[0],
#   classes=model.names,
#   # No verbose.
#   verbose=False,
# )
# print(cls, model.names[pred[0].probs.top1])
