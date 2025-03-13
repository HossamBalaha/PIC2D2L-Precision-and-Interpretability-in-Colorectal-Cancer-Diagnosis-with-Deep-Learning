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

import os
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def CalculateMetrics(references, predictions):
  cm = confusion_matrix(references, predictions)

  FP = cm.sum(axis=0) - np.diag(cm)
  FN = cm.sum(axis=1) - np.diag(cm)
  TP = np.diag(cm)
  TN = cm.sum() - (FP + FN + TP)

  if (len(np.unique(references)) <= 2):
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
  f1 = (2.0 * (precision * recall) + eps) / (
    precision + recall + eps)
  iou = (TP + eps) / (TP + FP + FN + eps)
  bac = (specificity + recall) / 2.0
  mcc = ((TP * TN) - (FP * FN) + eps) / np.sqrt(
    (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps
  )
  youden = recall + specificity - 1.0
  yule = (TP * TN - FP * FN + eps) / (TP * TN + FP * FN + eps)

  # Calculate Mutual Information (MI)
  # Compute joint probability distribution
  joint_prob = np.zeros((len(np.unique(references)), len(np.unique(predictions))))
  for i, true_label in enumerate(np.unique(references)):
    for j, pred_label in enumerate(np.unique(predictions)):
      joint_prob[i, j] = np.sum((references == true_label) & (predictions == pred_label)) / len(references)

  # Compute marginal probabilities
  p_x = np.sum(joint_prob, axis=1)  # P(x_i)
  p_y = np.sum(joint_prob, axis=0)  # P(y_j)

  # Compute Mutual Information
  mutualInfo = 0
  for i, true_label in enumerate(np.unique(references)):
    for j, pred_label in enumerate(np.unique(predictions)):
      if (joint_prob[i, j] > 0):
        mutualInfo += joint_prob[i, j] * np.log(joint_prob[i, j] / (p_x[i] * p_y[j]))

  references = np.array(references)
  totalSamples = len(references)
  weights = np.array(
    [(np.sum(references == label) / totalSamples) for label in
     np.unique(references)]
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

  metrics = {
    "Accuracy"   : np.round(weightedAccuracy * 100, 2),
    "Precision"  : np.round(weightedPrecision * 100, 2),
    "Recall"     : np.round(weightedRecall * 100, 2),
    "Specificity": np.round(weightedSpecificity * 100, 2),
    "F1"         : np.round(weightedF1 * 100, 2),
    "IoU"        : np.round(weightedIoU * 100, 2),
    "BAC"        : np.round(weightedBAC * 100, 2),
    "MCC"        : np.round(weightedMCC * 100, 2),
    # "Youden"     : np.round(weightedYouden * 100, 2),
    # "Yule"       : np.round(weightedYule, 4),
    # "MI": mutualInfo,

    # "FP"         : FPStr,
    # "FN"         : FNStr,
    # "TP"         : TPStr,
    # "TN"         : TNStr,
    # "Sum"        : np.sum(cm),
  }

  return metrics


resultsPath = r"runs-CK+48"

results = []
for modelKeyword in [
  "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
]:
  references = []
  predictions = []
  for datasetKeyword in ["Test"]:  # "Val", "Test", "Train"
    csvFile = os.path.join(
      resultsPath,
      f"{modelKeyword} Classification-{datasetKeyword}.csv"
    )
    if (not os.path.exists(csvFile)):
      # print(csvFile)
      continue
    df = pd.read_csv(csvFile)
    references.extend(list(df["Actual"].values))
    predictions.extend(list(df["Predicted"].values))

  if (len(references) == 0):
    continue

  metrics = CalculateMetrics(references, predictions)

  # print(f"Model: {modelKeyword}, Dataset: {datasetKeyword}")
  # for key, value in metrics.items():
  #   print(f"{key}: {value}")

  results.append({
    "Model": modelKeyword,
    # "Dataset" : datasetKeyword,
    **metrics,
  })

resultsDf = pd.DataFrame(results)
# Show LaTeX table.
# Round digits to 2 decimal places.
print(
  resultsDf.to_latex(
    index=False,
    float_format="%.2f",
  )
)
