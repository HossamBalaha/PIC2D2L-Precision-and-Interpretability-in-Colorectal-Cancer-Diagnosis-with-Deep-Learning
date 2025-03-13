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

# https://stackoverflow.com/questions/75111196/yolov8-runtimeerror-an-attempt-has-been-made-to-start-a-new-process-before-th
# https://docs.ultralytics.com/tasks/classify/#models
# https://docs.ultralytics.com/usage/cfg/#export-settings

# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import splitfolders, os, cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import heatmap

if __name__ == "__main__":
  outputFolderName = "CK+48 - Output"
  datasetDirName = r"CK+48"
  baseDir = os.getcwd()
  inputShape = (100, 100)
  epochs = 250

  if (not os.path.exists(outputFolderName)):
    splitfolders.ratio(
      os.path.join(baseDir, datasetDirName),
      output=os.path.join(baseDir, outputFolderName),
      seed=1337,
      ratio=(0.7, 0.15, 0.15),
      group_prefix=None,
      move=False,
    )

  for modelKeyword in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:  #
    model = YOLO(f"{modelKeyword}-cls.pt")
    model.train(
      data=os.path.join(baseDir, outputFolderName),
      epochs=epochs,
      imgsz=inputShape[0],
      plots=True,
      save=True,
      name=rf"{modelKeyword}-cls-{inputShape[0]}",
    )

    classes = model.names

    metrics = model.val(
      plots=True,
    )

    print(metrics.top1)
    print(metrics.top5)

    # print("Exporting model...")
    # model.export(
    #   format="saved_model",
    #   # keras=True,
    # )
    # model.save(os.path.join(baseDir, rf"runs\classify\{modelKeyword}-cls-{inputShape[0]}", f"model.keras"))

    # testDir = os.path.join(baseDir, outputFolderName, "test")
    # rndCls = classes[np.random.randint(0, len(classes))]
    # rndImg = np.random.choice(os.listdir(os.path.join(testDir, rndCls)))
    # rndImgPath = os.path.join(testDir, rndCls, rndImg)
    # print("Random image:", rndImgPath)

    # img = cv2.imread(rndImgPath)
    # img = cv2.resize(img, inputShape, interpolation=cv2.INTER_CUBIC)

    # print("Predicting...")
    # pred = model.predict(
    #   [img],
    #   # visualize=True,
    #   imgsz=inputShape[0],
    #   save=True,
    #   classes=model.names,
    # )

    # modelHeatmap = YOLO(os.path.join(baseDir, rf"runs\classify\{modelKeyword}-cls-{inputShape[0]}\weights\best.pt"))

    # print("Generating heatmap...")
    # heatmapObj = heatmap.Heatmap()
    # heatmapObj.set_args(
    #   colormap=cv2.COLORMAP_PARULA,
    #   imw=inputShape[0],
    #   imh=inputShape[1],
    #   view_img=True,
    #   shape="circle",
    #   classes_names=model.names,
    # )
    # results = modelHeatmap.track(
    #   img,
    #   persist=True,
    #   device="gpu",
    #   show=True,
    # )
    # im0 = heatmapObj.generate_heatmap(img, tracks=results)
    # print(im0.shape)
