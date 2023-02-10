import os
import shutil

def createCleanDir(base_path):
  try:
      shutil.rmtree(base_path)
  except:
      pass
  os.mkdir(base_path)


def createDir(dirPath):
  try:
    os.makedirs(dirPath)
  except:
    pass

def writeGeo(dirName, imgList):
  with open(dirName + "/geocoordinates.txt", "w") as f:
    for data in imgList:
      f.write(",".join(data) + "\n")