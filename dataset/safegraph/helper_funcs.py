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