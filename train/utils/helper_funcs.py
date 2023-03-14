import os
import shutil

def createCleanDir(base_path):
  try:
      shutil.rmtree(base_path)
  except:
      pass
  os.makedirs(base_path)