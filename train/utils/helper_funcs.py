import os
import shutil

def createCleanDir(base_path):
  try:
      shutil.rmtree(base_path)
  except:
      pass
  os.mkdir(base_path)