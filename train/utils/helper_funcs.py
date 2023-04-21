import os
import shutil
import skimage
import numpy as np

def createCleanDir(base_path):
  try:
      shutil.rmtree(base_path)
  except:
      pass
  os.makedirs(base_path)


def norm(band, maximum=1):
    band_min, band_max = np.nanmin(band), np.nanmax(band)
    
    return ((band - band_min)/(band_max - band_min))*maximum

def make_color_image_eqh(
    b1, b2, b3, 
):
   
    eq_b1 = skimage.exposure.equalize_hist(b1, mask=(~np.isnan(b1)))
    eq_b2 = skimage.exposure.equalize_hist(b2, mask=(~np.isnan(b2)))
    eq_b3 = skimage.exposure.equalize_hist(b3, mask=(~np.isnan(b3)))

    # normalize data to 0<->1
    b1_norm = norm(eq_b1)
    b2_norm = norm(eq_b2)
    b3_norm = norm(eq_b3)

    # create three color image
#     rgb = np.stack([b1_norm, b2_norm, b3_norm], axis=2)
    
    
    rgb = np.dstack((b1_norm, b2_norm, b3_norm))

    return rgb

def equalize_exposure(b1, b2, b3):
    eq_b1 = skimage.exposure.equalize_hist(b1, mask=(~np.isnan(b1)))
    eq_b2 = skimage.exposure.equalize_hist(b2, mask=(~np.isnan(b2)))
    eq_b3 = skimage.exposure.equalize_hist(b3, mask=(~np.isnan(b3)))
    
    rgb = np.dstack((eq_b1, eq_b2, eq_b3))

    return rgb