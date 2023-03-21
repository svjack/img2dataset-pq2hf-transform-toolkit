import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import cv2
from PIL import Image
import pathlib
import os
import pandas as pd
from tqdm import tqdm

def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
                         # numpy Image Channel system: [B,G,R,A]
    red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red;
    return i;

def toImgPIL(imgOpenCV): return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB));

def jpg_val_to_img(jpg_bytes):
    img_buf = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
    return toImgPIL(img)
