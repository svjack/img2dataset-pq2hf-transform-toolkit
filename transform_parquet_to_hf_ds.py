#### parquet merge
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import cv2
from PIL import Image
import pathlib
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk
import shutil
import hashlib

from img_toolkit import *

#### data source file "data/cc3m_1000.tsv"
'''
#### run in shell
img2dataset --url_list data/cc3m_1000.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format parquet\
           --output_folder data/cc3m_files_no --processes_count 16 --thread_count 64 --resize_mode no\
             --enable_wandb False
'''
#### This produce data/cc3m_files_no: contains output in parquet format

def retrieve_all_valid_path(dir_):
    assert os.path.exists(dir_)
    valid_p_files = pd.Series(pathlib.Path(dir_).rglob("*.parquet")).map(str).map(
        lambda p: p if os.path.exists(p.replace(".parquet", "_stats.json")) else np.nan
    ).dropna().drop_duplicates().values.tolist()
    return valid_p_files

def read_one_p_file_to_df(p_path, return_table = False,
    ):
    assert os.path.exists(p_path)
    bdf = pq.read_table(p_path)
    bdf = bdf.to_pandas()
    bdf0 = bdf[
        bdf["status"].map(lambda x: x == "success")
    ]
    if return_table:
        return pa.Table.from_pandas(bdf0)
    return bdf0

def save_to_one_parquet_func(p_files_list, file_path):
    print("length : {}".format(len(p_files_list)))
    assert file_path.endswith(".parquet")
    import shutil
    if os.path.exists(file_path):
        os.remove(file_path)
    table_list = list(map(lambda x:
        read_one_p_file_to_df(x, return_table = True)
    , p_files_list))
    ptable = pa.concat_tables(table_list)
    print(ptable.shape)
    pq.write_table(ptable, file_path)

def retrieve_caption_cols(p_path, caption_cols = ["caption"]):
    parquet_file = pq.ParquetFile(p_path)
    df = parquet_file.read(columns=caption_cols).to_pandas()
    return df

def transform_to_hf_ds(source_parquet_file, image_col, caption_col = None,
    image_process_func = lambda _:_, default_image_col = "image",
    tmp_img_save_dir = "tmp_img", sha256_column = None,
    sha256_gen_func = lambda x:  hashlib.sha256(x.encode() if hasattr(x, "encode") else x).hexdigest(),
    sha256_gen_apply_column = None
):
    os.makedirs(tmp_img_save_dir, exist_ok = True)
    assert callable(image_process_func)
    columns = [image_col]
    if caption_col is not None:
        columns.append(caption_col)
    ds = Dataset.from_parquet(source_parquet_file)

    req_list = []
    for i in tqdm(range(len(ds))):
        try:
            ele = ds[i]
            jpg_buffer = ele[image_col]
            if sha256_column is not None:
                sha256_string = ele[sha256_column]
            else:
                if sha256_gen_apply_column is not None:
                    sha256_string = sha256_gen_func(ele[sha256_gen_apply_column])
                else:
                    assert hasattr(ele[image_col], "tobytes")
                    sha256_string = sha256_gen_func(ele[image_col].tobytes())

            img_path = os.path.join(tmp_img_save_dir,
            "{}.jpg".format(sha256_string))

            img = image_process_func(jpg_buffer)
            img.save(img_path)
            if caption_col is not None:
                req_list.append({
                    "img_path": img_path,
                    "caption": ele[caption_col]
                })
            else:
                req_list.append({
                    "img_path": img_path,
                    #"caption": ele[i][caption_col]
                })
        except:
            print("err")

    ds = Dataset.from_pandas(pd.DataFrame(req_list))
    ds = ds.map(
        lambda x: {default_image_col:
            Image.open(x["img_path"])
        },
    )
    req_columns = [default_image_col]
    if caption_col is not None:
        req_columns.append(caption_col)
    ds = ds.remove_columns(set(ds.column_names).difference(set(req_columns)))
    #shutil.rmtree(tmp_img_save_dir)
    return ds

if __name__ == "__main__":
    file_list = retrieve_all_valid_path("data/cc3m_files_no/")
    target_parquet_path = "data/cc3m_tiny_no.parquet"
    save_to_one_parquet_func(file_list, target_parquet_path)

    #### return pandas dataframe may too big when massive data
    cc3m_tiny_no_df = read_one_p_file_to_df(target_parquet_path,
     return_table = False)

    #### if only require caption cols (for example: want to
    #### translate caption into another language), read by ParquetFile interface, Column storage efficiency
    cc3m_tiny_caption_df = retrieve_caption_cols(target_parquet_path, caption_cols = ["caption"])

    ds = transform_to_hf_ds(target_parquet_path, "jpg", "caption",
        image_process_func = jpg_val_to_img,
        sha256_column = "sha256"
    )
    ds.push_to_hub("svjack/cc3m_500_sample")
