from transform_parquet_to_hf_ds import *

'''
#### run in shell
img2dataset --url_list data/cc3m_1000.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format parquet\
           --output_folder data/cc3m_files_no --processes_count 16 --thread_count 64 --resize_mode no\
             --enable_wandb False
'''

file_list = retrieve_all_valid_path("data/cc3m_files_no/")
target_parquet_path = "data/cc3m_tiny_no.parquet"
save_to_one_parquet_func(file_list, target_parquet_path)

ds = transform_to_hf_ds(target_parquet_path, "jpg", "caption",
    image_process_func = jpg_val_to_img,
    sha256_column = "sha256"
)
ds.push_to_hub("svjack/cc3m_500_sample")
