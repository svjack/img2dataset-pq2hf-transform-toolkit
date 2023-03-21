<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">img2dataset-pq2hf-transform-toolkit</h3>

  <p align="center">
   		A simple toolkit to transform datasource generate by img2dataset from
      parquet file to Huggingface dataset.
    <br />
  </p>
</p>

### Brief introduction
[img2dataset](https://github.com/rom1504/img2dataset) can easily turn large sets of image urls to an image dataset. Can download, resize and package 100M urls in 20h on one machine. Which is a simple and convenient tool that people
can use it as a image dataset source retrieve toolkit.
Unfortunately, It not provide toolkit that can transform the download dataset into Huggingface official dataset format. i.e. [datasets](https://github.com/huggingface/datasets).
If one take the dataset format in  [datasets](https://github.com/huggingface/datasets)'s form, then it will be a seamless connection between [img2dataset](https://github.com/rom1504/img2dataset) and massive projects in the Huggingface transformers' ecosphere.
This project give a simple toolkit to transform datasource generate by img2dataset from parquet file to Huggingface dataset. And test it on a sample
of Conceptual Captions (CC3M) dataset. And also worked for the fully Conceptual Captions (CC3M) dataset (I have test it by training [svjack/concept-caption-3m-sd-lora-en](https://huggingface.co/svjack/concept-caption-3m-sd-lora-en) and [svjack/concept-caption-3m-sd-lora-zh](https://huggingface.co/svjack/concept-caption-3m-sd-lora-zh) from 400000 images download by this project)

### Installtation
```bash
pip install -r requirements.txt
```

### Use Step
* 1 call img2dataset from console to download images with captions
```bash
#### run in shell
img2dataset --url_list data/cc3m_1000.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format parquet\
           --output_folder data/cc3m_files_no --processes_count 16 --thread_count 64 --resize_mode no\
             --enable_wandb False
```
The above cmd command will download the dataset into data/cc3m_files_no in parquet format

* 2 refer to main.py process the dataset step by step.
This will retrieve all valid files in parquets, and only keep valid download images.
And save them into a parquet file in "data/cc3m_tiny_no.parquet"
```python
file_list = retrieve_all_valid_path("data/cc3m_files_no/")
target_parquet_path = "data/cc3m_tiny_no.parquet"
save_to_one_parquet_func(file_list, target_parquet_path)
```

* 3 transform the "data/cc3m_tiny_no.parquet" into huggingface dataset format <br/>

The implementation decode the download image bytes into PIL's image save and
read them to construct a Huggingface dataset. the image path construct by sha256_column <br/>

Or, if you want to change it to another, you can set sha256_column = None, and
sha256_gen_apply_column to the column you used, Now sha256_gen_apply_column can
be a column with text type <br/>

when both set sha256_column = None and sha256_gen_apply_column = None, the program use image_col as input, call tobytes
to generate sha256.

```python
ds = transform_to_hf_ds(target_parquet_path, "jpg", "caption",
    image_process_func = jpg_val_to_img,
    sha256_column = "sha256"
)
```
This will produce the Huggingface dataset instance as output.

* 4 Push the final produce to the hub. (Or you can use yourself)
```python
ds.push_to_hub("svjack/cc3m_500_sample")
```

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/img2dataset-pq2hf-transform-toolkit](https://github.com/svjack/img2dataset-pq2hf-transform-toolkit)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)

* [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)
* [diffusers](https://github.com/huggingface/diffusers)
* [diffusiondb](https://github.com/poloclub/diffusiondb)
* [Taiyi-Stable-Diffusion-1B-Chinese-v0.1](IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1)
* [prompt-extend](https://github.com/daspartho/prompt-extend)
* [EasyNMT](https://github.com/UKPLab/EasyNMT)
* [Stable-Diffusion-Pokemon](https://github.com/svjack/Stable-Diffusion-Pokemon)
* [svjack](https://huggingface.co/svjack)
-->
* [img2dataset](https://github.com/rom1504/img2dataset)
* [datasets](https://github.com/huggingface/datasets)
* [svjack/concept-caption-3m-sd-lora-en](https://huggingface.co/svjack/concept-caption-3m-sd-lora-en)
* [svjack/concept-caption-3m-sd-lora-zh](https://huggingface.co/svjack/concept-caption-3m-sd-lora-zh)
* [svjack/ControlLoRA-Chinese](https://github.com/svjack/ControlLoRA-Chinese)
* [svjack](https://huggingface.co/svjack)
