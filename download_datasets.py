
URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/1fZYNJRjnjYZzf3e21zZbGjW7V9uG3h03/view?usp=sharing",
    "sf_xs": "https://drive.google.com/file/d/1mdHr_TPm0lmm4lek6GcMh-ZGRHE72IB4/view?usp=sharing",
    "gsv_xs": "https://drive.google.com/file/d/1DpYZGpXOkcv4XvqJs-j03RRBaHGqGjyG/view?usp=sharing"
}

import os
import gdown
import shutil

os.makedirs("data", exist_ok=True)
for dataset_name, url in URLS.items():
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data")
    os.remove(zip_filepath)

