# pfs_target_uploader


## Install

### Clone the repository

```sh
git clone https://github.com/monodera/pfs_target_uploader.git
cd pfs_target_uploader
```

### Installing dependencies

```sh
pip install -r requirements
pip install -e .

mkdir -p data/target_lists
```


### Build documentation

```sh
cd docs
mkdocs build
cd ..
```

### Run the app

```sh
panel serve ./uploader.py ./uploader-admin.py \
    --autoreload \
    --static-dirs uploader-docs=./docs/site uploader-data=./data/target_lists
```

Open the target uploader at http://localhost:5006/uploader and the admin page at http://localhost:5006/uploader-admin. Uploaded files will be stored under `data/target_list`. This parameter can be controlled by editing `OUTPUT_DIR` environment variable in `.env.shared`.
