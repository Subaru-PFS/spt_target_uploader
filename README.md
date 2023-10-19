# pfs_target_uploader


## Install

### Clone the repository

```sh
git clone https://github.com/monodera/pfs_target_uploader.git
cd pfs_target_uploader
```

### Installing dependencies

```sh
pip install -r requirements.txt
pip install -e .

mkdir -p data/{target_lists,psl_lists,ppc_lists}
```


### Build documentation

```sh
cd docs
mkdocs build
cd ..
```

### Run the app

```sh
panel serve ./app.py ./admin.py --autoreload --static-dirs doc=./docs/site data=./data
```

Open the target uploader at http://localhost:5006/app and the admin page at http://localhost:5006/app-admin. Uploaded files will be stored under `data/target_lists`, and the pointing lists will be stored under `data/ppc_lists`. This parameter can be controlled by editing `OUTPUT_DIR_data` and `OUTPUT_DIR_ppc` environment variable in `.env.shared`. An example of `.env.shared` is the following.

```
OUTPUT_DIR_PREFIX="data"
OUTPUT_DIR_data="target_lists"
OUTPUT_DIR_ppp="psl_lists"
OUTPUT_DIR_ppc="ppc_lists"
```
