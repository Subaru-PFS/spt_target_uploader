# pfs_target_uploader

Note: development main branch is the `dev-main` branch. If you develop the software, it's recommended to start from the `dev-main` branch.

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

mkdir -p data/
mkdir -p data/temp/
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

Open the target uploader at http://localhost:5006/app and the admin page at http://localhost:5006/admin.
Uploaded files will be stored under `data` with the following structure.

```
$ tree data/
data/
└── <year>
    └── <month>
        └── <year month day>-<hour minute second>-<upload_id>
            ├── ppc_<upload_id>.ecsv
            ├── psl_<upload_id>.ecsv
            ├── target_<upload_id>.ecsv
            └── <original file>
#
# data/
# └── 2023
#     └── 10
#         ├── 20231001-055542-fd56fdccf644972f
#         │   ├── ppc_fd56fdccf644972f.ecsv
#         │   ├── psl_fd56fdccf644972f.ecsv
#         │   └── target_fd56fdccf644972f.ecsv
#         ├── 20231021-010841-ecb95398a3fd10ff
#         │   ├── ppc_ecb95398a3fd10ff.ecsv
#         │   ├── psl_ecb95398a3fd10ff.ecsv
#         │   └── target_ecb95398a3fd10ff.ecsv
#         └── 20231025-042607-5b7849c9ec92703b
#             ├── ppc_5b7849c9ec92703b.ecsv
#             ├── psl_5b7849c9ec92703b.ecsv
#             ├── random_example_n00010.csv
#             └── target_5b7849c9ec92703b.ecsv
```

`ppc`, `psl`, and `target` files corresponds to the lists of pointing centers, pointing summary, and input targets, respectively.
The `data` directory can be controlled by the `OUTPUT_DIR` environment variable in `.env.shared`. An example of `.env.shared` is the following.

```
# OUTPUT_DIR_PREFIX must be identical to the directory value specified as `data` above.
OUTPUT_DIR="data"
```
