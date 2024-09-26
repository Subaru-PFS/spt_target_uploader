# PFS Target Uploader


!!! danger

    The locally launched web app should only be used for a testing purpose and **is NOT intended to submit** the target list to the observatory.
    Please use [the official web app](https://pfs-etc.naoj.hawaii.edu/uploader/) to submit target list.
    If you have any issues for submission via the web app, please [contact us](./about.md).

[The PFS Target Uploader](https://pfs-etc.naoj.hawaii.edu/uploader/) is a web app to validate and submit the target list supplied by users with an observing time estimate by a pointing simulation.

## Install

### Clone the repository

```sh
git clone https://github.com/Subaru-PFS/spt_target_uploader.git
cd spt_target_uploader
```

### Installing dependencies

```sh
pip install -r requirements.txt  # perhaps optional
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

## Run the app

```sh
pfs-uploader-cli start-app uploader \
    --allow-websocket-origin=localhost:5008 \
    --static-dirs doc="./docs/site/" \
    --static-dirs data="./data"
```

Open the target uploader at http://localhost:5008/ .
Uploaded files will be stored under `data` with the following structure.

```
$ tree data/
data/
└── <year>
    └── <month>
        └── <year month day>-<hour minute second>-<upload_id>
            ├── README.txt
            ├── pfs_target-yyyymmdd-hhmmss-<upload_id>.zip
            ├── ppc_<upload_id>.ecsv
            ├── ppp_figure_<upload_id>.html
            ├── psl_<upload_id>.ecsv
            ├── target_<upload_id>.ecsv
            ├── target_summary_<upload_id>.ecsv
            └── <original file>
```

`ppc`, `psl`, and `target` files correspond to the lists of pointing centers, the pointing summary, and input targets, respectively.
Plots are available in the `ppp_figure` file and all files are included in the `zip` file.

The path to the `data` directory can be controlled by the `OUTPUT_DIR` environment variable in `.env.shared`. An example of `.env.shared` is the following.

```
# OUTPUT_DIR_PREFIX must be identical to the directory value specified as `data` above.
OUTPUT_DIR="data"
```

## Configuration

The following parameters can be set in the `.env.shared` file to configure the app.

```
# Output directory for the submitted files
OUTPUT_DIR="data"

# maximum execution time (s) to terminate the calculation (default: 900s = 15min, 0 = no limit)
# MAX_EXETIME=0

# maximum number of pointings to be considered (default: 200, 0 = no limit)
# MAX_NPPC=100

# email setting (email will be sent at each submission)
# EMAIL_FROM=
# EMAIL_TO=
# SMTP_SERVER=

# Supress output of netflow
# 0: verbose
# 1: quiet
PPP_QUIET=1

# Target clustering algorithm
# HDBSCAN or DBSCAN
CLUSTERING_ALGORITHM=HDBSCAN

# Text to be announce at the beginning (Markdown)
ANN_FILE="user_announcement.md"

# SQLite database file to be used for the duplication check of upload_id
# The file will be created under $OUTPUT_DIR
UPLOADID_DB="upload_id.sqlite"
```

## Preparing database

When `UPLOADID_DB` is set, the uploader looks up `$OUTPUT_DIR/$UPLOADID_DB` file for the duplication check of `upload_id`.
The following command can be used to generate the database file.

```sh
pfs-uploader-cli uid2sqlite -d $OUTPUT_DIR --db $UPLOADID_DB
```

If you have a list of `upload_id`s to be inserted into the database (`upload_id.csv`), you can run the command as follows.

```sh
pfs-uploader-cli uid2sqlite -d $OUTPUT_DIR --db $UPLOADID_DB upload_id.csv
```

The example content of `upload_id.csv` is as follows.

```csv
upload_id
c748124208176c40
4cd4bc355c092ad7
1b8d0c4f808972bb
2e07c75691e5ba26
c695c6b755930209
```

If you want to scan a directory (e.g., `$OUTPUT_DIR`) containing submitted uploads, you can run the command as follows.

```sh
pfs-uploader-cli uid2sqlite -d $OUTPUT_DIR --db $UPLOADID_DB --scan-dir $OUTPUT_DIR
```

You can remove duplicates by the following command.

```sh
pfs-uploader-cli clean-uid $OUTPUT_DIR/$UPLOADID_DB
```

See [the manual](./cli.md) for more options.
