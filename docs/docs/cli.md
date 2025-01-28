# CLI tool

The command-line interface (CLI) tool `pfs-uploader-cli` is provided to run the validation and simulation locally.
You need a valid [Gurobi](https://www.gurobi.com/) license and `GRB_LICENSE_FILE` environment variable needs to be set appropriately.

!!! danger

    The CLI tools are experimental and **NOT intended to submit** the target list to the observatory.
    Please use [the official web app](https://pfs-etc.naoj.hawaii.edu/uploader/) to submit target list.
    If you have any issues for submission via the web app, please [contact us](./about.md).

## `pfs-uploader-cli`

PFS Target Uploader CLI Tool

**Usage**:

```console
$ pfs-uploader-cli [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `clean-uid`: Remove duplicates from a SQLite database...
* `simulate`: Run the online PPP to simulate pointings.
* `start-app`: Launch the PFS Target Uploader Web App.
* `uid2sqlite`: Generate a SQLite database of upload_id
* `validate`: Validate a target list for PFS openuse.

---

### `clean-uid`

Remove duplicates from a SQLite database of upload_id

**Usage**:

```console
$ pfs-uploader-cli clean-uid [OPTIONS] DBFILE
```

**Arguments**:

* `DBFILE`: Full path to the SQLite database file.  [required]

**Options**:

* `--backup / --no-backup`: Create a backup of the database before cleaning. Default is True.  [default: backup]
* `--dry-run / --no-dry-run`: Do not remove duplicates; just check the duplicates. Default is False.  [default: no-dry-run]
* `--log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: Set the log level.  [default: INFO]
* `--help`: Show this message and exit.

---

### `simulate`

Run the online PPP to simulate pointings.

The result is written under the directory set by the `--dir` option with a 16 character random string.

**Usage**:

```console
$ pfs-uploader-cli simulate [OPTIONS] INPUT_LIST
```

**Arguments**:

* `INPUT_LIST`: Input CSV file  [required]

**Options**:

* `-d, --dir TEXT`: Output directory to save the results.  [default: .]
* `--date-begin TEXT`: Begin date (e.g., 2023-02-01). The default is the first date of the next Subaru semester.
* `--date-end TEXT`: End date (e.g., 2023-07-31). The default is the last date of the next Subaru semester.
* `--single-exptime INTEGER`: Single exposure time (s).  [default: 900]
* `--max-exec-time INTEGER`: Max execution time (s). 0 means no limit.  [default: 0]
* `--obs-type [queue|classical|filler]`: Observation type.  [default: queue]
* `--log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: Set the log level.  [default: INFO]
* `--help`: Show this message and exit.

---

### `start-app`

Launch the PFS Target Uploader Web App.

**Usage**:

```console
$ pfs-uploader-cli start-app [OPTIONS] APP:{uploader|admin}
```

**Arguments**:

* `APP:{uploader|admin}`: App to launch.  [required]

**Options**:

* `--port INTEGER`: Port number to run the server.  [default: 5008]
* `--prefix TEXT`: URL prefix to serve the app.
* `--allow-websocket-origin TEXT`: Allow websocket origin.
* `--static-dirs TEXT`: Static directories.
* `--use-xheaders / --no-use-xheaders`: Set --use-xheaders option.  [default: no-use-xheaders]
* `--num-procs INTEGER`: Number of processes to run.  [default: 1]
* `--autoreload / --no-autoreload`: Set --autoreload option.  [default: no-autoreload]
* `--max-upload-size INTEGER`: Maximum file size in MB.  [default: 500]
* `--session-token-expiration INTEGER`: Session token expiration time in seconds.  [default: 1800]
* `--basic-auth TEXT`: Basic authentication config (.json).
* `--basic-login-template TEXT`: Basic login template.
* `--log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: Set the log level.  [default: INFO]
* `--help`: Show this message and exit.

---

### `uid2sqlite`

Generate a SQLite database of upload_id

**Usage**:

```console
$ pfs-uploader-cli uid2sqlite [OPTIONS] [INPUT_LIST]
```

**Arguments**:

* `[INPUT_LIST]`: Input CSV file.

**Options**:

* `-d, --dir TEXT`: Output directory to save the results.  [default: .]
* `--db TEXT`: Filename of the SQLite database to save the upload_id.  [default: upload_id.sqlite]
* `--scan-dir TEXT`: Directory to scan for the upload_id. Default is None (use input file)
* `--clean`: Remove duplicates from the database. Default is False.
* `--log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: Set the log level.  [default: INFO]
* `--help`: Show this message and exit.

---

### `validate`

Validate a target list for PFS openuse.

**Usage**:

```console
$ pfs-uploader-cli validate [OPTIONS] INPUT_LIST
```

**Arguments**:

* `INPUT_LIST`: Input CSV file.  [required]

**Options**:

* `-d, --dir TEXT`: Output directory to save the results.  [default: .]
* `--date-begin TEXT`: Begin date (e.g., 2023-02-01). The default is the first date of the next Subaru semester.
* `--date-end TEXT`: End date (e.g., 2023-07-31). The default is the last date of the next Subaru semester.
* `--save / --no-save`: Save the validated target list in the directory specified by "--dir".  [default: no-save]
* `--obs-type [queue|classical|filler]`: Observation type.  [default: queue]
* `--log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: Set the log level.  [default: INFO]
* `--help`: Show this message and exit.
