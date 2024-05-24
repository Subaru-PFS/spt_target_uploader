# CLI tool

The command-line interface (CLI) tool `pfs-uploader-cli` is provided to run the validation and simulation locally.
You need a valid [Gurobi](https://www.gurobi.com/) license and `GRB_LICENSE_FILE` environment variable needs to be set appropriately.

## `pfs-uploader-cli`

PFS Target Uploader CLI Tool

**Usage**:

```console
$ pfs-uploader-cli [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `simulate`: Run the online PPP to simulate pointings.
* `start-app`: Launch the PFS Target Uploader Web App.
* `validate`: Validate a target list for PFS openuse.

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
* `--max-exec-time INTEGER`: Max execution time (s). Default is 0 (no limit).
* `--max-nppc INTEGER`: Max number of pointings to consider. Default is 0 (no limit).
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
* `--basic-auth TEXT`: Basic authentication config (.json).
* `--cookie-secret TEXT`: Cookie secret.
* `--basic-login-template TEXT`: Basic login template.
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
* `--help`: Show this message and exit.
