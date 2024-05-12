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

### `validate`

Validate a target list for PFS openuse.

**Usage**:

```console
$ pfs-uploader-cli validate [OPTIONS] INPUT_LIST
```

**Arguments**:

* `INPUT_LIST`: Input CSV file.  [required]

**Options**:

* `--date-begin TEXT`: Begin date (e.g., 2023-02-01). The default is the first date of the next Subaru semester.
* `--date-end TEXT`: End date (e.g., 2023-07-31). The default is the last date of the next Subaru semester.
* `--help`: Show this message and exit.


