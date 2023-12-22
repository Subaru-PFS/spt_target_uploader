# Input Target List

## File format

Input target list must be in the [Comma-separated values (CSV)](https://en.wikipedia.org/wiki/Comma-separated_values) format (`.csv`) or
the [Enhanced Character-Separated Values (ECSV)](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) format (`.ecsv`).

## Content

Here the word "target" is a combination of values of the fields described below. There are required and optional fields. String values must consist of `[A-Za-z0-9_-+.]`.

### Quick example

An example CSV file containing 100 random targets can be found in [this link](examples/example_targetlist_random10.csv).
A quick example of the content is shown below.

| ob_code     | obj_id |                 ra |                dec | exptime | priority | resolution |     g_hsc | g_hsc_error |
|-------------|-------:|-------------------:|-------------------:|--------:|---------:|------------|----------:|------------:|
| ob_00000000 |      0 |  278.6241774801468 |  56.29564017478978 |  3600.0 |        9 | L          |   3093.21 |      388.14 |
| ob_00000001 |      1 | 157.99623831073885 |  31.71664232033844 |  3600.0 |        7 | M          | 108911.74 |    30452.86 |
| ob_00000002 |      2 | 309.09525116809766 | -6.329978341797448 |  3600.0 |        3 | M          |  11842.32 |     2905.20 |

### Required fields

Mandatory fields are listed below.

| Name       | Datatype   | Unit   | Description                                                                                              |
|------------|------------|--------|----------------------------------------------------------------------------------------------------------|
| ob_code    | str        |        | A string identifier for the target. Each `ob_code` must be unique within the list.                       |
| obj_id     | 64-bit int |        | Object ID (-9223372036854775808 to +9223372036854775807).                                                |
| ra         | float      | degree | Right Ascension (J2000.0 or ICRS at the epoch of 2000.0)                                                 |
| dec        | float      | degree | Declination (J2000.0 or ICRS at the epoch of 2000.0)                                                     |
| exptime    | float      | second | Exposure time requested for the object under the nominal observing condition.                            |
| priority   | int        |        | Priority (integer value in [0-9]) for the object within the list. Smaller the value, higher the priority |
| resolution | str        |        | Grating used in the red optical arms. `L` for the low resolution and `M` for the medium resolution       |

#### About `ob_code`

In a given target list, each `ob_code` must be unique. Some examples for good cases are shown below.

A standard case.

| ob_code | obj_id |
|---------|-------:|
| ob_1    |      1 |
| ob_2    |      2 |

You can request to observe an object with both `L` and `M` resolutions, but you need to use different `ob_code` for each case.

| ob_code | obj_id | resolution |
|---------|-------:|------------|
| ob_1_L  |      1 | L          |
| ob_1_M  |      1 | M          |

You can request to observe an object multiple times instead of summing up the exposure time, but you need to use different `ob_code` for each case.

| ob_code     | obj_id | exptime |
|-------------|-------:|--------:|
| ob_1_900s_1 |      1 |     900 |
| ob_1_900s_2 |      1 |     900 |
| ob_1_900s_3 |      1 |     900 |
| ob_1_900s_4 |      1 |     900 |

This is essentially equivalent to the following if no time constraint is specified.

| ob_code    | obj_id | exptime |
|------------|-------:|--------:|
| ob_1_3600s |      1 |    3600 |

#### About astrometry

Since the [Gaia DR3](https://www.cosmos.esa.int/web/gaia/data-release-3) catalog is used to find guide stars,
coordinates must be in the International Celestial Reference System (ICRS).
Users are required to make coordinates of targets consistent with the Gaia astrometry at the epoch of 2000.0.
Note that coordinates in ICRS at the epoch of 2000.0 is known to be consistent with those with equinox J2000.0 represented by the FK5 within the errors of the FK5.

#### About Flux information

Flux columns must conform the following requirements.

- **At least one** flux information for each `ob_code` is required.
- **The names of flux columns must be chosen** from the pre-defined [filters](#filters).
- Filters are categorized as shown in the [filter list](#filters).
  An `ob_code` cannot have more than one fluxes in the same filter category.
- If more than one flux columns with finite values are found for an `ob_code`,
  the value of the first column (left-most one in the input CSV file) will be used.
- Flux values are in the unit of <font size=5>**nJy**</font>.
- Flux values are assumed to be total flux.
- Errors can be provided by using column names by adding `_error` following the filter names.

##### Example for flux information

✅ Good

| ob_code | g_hsc | g_hsc_error | i_hsc | i_hsc_error | g_ps1 | g_ps1_error |
|---------|-------|-------------|-------|-------------|-------|-------------|
| 1       | 10000 | 100         |       |             |       |             |
| 2       | 20000 | 200         | 20000 |             |       |             |
| 3       |       |             |       |             | 30000 | 300         |

⚠️ OK

- For the `ob_code 3`, `g_hsc` will be used and `g_ps1` will be ignored.

| ob_code | g_hsc | g_hsc_error | i_hsc | i_hsc_error | g_ps1 | g_ps1_error |
|---------|-------|-------------|-------|-------------|-------|-------------|
| 1       | 10000 | 100         |       |             |       |             |
| 2       | 20000 | 200         | 20000 |             |       |             |
| 3       | 35000 | 350         |       |             | 30000 | 300         |

🚫 Bad

- The `ob_code 1` does not have flux information at all.

| ob_code | g_hsc | g_hsc_error | i_hsc | i_hsc_error | g_ps1 | g_ps1_error |
|---------|-------|-------------|-------|-------------|-------|-------------|
| 1       |       |             |       |             |       |             |
| 2       | 20000 | 200         | 20000 |             |       |             |
| 3       |       |             |       |             | 30000 | 300         |

### Optional fields

Optional fields are listed below.

| Name     | Datatype | Unit     | Default | Description                                |
|----------|----------|----------|---------|--------------------------------------------|
| pmra     | float    | mas/year | 0       | Proper motion in right ascension direction |
| pmdec    | float    | mas/year | 0       | Proper motion in declination direction     |
| parallax | float    | mas      | 1e-7    | Parallax                                   |
| tract    | int      |          | None    | Tract ID                                   |
| patch    | int      |          | None    | Patch ID                                   |

Note that, if provided, `tract` and `patch` are expected to follow the output of [the HSC pipeline](https://hsc.mtk.nao.ac.jp/pipedoc_e/).
See the [relevant section](https://hsc.mtk.nao.ac.jp/pipedoc/pipedoc_8_e/tutorial_e/basic_info.html#tract-patch) of the pipeline.
If they are not provided, `None` will be used as the default value.

### Filters

Currently, the following filters are registered in our database. Filters are categorized as follows.
Specifying filters not in the following will be ignored.

#### `g` category filters

| Name    | Description          |
|---------|----------------------|
| g_hsc   | HSC g filter         |
| g_ps1   | Pan-STARRS1 g filter |
| g_sdss  | SDSS g filter        |
| bp_gaia | Gaia BP filter       |

#### `r` category filters

| Name      | Description                 |
|-----------|-----------------------------|
| r_old_hsc | HSC r filter (old r filter) |
| r2_hsc    | HSC r2 filter               |
| r_ps1     | Pan-STARRS1 r filter        |
| r_sdss    | SDSS r filter               |
| g_gaia    | Gaia G filter               |

#### `i` category filters

| Name      | Description                 |
|-----------|-----------------------------|
| i_old_hsc | HSC i filter (old i filter) |
| i2_hsc    | HSC i2 filter               |
| i_ps1     | Pan-STARRS1 i filter        |
| i_sdss    | SDSS i filter               |
| rp_gaia   | Gaia RP filter              |

#### `z` category filters

| Name   | Description          |
|--------|----------------------|
| z_hsc  | HSC z filter         |
| z_ps1  | Pan-STARRS1 z filter |
| z_sdss | SDSS z filter        |

#### `y` category filters

| Name  | Description          |
|-------|----------------------|
| y_hsc | HSC Y filter         |
| y_ps1 | Pan-STARRS1 y filter |

#### `j` category filters

TBD
