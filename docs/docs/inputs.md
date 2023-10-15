# Input Target List

## Contents of the list

There are required and optional fields. String values must consist of `[A-Za-z0-9_-+.]`.


### Required fields

Mandatory fields are listed below.

| Name       | Datatype   | Unit   | Description                                                                                        |
|------------|------------|--------|----------------------------------------------------------------------------------------------------|
| ob_code    | str        |        | A unique string to identify the entry                                                              |
| obj_id     | 64 bit int |        | Object ID                                                                                          |
| ra         | float      | degree | Right Ascension at the reference epoch specified by `equinox`                                      |
| dec        | float      | degree | Declination at the the reference epoch specified by `equinox`                                      |
| equinox    | str        |        | Equinox string (must be J2000.0)                                                                   |
| exptime    | float      | second | Exposure time requested for the object                                                             |
| priority   | int        |        | Priority (0-9) for the object within the list. Smaller value for higher priority                   |
| resolution | str        |        | Grating used in the red optical arms. `L` for the low resolution and `M` for the medium resolution |

#### Equinox

It is users' responsibility to make sure the coordinates are in J2000.0.

#### Flux information

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

##### Example

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
| tract    | int      |          |         | Tract ID                                   |
| patch    | int      |          |         | Patch name                                 |

### Filters

Currently, the following filters are registered in our database. Filters are categorized as follows.

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

## File format

Input target list must be a Comma-separated values (CSV) file.


## Example

An example CSV file can be found in [this link](examples/example_targetlist.csv)
