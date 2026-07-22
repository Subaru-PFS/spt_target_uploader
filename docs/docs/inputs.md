# Inputs

## Input Target List

### File format

An input target list must be in the [Comma-separated values (CSV)](https://en.wikipedia.org/wiki/Comma-separated_values) format (`.csv`) or
the [Enhanced Character-Separated Values (ECSV)](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) format (`.ecsv`).

### Content

Here the word "target" is a combination of values of the fields described below. There are required and optional fields. String values must consist of `[A-Za-z0-9_-+.]`.

### Quick example

[An example CSV file](examples/example_perseus_cluster_r60arcmin.csv) containing 539 galaxies within 60 arcmin from the Perseus cluster ([Meusinger et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...640A..30M/abstract)) is available.

A quick example of the content is shown below.

| ob_code     | obj_id |                 ra |                dec | exptime | priority | resolution | reference_arm |     g_hsc | g_hsc_error |
| ----------- | -----: | -----------------: | -----------------: | ------: | -------: | ---------- | ------------- | --------: | ----------: |
| ob_00000000 |      0 |  278.6241774801468 |  56.29564017478978 |  3600.0 |        9 | L          | b             |   3093.21 |      388.14 |
| ob_00000001 |      1 | 157.99623831073885 |  31.71664232033844 |  3600.0 |        7 | M          | r             | 108911.74 |    30452.86 |
| ob_00000002 |      2 | 309.09525116809766 | -6.329978341797448 |  3600.0 |        3 | M          | n             |  11842.32 |     2905.20 |

### Required fields

Mandatory fields are listed below.

| Name          | Datatype   | Unit   | Description                                                                                                                         |
| ------------- | ---------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| ob_code       | str        |        | A string identifier for the target. Each `ob_code` must be unique within the list.                                                  |
| obj_id        | 64-bit int |        | Object ID (-9223372036854775808 to +9223372036854775807).                                                                           |
| ra            | float      | degree | Right Ascension (ICRS at the reference epoch of 2000.0)                                                                             |
| dec           | float      | degree | Declination (ICRS at the reference epoch of 2000.0)                                                                                 |
| exptime       | float      | second | Exposure time requested for the object under the nominal observing condition.                                                       |
| priority      | int        |        | Priority (integer value in [0-9]) for the object within the list. Smaller the value, higher the priority                            |
| resolution    | str        |        | Grating used in the red optical arms. `L` for the low resolution and `M` for the medium resolution                                  |
| flux          | float      | nJy    | Flux of at least one filter in the pre-defined [list](#filters)                                                                     |
| reference_arm | str        |        | Reference arm name used to evaluate the effective exposure time (`b`: blue, `r`: red, `n`: near-IR, and `m`: medium-resolution red) |

#### About `reference_arm`

When `resolution` is `L`, `reference_arm` must be one of `b`, `r`, and `n`.
On the other hand, `reference_arm` must be one of `b`, `m`, and `n` when `resolution` is `M`.

#### About uniqueness condition by `(obj_id, resolution)` and `ob_code`

In a given target list, each pair of `(obj_id, resolution)` must be unique.
Each `ob_code` must also be unique.

Some examples of good and bad cases are shown below.

✅ Good

A standard case.

| ob_code | obj_id |
| ------- | -----: |
| ob_1    |      1 |
| ob_2    |      2 |

🚫 Bad

The following case violates the unique constraints by setting the duplicated `(obj_id, resolution)` values.

| ob_code  | obj_id | resolution |
| -------- | -----: | ---------- |
| ob_1_L_1 |      1 | L          |
| ob_1_L_2 |      1 | L          |

✅ Good

You can request to observe an object with both `L` and `M` resolutions, but you need to use a different `ob_code` for each case.

| ob_code | obj_id | resolution |
| ------- | -----: | ---------- |
| ob_1_L  |      1 | L          |
| ob_1_M  |      1 | M          |

🚫 Bad

You cannot have multiple rows of an object a target list, even if you assign different `ob_code` for each row.

| ob_code     | obj_id | exptime | resolution |
| ----------- | -----: | ------: | ---------- |
| ob_1_900s_1 |      1 |     900 | L          |
| ob_1_900s_2 |      1 |     900 | L          |
| ob_1_900s_3 |      1 |     900 | L          |
| ob_1_900s_4 |      1 |     900 | L          |

✅ Good

For the case above, please make a row by summing up the exposure time as follows.

| ob_code    | obj_id | exptime | resolution |
| ---------- | -----: | ------: | ---------- |
| ob_1_3600s |      1 |    3600 | L          |

#### About astrometry

Since the [Gaia DR3](https://www.cosmos.esa.int/web/gaia/data-release-3) catalog is used to find guide stars,
coordinates must be in the International Celestial Reference System (ICRS).
Users are required to make coordinates of targets consistent with the Gaia astrometry at the reference epoch of 2000.0.

#### About Flux information

Flux columns must conform to the following requirements.

- **At least one** flux information for each `ob_code` is required.
- **The names of flux columns must be chosen** from the pre-defined [filters](#filters).
- Filters are categorized as shown in the [filter list](#filters).
  An `ob_code` cannot have more than one flux in the same filter category.
- If more than one flux columns with finite values are found for an `ob_code`,
  the value of the first column (the left-most one in the input CSV file) will be used.
- For `ob_code`s with no measurement in a given filter column, use `NaN` (or `Inf`) as the placeholder. Do **not** use `0.0`, as it will be treated as a valid flux value.
- Missing flux values are detected using Numpy's `isfinite()`. Please refer to the table
  below to specify missing values correctly.

  | Value | Result                            |
  | ----- | --------------------------------- |
  | `0.0` | ❌ Treated as a flux measurement  |
  | `NaN` | ✅ Treated as missing and ignored |
  | `Inf` | ✅ Treated as missing and ignored |

- Flux values are in the unit of <font size=5>**nJy**</font> (nano-Jansky).
- Flux values are assumed to be total flux.
- Flux errors can be provided by using column names by adding `_error` following the filter names.
- All **queue-mode** targets are required to be fainter than **16 AB mag** in all provided filters.
- All **filler-mode** targets are required to be fainter than **17 AB mag** in all provided filters.

!!! note

    The bright magnitude limits above are applied to avoid detector saturation and
    to reduce near infrared persistency effects.
    Users can still submit targets brighter than these limits,
    but additional validation and checks will be performed
    at the time of fiber assignment processing.
    See the [Observation](https://www.naoj.org/Instruments/PFS/observations.html)
    page of the PFS web site for details.

##### Example of flux information

✅ Good

| ob_code | g_hsc | g_hsc_error | i2_hsc | i2_hsc_error | g_ps1 | g_ps1_error |
| ------- | ----- | ----------- | ------ | ------------ | ----- | ----------- |
| 1       | 10000 | 100         | NaN    | NaN          | NaN   | NaN         |
| 2       | 20000 | 200         | 20000  | NaN          | NaN   | NaN         |
| 3       | NaN   | NaN         | NaN    | NaN          | 30000 | 300         |

⚠️ OK

- `ob_code 3` has both `g_hsc` and `g_ps1` fluxes, but only `g_hsc` will be used as the flux value, as it is the first column appeared and has a finite value.

| ob_code | g_hsc     | g_hsc_error | i2_hsc | i2_hsc_error | g_ps1     | g_ps1_error |
| ------- | --------- | ----------- | ------ | ------------ | --------- | ----------- |
| 1       | 10000     | 100         | NaN    | NaN          | NaN       | NaN         |
| 2       | 20000     | 200         | 20000  | NaN          | NaN       | NaN         |
| **3**   | **35000** | **350**     | NaN    | NaN          | **30000** | **300**     |

🚫 Bad

- The `ob_code 1` does not have flux information at all.
- The `i_hsc` must be either `i_old_hsc` or `i2_hsc`.

| ob_code | g_hsc   | g_hsc_error | **i_hsc** | **i_hsc_error** | g_ps1   | g_ps1_error |
| ------- | ------- | ----------- | --------- | --------------- | ------- | ----------- |
| **1**   | **NaN** | **NaN**     | **NaN**   | **NaN**         | **NaN** | **NaN**     |
| 2       | 20000   | 200         | 20000     | NaN             | NaN     | NaN         |
| 3       | NaN     | NaN         | NaN       | NaN             | 30000   | 300         |

- `ob_code 3` has two flux measurements in the `g` category. The value `g_hsc=0.0` will be used as the flux value, because it is the first column appeared and has a finite value. This is likely not the intended behavior.

| ob_code | g_hsc   | g_hsc_error | i2_hsc | i2_hsc_error | g_ps1 | g_ps1_error |
| ------- | ------- | ----------- | ------ | ------------ | ----- | ----------- |
| 1       | 10000   | 100         | NaN    | NaN          | NaN   | NaN         |
| 2       | 20000   | 200         | 20000  | NaN          | NaN   | NaN         |
| **3**   | **0.0** | **0.0**     | NaN    | NaN          | 30000 | 300         |

### Optional fields

Optional fields are listed below.

| Name     | Datatype | Unit     | Default | Description                                |
| -------- | -------- | -------- | ------- | ------------------------------------------ |
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
| ------- | -------------------- |
| g_hsc   | HSC g filter         |
| g_ps1   | Pan-STARRS1 g filter |
| g_sdss  | SDSS g filter        |
| bp_gaia | Gaia BP filter       |

#### `r` category filters

| Name      | Description                 |
| --------- | --------------------------- |
| r_old_hsc | HSC r filter (old r filter) |
| r2_hsc    | HSC r2 filter               |
| r_ps1     | Pan-STARRS1 r filter        |
| r_sdss    | SDSS r filter               |
| g_gaia    | Gaia G filter               |

#### `i` category filters

| Name      | Description                 |
| --------- | --------------------------- |
| i_old_hsc | HSC i filter (old i filter) |
| i2_hsc    | HSC i2 filter               |
| i_ps1     | Pan-STARRS1 i filter        |
| i_sdss    | SDSS i filter               |
| rp_gaia   | Gaia RP filter              |

#### `z` category filters

| Name   | Description          |
| ------ | -------------------- |
| z_hsc  | HSC z filter         |
| z_ps1  | Pan-STARRS1 z filter |
| z_sdss | SDSS z filter        |

#### `y` category filters

| Name  | Description          |
| ----- | -------------------- |
| y_hsc | HSC Y filter         |
| y_ps1 | Pan-STARRS1 y filter |

#### `j` category filters

TBD

## (Optional) Input Pointing List

For classical-mode observations, a user-defined pointing list can be provided.

### File format for the pointing list

An input target list must be in the [Comma-separated values (CSV)](https://en.wikipedia.org/wiki/Comma-separated_values) format (`.csv`) or
the [Enhanced Character-Separated Values (ECSV)](https://docs.astropy.org/en/stable/io/ascii/ecsv.html) format (`.ecsv`).

### Quick example of the pointing list

[An example CSV file](examples/example_ppclist.csv) is available.

A quick example of the content is shown below.

| ppc_code    | ppc_ra             | ppc_dec        | ppc_pa | ppc_resolution | ppc_priority |
| ----------- | ------------------ | -------------- | ------ | -------------- | ------------ |
| Point_low_1 | 49.89556493946     | 41.51756603678 | 0.0    | L              | 0.0          |
| Point_low_2 | 48.89683299114     | 41.33165269544 | 0.0    | L              | 0.0          |
| Point_low_3 | 50.020406433000005 | 40.80069845724 | 0.0    | L              | 0.0          |
| Point_low_4 | 50.6446139007      | 42.03279417592 | 0.0    | L              | 0.0          |

### Content of the pointing list

Mandatory fields are listed below.

| Name           | Datatype | Unit   | Description                                                                                        |
| -------------- | -------- | ------ | -------------------------------------------------------------------------------------------------- |
| ppc_ra         | float    | degree | Right Ascension (ICRS at the reference epoch of 2000.0)                                            |
| ppc_dec        | float    | degree | Declination (ICRS at the reference epoch of 2000.0)                                                |
| ppc_resolution | str      |        | Grating used in the red optical arms. `L` for the low resolution and `M` for the medium resolution |

Optional fields are listed below.

| Name         | Datatype | Unit   | Description                         |
| ------------ | -------- | ------ | ----------------------------------- |
| ppc_pa       | float    | degree | Position angle                      |
| ppc_priority | float    |        | Priority of the ppc                 |
| ppc_code     | str      |        | A string identifier for the target. |

### Practical example

When submitting a user-defined pointing list, you need to consider the **number of pointing centers** and the **total exposure time per pointing center** (see also [configurations](PPP.md#different-observation-types) section).

Suppose you want to observe the COSMOS field with a **total integration time of 1 hour** at a fixed pointing center. You also consider a **unit exposure time of 15 minutes**. In this case, you need to prepare a pointing list like below.

| ppc_ra | ppc_dec | ppc_resolution |
| ------ | ------- | -------------- |
| 150.1  | 2.2     | L              |
| 150.1  | 2.2     | L              |
| 150.1  | 2.2     | L              |
| 150.1  | 2.2     | L              |

This will result in **four 15-minute exposure sequences (=unit exposure)** at the same pointing center. Each exposure sequence will be split into 2 exposures for better cosmic-ray rejection (see [the instrument web page](https://www.naoj.org/Instruments/PFS/status.html#cosmic-ray-rejection)). Therefore, your 15-minute exposure sequence will consist of 2 exposures of 7.5 minutes. This results in **8 visits (or 8 FITS files per arm per spectrograph)**.
