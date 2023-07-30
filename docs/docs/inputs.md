# Input Target List

## Contents

There are required and optional fields. String values must consist of `[A-Za-z0-9_-+.]`.


### Required fields

Mandatory fields are listed below.

| Name       | Datatype   | Unit   | Description                                                                                        |
|------------|------------|--------|----------------------------------------------------------------------------------------------------|
| ob_code    | str        |        | A unique string to identify the entry                                                              |
| obj_id     | 64 bit int |        | Object ID                                                                                          |
| ra         | float      | degree | Right Ascension at the reference epoch specified by `equinox`                                      |
| dec        | float      | degree | Declination at the the reference epoch specified by `equinox`                                      |
| equinox    | str        |        | Equinox string (e.g., J2000.0, J2016.0, etc.)                                                      |
| exptime    | float      | second | Exposure time requested for the object                                                             |
| priority   | float      |        | Priority for the object within the list. Smaller value for higher priority                         |
| resolution | str        |        | Grating used in the red optical arms. `L` for the low resolution and `M` for the medium resolution |




### Optional fields

Optional fields are listed below.

| Name         | Datatype | Unit     | Default | Description                                |
|--------------|----------|----------|---------|--------------------------------------------|
| pmra         | float    | mas/year | 0       | Proper motion in right ascension direction |
| pmdec        | float    | mas/year | 0       | Proper motion in declination direction     |
| parallax     | float    | mas      | 1e-7    | Parallax                                   |
| tract        | int      |          |         | Tract ID                                   |
| patch        | int      |          |         | Patch name                                 |
| filter_g     | str      |          |         | Filter name for g-band                     |
| filter_r     | str      |          |         | Filter name for r-band                     |
| filter_i     | str      |          |         | Filter name for i-band                     |
| filter_z     | str      |          |         | Filter name for z-band                     |
| filter_y     | str      |          |         | Filter name for y-band                     |
| filter_j     | str      |          |         | Filter name for J-band                     |
| flux_g       | float    | nJy      |         | Flux in g-band                             |
| flux_r       | float    | nJy      |         | Flux in r-band                             |
| flux_i       | float    | nJy      |         | Flux in i-band                             |
| flux_z       | float    | nJy      |         | Flux in z-band                             |
| flux_y       | float    | nJy      |         | Flux in y-band                             |
| flux_j       | float    | nJy      |         | Flux in J-band                             |
| flux_error_g | float    | nJy      |         | Error in g-band flux                       |
| flux_error_r | float    | nJy      |         | Error in r-band flux                       |
| flux_error_i | float    | nJy      |         | Error in i-band flux                       |
| flux_error_z | float    | nJy      |         | Error in z-band flux                       |
| flux_error_y | float    | nJy      |         | Error in y-band flux                       |
| flux_error_j | float    | nJy      |         | Error in J-band flux                       |

### Filters

Currently, the following filters are registered in our database.

| Name      | Description                 |
|-----------|-----------------------------|
| g_hsc     | HSC g filter                |
| r_old_hsc | HSC r filter (old r filter) |
| r2_hsc    | HSC r2 filter               |
| i_old_hsc | HSC i filter (old i filter) |
| i2_hsc    | HSC i2 filter               |
| z_hsc     | HSC z filter                |
| y_hsc     | HSC Y filter                |
| g_ps1     | Pan-STARRS1 g filter        |
| r_ps1     | Pan-STARRS1 r filter        |
| i_ps1     | Pan-STARRS1 i filter        |
| z_ps1     | Pan-STARRS1 z filter        |
| y_ps1     | Pan-STARRS1 y filter        |
| bp_gaia   | Gaia BP filter              |
| rp_gaia   | Gaia RP filter              |
| g_gaia    | Gaia G filter               |
| u_sdss    | SDSS u filter               |
| g_sdss    | SDSS g filter               |
| r_sdss    | SDSS r filter               |
| i_sdss    | SDSS i filter               |
| z_sdss    | SDSS z filter               |


## File format

Input target list must be a Comma-separated values (CSV) file.


## Example

An example CSV file can be found in [this link](examples/example_targetlist.csv)
