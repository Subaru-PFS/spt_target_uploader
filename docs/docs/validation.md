# Validation

Validation of a input target list is carried out in 4 stages.

## Stages

### Stage 0

At **Stage 0**, whether a proper readable CSV file is provided is checked.

!!! danger "Errors are raised in the following cases"

    - When the `validation`button is clicked without selecting a input file, an error will be raised.
    - When `pandas.read_csv()` fails to read the input CSV file, an error will be raised. This is likely caused by wrong formats in the fields for numbers.


### Stage 1

At **Stage 1**, the input column names are checked against the required and optional keys.

!!! danger "Errors are raised in the following case"

    - A mandatory keyword `key name` is missing. Please add them with proper values.


!!! warning "Warnings are raised in the following case"

    - A mandatory keyword `key name` is missing. Please add them with proper values.

### Stage 2

At **Stage 2**, cells with string data will be validated to contain only allowed characters, `[A-Za-z0-9_-+.]`.

!!! danger "Errors are raised in the following case"

    - Characters not in `[A-Za-z0-9_-+.]` are detected.


### Stage 3

At **Stage 3**, values are checked whether they are in allowed ranges.

!!! note "Following checks are conducted and errors are raised when violations are detected"

    - $0 \le \mathrm{ra} \le 360$.
    - $-90 \le \mathrm{dec} \le 360$.
    - `equinox` must start with `J` (Julien epoch) or `B` (Besselian epoch) followed by string which can be converted to a float number. Note that down to the first digit after the decimal is considered.
    - `priority` must be positive.
    - `exptime`must be positive.
    - `resolution` must be either `L` or `M`.


### Stage 4

At **Stage 4**, `ob_code` are checked not to have duplicates.

!!! danger "Errors are raised in the following case"

    - Duplicates in `ob_code` are detected. They must be unique within the list.

## Results

### Side panel (left)

#### <u>Input file selector</u>

Press "Browse" button to select a CSV file to be validated.

#### <u>"Validation" button</u>

Press "Validation" button to start the validation process.

#### <u>Stage indicators</u>

Circles show the status of each stage. Meaning of each color is the following.

**White**
: Validation process was not carried out at **Stage** because it failed at an earlier Stage.

<span style="color: darkorange;">**Yellow**</span>
: Validation at **Stage** ended with warnings.

<span style="color: darkgreen;">**Green**</span>
: Validation at **Stage** was successful without errors and warnings.

#### <u>Number of objects</u>

Total number of objects in the list is shown.


#### <u>Number of objects</u>

Total fiberhours requested in the list is shown.

#### <u>Breakdown table by priority</u>

A table showing the number of objects and fiberhours in each priority group is presented.

## Main panel (right)

The details of validation processes are shown in the main panel.

First, errors are shown, followed by warnings. The successful checks are shown at the bottom.

Errors must be fixed before submitting the target list.  Warnings can be ignored, but please consider carefully when ignoring them.