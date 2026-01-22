#!/usr/bin/env python3
"""Configuration management for PFS Target Uploader.

This module provides dataclass-based configuration loading from .env.shared files,
with validation and type conversion.
"""

import os
from dataclasses import dataclass, field
from pprint import pformat

from dotenv import dotenv_values
from loguru import logger


@dataclass(frozen=True)
class AppConfig:
    """Configuration for the PFS Target Uploader application.

    Attributes
    ----------
    output_dir : str
        Directory for data file output. Required.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    max_exetime : int
        Maximum execution time for PPP in seconds. 0 means no limit.
    ppp_quiet : bool
        Whether to suppress verbose PPP output.
    clustering_algorithm : str
        Target clustering algorithm (HDBSCAN, DBSCAN, FAST_HDBSCAN).
    ann_file : str | None
        Path to announcement file, or None if not configured.
    uploadid_db : str | None
        Filename of upload ID database, or None if not using DB.
    db_path : str | None
        Full path to upload ID database (OUTPUT_DIR + UPLOADID_DB).
    use_uid_db : bool
        Whether to use upload ID database.
    min_fluxmag_queue : float | None
        Min AB magnitude for queue obs_type (bright limit).
    min_fluxmag_classical : float | None
        Min AB magnitude for classical obs_type (bright limit).
    min_fluxmag_filler : float | None
        Min AB magnitude for filler obs_type (bright limit).
    max_fluxmag : float | None
        Max AB magnitude (faint limit, shared across all modes).
    raw_config : dict[str, str]
        Original dotenv configuration dict for any additional keys.
    """

    output_dir: str
    log_level: str = "INFO"
    max_exetime: int = 900
    ppp_quiet: bool = True
    clustering_algorithm: str = "HDBSCAN"
    ann_file: str | None = None
    uploadid_db: str | None = None
    db_path: str | None = None
    use_uid_db: bool = False
    min_fluxmag_queue: float | None = None
    min_fluxmag_classical: float | None = None
    min_fluxmag_filler: float | None = None
    max_fluxmag: float | None = None
    raw_config: dict[str, str] = field(default_factory=dict)

    def format_for_logging(self, include_raw: bool = True) -> str:
        """Format configuration as a multi-line string for logging.

        Parameters
        ----------
        include_raw : bool
            If True, include raw dotenv dictionary. Default: True.

        Returns
        -------
        str
            Formatted configuration string with key settings.
        """
        lines = ["Application Configuration:"]

        # Build list of (key, value) tuples for sorting
        items = []

        if self.ann_file:
            items.append(("ANN_FILE", self.ann_file))

        items.append(("CLUSTERING_ALGORITHM", self.clustering_algorithm))

        if self.use_uid_db:
            items.append(("DB_PATH", self.db_path))

        items.append(("LOG_LEVEL", self.log_level))
        items.append(("MAX_EXETIME", f"{self.max_exetime} sec"))

        if self.max_fluxmag is not None:
            items.append(("MAX_FLUXMAG", str(self.max_fluxmag)))

        # Flux magnitude settings
        if self.min_fluxmag_classical is not None:
            items.append(("MIN_FLUXMAG_CLASSICAL", str(self.min_fluxmag_classical)))
        if self.min_fluxmag_filler is not None:
            items.append(("MIN_FLUXMAG_FILLER", str(self.min_fluxmag_filler)))
        if self.min_fluxmag_queue is not None:
            items.append(("MIN_FLUXMAG_QUEUE", str(self.min_fluxmag_queue)))

        items.append(("OUTPUT_DIR", self.output_dir))
        items.append(("PPP_QUIET", str(self.ppp_quiet)))

        if self.use_uid_db:
            items.append(("UPLOADID_DB", self.uploadid_db))
        else:
            items.append(("UPLOADID_DB", "Not used"))

        # Sort by key name and format
        for key, value in sorted(items):
            lines.append(f"  {key}: {value}")

        # Show any additional config from raw_config not covered above
        covered_keys = {
            "OUTPUT_DIR",
            "LOG_LEVEL",
            "MAX_EXETIME",
            "PPP_QUIET",
            "CLUSTERING_ALGORITHM",
            "ANN_FILE",
            "UPLOADID_DB",
            "MIN_FLUXMAG_QUEUE",
            "MIN_FLUXMAG_CLASSICAL",
            "MIN_FLUXMAG_FILLER",
            "MAX_FLUXMAG",
        }
        extra_keys = set(self.raw_config.keys()) - covered_keys
        if extra_keys:
            lines.append("  Additional settings:")
            for key in sorted(extra_keys):
                lines.append(f"    {key}: {self.raw_config[key]}")

        # Include raw dotenv configuration if requested
        if include_raw and self.raw_config:
            lines.append("")
            lines.append("Raw configuration from .env.shared:")
            # Indent each line of pformat output
            formatted = pformat(self.raw_config, width=80, sort_dicts=True)
            for line in formatted.split("\n"):
                lines.append(f"  {line}")

        return "\n".join(lines)


def _parse_bool_int(value: str | None, default: bool) -> bool:
    """Parse bool from int string (e.g., "1" -> True, "0" -> False).

    Parameters
    ----------
    value : str | None
        String value to parse (e.g., "1", "0").
    default : bool
        Default value if parsing fails or value is None.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    if value is None:
        return default
    try:
        return bool(int(value))
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid boolean value '{value}', using default: {default}"
        )
        return default


def _parse_optional_float(
    config: dict[str, str],
    key: str,
    key_label: str | None = None,
) -> float | None:
    """Parse optional float with logging on parse errors.

    Parameters
    ----------
    config : dict[str, str]
        Configuration dictionary.
    key : str
        Key to parse from config.
    key_label : str | None
        Human-readable label for logging. If None, uses key.

    Returns
    -------
    float | None
        Parsed float value, or None if key not present or empty.
    """
    label = key_label or key

    if key not in config or config[key] == "":
        return None

    try:
        value = float(config[key])
        logger.info(f"{label} is set to {value}")
        return value
    except ValueError:
        logger.warning(f"Invalid {label} value: {config[key]}")
        return None


def _validate_magnitude_ranges(
    min_fluxmag_queue: float | None,
    min_fluxmag_classical: float | None,
    min_fluxmag_filler: float | None,
    max_fluxmag: float | None,
) -> None:
    """Validate that min_mag < max_mag for all observation modes.

    Parameters
    ----------
    min_fluxmag_queue : float | None
        Minimum flux magnitude for queue mode.
    min_fluxmag_classical : float | None
        Minimum flux magnitude for classical mode.
    min_fluxmag_filler : float | None
        Minimum flux magnitude for filler mode.
    max_fluxmag : float | None
        Maximum flux magnitude (shared across all modes).

    Raises
    ------
    ValueError
        If min_mag > max_mag for any observation mode.
    """
    for mode_name, min_mag in [
        ("queue", min_fluxmag_queue),
        ("classical", min_fluxmag_classical),
        ("filler", min_fluxmag_filler),
    ]:
        if min_mag is not None and max_fluxmag is not None:
            if min_mag > max_fluxmag:
                error_msg = (
                    f"Configuration error for {mode_name} mode: "
                    f"MIN_FLUXMAG_{mode_name.upper()} ({min_mag}) > MAX_FLUXMAG ({max_fluxmag}). "
                    f"MIN_FLUXMAG should be brighter (smaller value) than MAX_FLUXMAG."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)


def _resolve_ann_file(
    config: dict[str, str],
    validate: bool,
) -> str | None:
    """Resolve and optionally validate announcement file path.

    Parameters
    ----------
    config : dict[str, str]
        Configuration dictionary.
    validate : bool
        If True, check that the file exists.

    Returns
    -------
    str | None
        Path to announcement file, or None if not configured or not found.
    """
    if "ANN_FILE" not in config:
        return None

    if config["ANN_FILE"] == "":
        return None

    ann_file = config["ANN_FILE"]

    if validate:
        if not os.path.exists(ann_file):
            logger.error(f"{ann_file} not found")
            return None
        else:
            logger.info(f"{ann_file} found")

    return ann_file


def _resolve_db_path(
    config: dict[str, str],
    output_dir: str,
    validate: bool,
) -> tuple[str | None, str | None, bool]:
    """Resolve database path and return tuple of (uploadid_db, db_path, use_uid_db).

    Parameters
    ----------
    config : dict[str, str]
        Configuration dictionary.
    output_dir : str
        Output directory path.
    validate : bool
        If True, validate that the database file exists.

    Returns
    -------
    tuple[str | None, str | None, bool]
        Tuple of (uploadid_db, db_path, use_uid_db).

    Raises
    ------
    FileNotFoundError
        If validate=True and database file doesn't exist.
    """
    uploadid_db = config.get("UPLOADID_DB")

    if uploadid_db is None:
        logger.info("No upload ID database is used. Scan output directories directly.")
        return None, None, False

    db_path = os.path.join(output_dir, uploadid_db)

    if validate:
        if os.path.exists(db_path):
            logger.info(f"{db_path} found")
        else:
            logger.error(f"{db_path} not found")
            raise FileNotFoundError(f"{db_path} not found")

    return uploadid_db, db_path, True


def load_app_config(
    env_file: str = ".env.shared",
    *,
    create_output_dir: bool = True,
    validate_db: bool = True,
    validate_ann_file: bool = True,
) -> AppConfig:
    """Load and validate application configuration from .env.shared file.

    Parameters
    ----------
    env_file : str
        Path to the environment file. Default: ".env.shared"
    create_output_dir : bool
        If True, create OUTPUT_DIR if it doesn't exist. Default: True
    validate_db : bool
        If True, validate that the database file exists. Default: True
    validate_ann_file : bool
        If True, validate that the announcement file exists. Default: True

    Returns
    -------
    AppConfig
        Validated configuration object.

    Raises
    ------
    KeyError
        If required configuration key OUTPUT_DIR is missing.
    FileNotFoundError
        If validate_db=True and database file doesn't exist.
    ValueError
        If MIN_FLUXMAG > MAX_FLUXMAG for any observation mode.
    """
    config = dotenv_values(env_file)

    # OUTPUT_DIR is required
    if "OUTPUT_DIR" not in config:
        raise KeyError("OUTPUT_DIR is required in configuration")

    output_dir = config["OUTPUT_DIR"]

    # Create output directory if requested
    if create_output_dir:
        if os.path.exists(output_dir):
            logger.info(f"{output_dir} already exists.")
        else:
            os.makedirs(output_dir)
            logger.info(f"{output_dir} created.")

    # Parse simple config values with defaults
    log_level = config.get("LOG_LEVEL", "INFO")

    max_exetime = 900
    if "MAX_EXETIME" in config:
        try:
            max_exetime = int(config["MAX_EXETIME"])
        except ValueError:
            logger.warning(
                f"Invalid MAX_EXETIME value: {config['MAX_EXETIME']}, using default: 900"
            )

    ppp_quiet = _parse_bool_int(config.get("PPP_QUIET"), default=True)

    clustering_algorithm = config.get("CLUSTERING_ALGORITHM", "HDBSCAN")

    # Resolve announcement file
    ann_file = _resolve_ann_file(config, validate=validate_ann_file)

    # Resolve database path
    uploadid_db, db_path, use_uid_db = _resolve_db_path(
        config, output_dir, validate=validate_db
    )

    # Parse flux magnitude settings
    min_fluxmag_queue = _parse_optional_float(
        config, "MIN_FLUXMAG_QUEUE", "MIN_FLUXMAG_QUEUE"
    )
    min_fluxmag_classical = _parse_optional_float(
        config, "MIN_FLUXMAG_CLASSICAL", "MIN_FLUXMAG_CLASSICAL"
    )
    min_fluxmag_filler = _parse_optional_float(
        config, "MIN_FLUXMAG_FILLER", "MIN_FLUXMAG_FILLER"
    )
    max_fluxmag = _parse_optional_float(
        config, "MAX_FLUXMAG", "MAX_FLUXMAG"
    )

    # Validate magnitude ranges
    _validate_magnitude_ranges(
        min_fluxmag_queue,
        min_fluxmag_classical,
        min_fluxmag_filler,
        max_fluxmag,
    )

    return AppConfig(
        output_dir=output_dir,
        log_level=log_level,
        max_exetime=max_exetime,
        ppp_quiet=ppp_quiet,
        clustering_algorithm=clustering_algorithm,
        ann_file=ann_file,
        uploadid_db=uploadid_db,
        db_path=db_path,
        use_uid_db=use_uid_db,
        min_fluxmag_queue=min_fluxmag_queue,
        min_fluxmag_classical=min_fluxmag_classical,
        min_fluxmag_filler=min_fluxmag_filler,
        max_fluxmag=max_fluxmag,
        raw_config=dict(config),
    )


def load_minimal_config(env_file: str = ".env.shared") -> AppConfig:
    """Load minimal configuration without full validation.

    Use this for admin app or CLI tools that only need OUTPUT_DIR.
    Does not validate database existence or create directories.

    Parameters
    ----------
    env_file : str
        Path to the environment file. Default: ".env.shared"

    Returns
    -------
    AppConfig
        Configuration object with minimal validation.

    Raises
    ------
    KeyError
        If required configuration key OUTPUT_DIR is missing.
    """
    return load_app_config(
        env_file,
        create_output_dir=False,
        validate_db=False,
        validate_ann_file=False,
    )


def get_min_fluxmag_for_obstype(
    obs_type: str,
    config: AppConfig,
) -> float | None:
    """Select appropriate minimum flux magnitude based on observation type.

    Parameters
    ----------
    obs_type : str
        Observation type: "queue", "classical", or "filler"
    config : AppConfig
        Application configuration object.

    Returns
    -------
    float | None
        Mode-specific minimum flux magnitude (brightest limit),
        or None if not configured.
    """
    if obs_type == "queue":
        return config.min_fluxmag_queue
    elif obs_type == "classical":
        return config.min_fluxmag_classical
    elif obs_type == "filler":
        return config.min_fluxmag_filler
    return None
