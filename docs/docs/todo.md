# Development notes

## TODO

- When `str` values are `None`, the validation crashes.
- Validation for string values are currently separated for required and optional keys and only the results (`success` flag) for required keys is used. Since malicious string can be found in optional columns, I need to consider it.
- Spacing between elements in the main panel is a bit weird. This is because I put a lot of placeholders. Maybe this is okay as it won't affect the validity of the validation.
- The heights of Results and Inputs tabs in the main section are set identical, so there is a lot of empty space. It is okay, but a bit strange looking.
- Coloring of the code block (e.g., `this style`) in Markdown panels does not look good. It seems that there is no control of this in the current version of panel.
