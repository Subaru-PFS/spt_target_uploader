"""Tests that drive the assembled ``target_uploader_app()``.

``test_relock_run_config_widgets.py`` pins the locking *policy* -- the pure
function, called directly. Nothing pinned the *wiring*: the closures
``target_uploader_app()`` registers on the widgets, and whether param can
actually call them.

That gap is not theoretical. A watcher was once declared

    def discard_pending_run_if_target_input_changed(_event):

while watching three parameters. A browser upload changes ``filename``,
``mime_type`` and ``value`` in a single batched ``param.update()``, which param
delivers as one call carrying one Event per changed parameter -- so every
upload raised ``TypeError``. Worse, the exception aborts param's watcher loop,
so the ``pn.bind`` that enables Validate never ran either: the file arrived and
every button stayed disabled. The whole unit suite stayed green.

So the uploads here go through Panel's own ``_process_events()``, the entry
point a real browser upload takes, rather than assigning the three parameters
one at a time -- assigning them separately produces three single-Event calls
and would not reproduce the batching that broke it.
"""

from base64 import b64encode
from types import SimpleNamespace

import panel as pn
import pytest

from pfs_target_uploader import pn_app
from pfs_target_uploader.utils.config import AppConfig

# The wiring under test never parses the upload -- it only reacts to the
# parameters changing -- so the payload just has to be bytes.
CSV = b"obj_id,ob_code,ra,dec,priority,exptime,resolution,reference_arm\n"

# The widget holders the assertions need. Captured as the app constructs them
# rather than dug out of the finished layout: these are the very objects the
# closures were handed, which is the point of the test, and a test that walks
# the sidebar would instead break every time the sidebar is rearranged.
HOLDERS = (
    "FileInputWidgets",
    "ValidateButtonWidgets",
    "RunPppButtonWidgets",
    "SubmitButtonWidgets",
    "ObsTypeWidgets",
)


class _Notifications:
    """Enough of pn.state.notifications for the app to configure and use it."""

    def __init__(self):
        self.position = None
        self.errors = []
        self.warnings = []

    def error(self, message, **kwargs):
        self.errors.append(message)

    def warning(self, message, **kwargs):
        self.warnings.append(message)

    def info(self, message, **kwargs):
        pass

    def clear(self):
        pass


@pytest.fixture
def app(monkeypatch, tmp_path):
    """Build the real app, with the config and notifications stubbed out.

    load_app_config() is replaced rather than pointed at a temporary
    .env.shared because target_uploader_app() calls it with validate_db=True,
    which would need an upload-ID database that has nothing to do with wiring.
    """
    monkeypatch.setattr(
        pn_app,
        "load_app_config",
        lambda **kwargs: AppConfig(output_dir=str(tmp_path), use_uid_db=False),
    )
    notifications = _Notifications()
    monkeypatch.setattr(
        type(pn.state), "notifications", property(lambda self: notifications)
    )

    captured = {}
    for name in HOLDERS:
        real = getattr(pn_app, name)

        def factory(*args, _real=real, _name=name, **kwargs):
            captured[_name] = obj = _real(*args, **kwargs)
            return obj

        monkeypatch.setattr(pn_app, name, factory)

    pn_app.target_uploader_app()

    return SimpleNamespace(
        target_input=captured["FileInputWidgets"].file_input,
        validate=captured["ValidateButtonWidgets"].validate,
        simulate=captured["RunPppButtonWidgets"].PPPrun,
        submit=captured["SubmitButtonWidgets"].submit,
        obs_type=captured["ObsTypeWidgets"].obs_type,
        notifications=notifications,
    )


def upload(file_input, payload, filename="targets.csv"):
    """Deliver a file the way the browser does: one batched property change."""
    file_input._process_events(
        {
            "filename": filename,
            "mime_type": "text/csv",
            "value": b64encode(payload).decode(),
        }
    )


def test_uploading_a_target_list_enables_validate_and_simulate(app):
    """The end state a user sees after picking a file, in the default mode."""
    assert app.validate.disabled is True
    assert app.simulate.disabled is True

    upload(app.target_input, CSV)

    assert app.target_input.value == CSV
    assert app.validate.disabled is False
    assert app.simulate.disabled is False
    # Nothing has been validated yet, let alone simulated.
    assert app.submit.disabled is True


def test_uploading_a_target_list_keeps_simulate_off_in_filler_mode(app):
    """filler has no pointing simulation; the upload must not offer one.

    Also pins that the obs_type argument of the binding is still live -- the
    upload path and the mode path meet in the same reactive call.
    """
    app.obs_type.value = "filler"

    upload(app.target_input, CSV)

    assert app.validate.disabled is False
    assert app.simulate.disabled is True


def test_a_new_target_list_disables_submit(app):
    """The watcher must run, not merely fail to raise.

    Submit is the one control whose state cannot survive a change of input:
    whatever it was enabled for is no longer what is loaded.
    """
    upload(app.target_input, CSV)
    app.submit.disabled = False  # stand in for a completed validation

    upload(app.target_input, CSV + b"1,a,150.0,2.0,1,900,M,n\n", filename="other.csv")

    assert app.submit.disabled is True
