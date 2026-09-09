"""
Console entry point for the database application.

A systemd unit cannot compute a path, and a container command should not carry
one: the entry script lives inside the installed package, at a site-packages
path that contains the Python minor version and therefore moves under a
distribution upgrade. `photocat-app` resolves it at run time instead, so one
service definition works for a source checkout, a wheel and an image alike.
"""

import os
import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli

from pyKES.database_app.config import DEFAULT_CONFIG
from pyKES.database_app.deployment import describe_startup, verify_data_root

# Entry script Streamlit is asked to run. Its siblings in `pages/` are
# discovered relative to it, which is why the multipage app works from an
# installed wheel at all.
ENTRY_SCRIPT = "Home.py"

# Arguments that ask a question rather than starting a server. A data root is
# a precondition of serving, not of printing usage, and requiring one turned
# `photocat-app --help` into a traceback.
INFORMATIONAL_FLAGS = ("--help", "-h")


def main() -> None:
    """
    Run the database application, resolving its entry script from the package.

    The index is opened once before the server starts, so a data root that is
    absent or not writable stops the service rather than surfacing as a
    traceback on whichever page a user opens first, and the additive column
    migration happens once instead of inside a request.

    Changes to the package directory next. Streamlit reads
    ``.streamlit/config.toml`` relative to the *working directory* rather than
    to the script it runs, so the theme and the raised upload limit shipped
    inside the wheel apply only from there. Environment variables and
    command-line flags still take precedence, which is how a deployment
    overrides the server settings.

    Returns
    -------
    None : None
    """
    if not set(sys.argv[1:]) & set(INFORMATIONAL_FLAGS):
        print(describe_startup(verify_data_root(DEFAULT_CONFIG), DEFAULT_CONFIG),
              flush=True)

    package_directory = Path(__file__).resolve().parent
    os.chdir(package_directory)

    sys.argv = ["streamlit", "run", str(package_directory / ENTRY_SCRIPT),
                *sys.argv[1:]]

    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
