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

# Entry script Streamlit is asked to run. Its siblings in `pages/` are
# discovered relative to it, which is why the multipage app works from an
# installed wheel at all.
ENTRY_SCRIPT = "Home.py"


def main() -> None:
    """
    Run the database application, resolving its entry script from the package.

    Changes to the package directory first. Streamlit reads
    ``.streamlit/config.toml`` relative to the *working directory* rather than
    to the script it runs, so the theme and the raised upload limit shipped
    inside the wheel apply only from there. Environment variables and
    command-line flags still take precedence, which is how a deployment
    overrides the server settings.

    Returns
    -------
    None : None
    """
    package_directory = Path(__file__).resolve().parent
    os.chdir(package_directory)

    sys.argv = ["streamlit", "run", str(package_directory / ENTRY_SCRIPT),
                *sys.argv[1:]]

    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
