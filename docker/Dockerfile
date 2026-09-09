# Pinned on purpose. A Python minor bump can move a dependency off a prebuilt
# wheel, which turns a two-minute build into a source compile of scipy that
# wants a Fortran toolchain and then runs out of memory on a small droplet, so
# it is a reviewed change rather than a default.
ARG PYTHON_VERSION=3.12

# Matches `uv_build>=0.11.13,<0.12` in pyproject.toml and the version the
# release workflow uses.
ARG UV_VERSION=0.11.13


# One stage for the sole purpose of resolving ${UV_VERSION}: `COPY --from` takes
# no variable expansion ("variable expansion is not supported for --from"),
# while `FROM` reads the global-scope ARG, so naming the stage here is what
# keeps the pinned version in one place instead of two.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


# ============================================================== builder stage
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    # Turns "no wheel published for this Python" into an immediate, readable
    # failure instead of a long source build. Every one of these ships
    # manylinux wheels for the pinned interpreter.
    UV_NO_BUILD_PACKAGE="numpy scipy pandas pyarrow h5py matplotlib pillow kiwisolver"

WORKDIR /src

# Keyed on the manifest and the lock alone, so editing any .py file below does
# not reinstall scipy — and the droplet then pulls a few megabytes on a
# code-only change rather than the whole image. README.md is here because
# uv_build reads project.readme.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# --locked, never --frozen: it fails the build when the lock does not match
# pyproject.toml, rather than installing a stale dependency set. --no-editable
# because `uv sync` installs the root project editable by default, and a .pth
# file pointing into /src does not survive the copy into the runtime stage.
COPY src/pyKES ./src/pyKES
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


# ============================================================== runtime stage
# Same base and the same Python minor version as the builder: the virtual
# environment carries absolute paths in pyvenv.cfg and in every console-script
# shebang, so a different interpreter path silently breaks it.
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG PYTHON_VERSION
ARG GIT_SHA=unknown
ARG IMAGE_TAG=dev

# Fixed rather than a build argument that varies. The bind-mounted data
# directory on the host must be owned by this id, and an image whose user
# changed per host could not be a prebuilt artefact.
ARG APP_UID=10001
ARG APP_GID=10001

LABEL org.opencontainers.image.title="pyKES photocatalysis database" \
      org.opencontainers.image.source="https://github.com/jschneidewind/pyKES" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.licenses="MIT"

# curl for the healthcheck, sqlite3 so looking at the index is a one-liner in
# the runbook, tini so SIGTERM reaches Streamlit instead of being swallowed by
# PID 1. No compilers: the numpy, scipy, h5py and pillow wheels vendor their
# own libopenblas and libhdf5, and matplotlib only needs the Agg backend.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl sqlite3 tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid ${APP_GID} photocat \
 && useradd --uid ${APP_UID} --gid ${APP_GID} --create-home \
            --shell /usr/sbin/nologin photocat

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/photocat \
    # matplotlib and Streamlit write caches under the home directory; without
    # these they warn on every start and rebuild the font cache each time.
    MPLCONFIGDIR=/home/photocat/.cache/matplotlib \
    XDG_CACHE_HOME=/home/photocat/.cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PHOTOCAT_GIT_SHA=${GIT_SHA} \
    PHOTOCAT_IMAGE_TAG=${IMAGE_TAG} \
    PHOTOCAT_ENV=production \
    PHOTOCAT_ALLOW_DEV_LOGIN=0 \
    # The same path inside as outside. `uploads.stored_path` is relative now,
    # but keeping the paths identical also means a rebuild run inside the
    # container and one run from a host virtual environment agree, and it lets
    # staging mount a different host directory here without rewriting a
    # restored index.
    PHOTOCAT_DATA_ROOT=/srv/photocat/data \
    # 0.0.0.0, which inverts the instruction in the provisioning guide and
    # has to: inside a container 127.0.0.1 is the container's own loopback and
    # host nginx cannot reach it. The confinement moves to the published port,
    # which compose binds to 127.0.0.1 on the host.
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer \
    STREAMLIT_LOGGER_LEVEL=info

# Build-time assertions. Each of these failures is silent at run time, and two
# of them are traps this package sets: an absent schema directory makes
# `load_entity_schemas` return {} *by design*, which removes validation, the
# download templates and every reference declaration at once; and a missing
# pages/ directory leaves a working app with no navigation.
RUN set -eu; \
    photocat-app --help > /dev/null; \
    python -c "\
from pyKES.database.entity_schema import load_entity_schemas;\
from pyKES.database.index_schema import ENTITY_TYPES;\
from pyKES.database_app import launch;\
from pyKES.utilities.version_information import get_pykes_version, UNKNOWN_VERSION;\
from pathlib import Path;\
app = Path(launch.__file__).parent;\
schemas = load_entity_schemas();\
assert set(schemas) == set(ENTITY_TYPES), schemas;\
assert (app / 'Home.py').is_file();\
assert len(list((app / 'pages').glob('[0-9]*.py'))) == 5;\
assert (app / '.streamlit' / 'config.toml').is_file();\
assert get_pykes_version() != UNKNOWN_VERSION, 'no distribution metadata';\
print('image assertions passed')"

USER ${APP_UID}:${APP_GID}
WORKDIR /home/photocat
EXPOSE 8501

# The start period covers the cold import of pandas, scipy, plotly, matplotlib
# and pyarrow plus Streamlit's own boot, which is several seconds; too short a
# period reports a healthy container as unhealthy and compose restarts it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

# `photocat-app` resolves the entry script from the installed package, opens
# the index once so an unwritable or absent data root stops the container, and
# logs which build came up against which database.
ENTRYPOINT ["/usr/bin/tini", "--", "photocat-app"]
