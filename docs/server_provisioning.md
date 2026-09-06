# Phase 0 — provisioning the database server

Everything needed to get from a bare machine to a TLS-protected, Authelia-gated
Streamlit application serving the photocatalysis database, before any of the
application code exists.

Phase 0 gates the *deployment*, not the code: Phase 1 (the data layer) can be
built in parallel. What Phase 0 must settle is the parts that are painful to
change once real data is on the machine — the certificate, the authentication
route, and the directory layout.

The decisions this document implements:

| Decision | Choice |
| --- | --- |
| Authentication | **Authelia**, in front of nginx, with TOTP |
| Name collisions | Both kept, newcomer suffixed `__v2` |
| Editing and deletion | **Owner + admin only** |
| Editing a shared ancestor | Owner + admin, descendants recomputed |
| Entity types | experiment, catalyst_batch, finished_semiconductor, precursor_semiconductor, precursor_chemical, commercial_chemical, stock_solution, other_entity |
| Original uploads | **Kept indefinitely** |

---

## 1. What the machine needs

Measured against the group's own data (§7), 10 000 experiments come to roughly
1.4 GB of payloads plus the same again of retained uploads, and the index itself
stays in the tens of megabytes. Nothing here is demanding.

| Resource | Minimum | Comfortable |
| --- | --- | --- |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 50 GB | 100 GB (SSD) |
| OS | Debian 12 or Ubuntu 22.04 LTS | either, kept patched |

One host runs all three services: nginx, the Streamlit application and Authelia.
Authelia's container is under 20 MB and uses under 30 MB of RAM, so it does not
change the sizing.

**Before starting, obtain:** a DNS name that resolves to the machine
(`photocat.example.uni-x.de`), a TLS certificate for it — the institutional CA
is preferable to Let's Encrypt if the host is not reachable from the public
internet — and the list of people who should have accounts.

---

## 2. Layout and system user

The application runs as an unprivileged user that owns the data and nothing
else. Never as root, and never as a login user.

```bash
sudo adduser --system --group --home /srv/photocat --shell /usr/sbin/nologin photocat

sudo -u photocat mkdir -p /srv/photocat/{app,data,venv,logs}
sudo -u photocat mkdir -p /srv/photocat/data/{payloads,uploads}
sudo chmod 750 /srv/photocat/data
```

| Path | Holds | Backed up |
| --- | --- | --- |
| `/srv/photocat/app` | the application checkout | from git |
| `/srv/photocat/data/index.sqlite` | the searchable index | **yes** |
| `/srv/photocat/data/uploads` | every uploaded file, verbatim | **yes** |
| `/srv/photocat/data/payloads` | per-experiment HDF5, derived | no — rebuildable |
| `/srv/photocat/venv` | Python environment | no |

`payloads/` is deliberately outside the backup: it is reconstructed from
`uploads/` by `rebuild_index`, which is the whole reason uploads are kept.

---

## 3. Python environment

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential nginx sqlite3

sudo -u photocat python3 -m venv /srv/photocat/venv
sudo -u photocat /srv/photocat/venv/bin/pip install --upgrade pip
sudo -u photocat /srv/photocat/venv/bin/pip install pyKES
```

For a checkout being developed against, install it editable instead:

```bash
sudo -u photocat /srv/photocat/venv/bin/pip install -e /srv/photocat/app
```

Verify the data layer imports and can create an index before going further:

```bash
sudo -u photocat /srv/photocat/venv/bin/python -c "
from pyKES.database.index_schema import IndexPaths, open_index, read_index_schema_version
paths = IndexPaths(root='/srv/photocat/data')
print('index schema', read_index_schema_version(open_index(paths)))"
```

---

## 4. The Streamlit service

Streamlit binds to localhost only. Everything from outside arrives through
nginx, so the application port is never exposed.

`/etc/systemd/system/photocat.service`:

```ini
[Unit]
Description=Photocatalysis database
After=network.target

[Service]
Type=simple
User=photocat
Group=photocat
WorkingDirectory=/srv/photocat/app
Environment="PHOTOCAT_DATA_ROOT=/srv/photocat/data"
ExecStart=/srv/photocat/venv/bin/streamlit run Home.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true \
    --server.maxUploadSize 1000 \
    --server.enableCORS false \
    --server.enableXsrfProtection true \
    --browser.gatherUsageStats false
Restart=on-failure
RestartSec=5

# The service needs nothing outside its own directory.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/srv/photocat/data /srv/photocat/logs

[Install]
WantedBy=multi-user.target
```

`--server.maxUploadSize 1000` matters: the default is 200 MB, and at roughly
140 KB per experiment a 1000-experiment batch would exceed it. 1000 MB leaves
generous headroom.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now photocat
sudo systemctl status photocat
```

---

## 5. Authelia

Authelia is a forward-authentication service: nginx asks it about every request
and refuses to proxy anything it does not approve. The application never sees an
unauthenticated request, and neither do the payload files.

### 5.1 Install

```bash
sudo mkdir -p /etc/authelia /var/lib/authelia
sudo useradd --system --no-create-home --shell /usr/sbin/nologin authelia
# Fetch the release binary for your architecture from the project's releases
# page, install it to /usr/local/bin/authelia, then:
sudo chown -R authelia:authelia /etc/authelia /var/lib/authelia
sudo chmod 700 /etc/authelia
```

### 5.2 Secrets

Never in the configuration file — Authelia reads them from files.

```bash
for secret in jwt session storage; do
  sudo sh -c "openssl rand -hex 48 > /etc/authelia/${secret}.secret"
done
sudo chown authelia:authelia /etc/authelia/*.secret
sudo chmod 600 /etc/authelia/*.secret
```

### 5.3 Configuration

`/etc/authelia/configuration.yml`:

```yaml
theme: light
default_redirection_url: https://photocat.example.uni-x.de/

server:
  address: 'tcp://127.0.0.1:9091'

log:
  level: info
  file_path: /var/lib/authelia/authelia.log

identity_validation:
  reset_password:
    jwt_secret_file: /etc/authelia/jwt.secret

authentication_backend:
  # A YAML file is the right backend for a research group. Point this at LDAP
  # instead if the institution offers one and you want accounts to follow
  # employment automatically.
  file:
    path: /etc/authelia/users.yml
    password:
      algorithm: argon2id

access_control:
  default_policy: deny
  rules:
    # Two factors for everyone. There is no public area of this application.
    - domain: photocat.example.uni-x.de
      policy: two_factor

session:
  name: authelia_session
  secret_file: /etc/authelia/session.secret
  expiration: 12h
  inactivity: 2h
  cookies:
    - domain: example.uni-x.de
      authelia_url: https://auth.example.uni-x.de

regulation:
  # Lock an account briefly after repeated failures, which is what makes a
  # password-guessing attempt expensive.
  max_retries: 3
  find_time: 2m
  ban_time: 10m

storage:
  encryption_key_file: /etc/authelia/storage.secret
  local:
    path: /var/lib/authelia/db.sqlite3

notifier:
  # Enrolment and password-reset messages. A filesystem notifier keeps
  # everything on this machine; swap in SMTP if the group prefers email.
  filesystem:
    filename: /var/lib/authelia/notification.txt
```

### 5.4 Users

Generate each password hash with Authelia itself, never by hand:

```bash
authelia crypto hash generate argon2 --password 'chosen-password'
```

`/etc/authelia/users.yml`:

```yaml
users:
  jschneidewind:
    displayname: "Jacob Schneidewind"
    password: "$argon2id$v=19$m=65536,t=3,p=4$..."
    email: jacob@example.uni-x.de
    groups:
      - admins
      - users
  ae:
    displayname: "…"
    password: "$argon2id$v=19$..."
    email: ae@example.uni-x.de
    groups:
      - users
```

The `admins` group is what the application reads to decide who may edit
another person's entry. Everyone else is in `users`: they can read everything
and edit their own entries.

Run it as a service (`/etc/systemd/system/authelia.service`) with
`ExecStart=/usr/local/bin/authelia --config /etc/authelia/configuration.yml`,
`User=authelia`, and the same hardening directives as the application unit.

---

## 6. nginx

Three things have to be right, and each has a failure mode that looks like
something else.

`/etc/nginx/sites-available/photocat`:

```nginx
server {
    listen 80;
    server_name photocat.example.uni-x.de auth.example.uni-x.de;
    return 301 https://$host$request_uri;
}

# Authelia's own login page.
server {
    listen 443 ssl http2;
    server_name auth.example.uni-x.de;

    ssl_certificate     /etc/ssl/certs/photocat.crt;
    ssl_certificate_key /etc/ssl/private/photocat.key;

    location / {
        proxy_pass http://127.0.0.1:9091;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $remote_addr;
    }
}

server {
    listen 443 ssl http2;
    server_name photocat.example.uni-x.de;

    ssl_certificate     /etc/ssl/certs/photocat.crt;
    ssl_certificate_key /etc/ssl/private/photocat.key;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # A 40-experiment batch is a few megabytes; leave room for a large one.
    client_max_body_size 1000M;

    # The internal endpoint nginx asks about every request.
    location /internal/authelia/authz {
        internal;
        proxy_pass http://127.0.0.1:9091/api/authz/auth-request;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URL $scheme://$http_host$request_uri;
        proxy_set_header X-Forwarded-For $remote_addr;
    }

    location / {
        auth_request /internal/authelia/authz;

        # Authelia returns the authenticated identity in these headers; they are
        # copied onto the proxied request so the application knows who is here.
        auth_request_set $user  $upstream_http_remote_user;
        auth_request_set $groups $upstream_http_remote_groups;
        proxy_set_header Remote-User   $user;
        proxy_set_header Remote-Groups $groups;

        # Send an unauthenticated visitor to the login page.
        error_page 401 =302 https://auth.example.uni-x.de/?rd=$scheme://$http_host$request_uri;

        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streamlit talks over a WebSocket. Without these three lines the page
        # loads and then appears frozen — which reads as an application bug and
        # is not one.
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # Payload files are served by nginx, not by Python — but only after the
    # application has authorised the request and issued an internal redirect.
    # `internal` is what makes the path unreachable from outside; without it the
    # entire dataset is simply on the web.
    location /payloads/ {
        internal;
        alias /srv/photocat/data/payloads/;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/photocat /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### The identity header, and the one thing to verify early

The application reads the signed-in user from the `Remote-User` header via
`st.context.headers`. **That reflects the `/_stcore/stream` WebSocket request,
not the initial page request**, so the header must be set on the location that
proxies the WebSocket — which above is the same `location /`, and is exactly why
it is not split into a separate block.

Verify it before writing any application code that depends on it:

```python
# scratch.py, run through the proxy, not directly against port 8501
import streamlit as st
st.write("headers:", dict(st.context.headers))
```

`Remote-User` must be present. If it is not, the application cannot attribute
uploads and the ownership rule cannot be enforced.

---

## 7. Sizing against the group's own data

Measured by ingesting the real 44-experiment plate dataset
(`260903_AE857_AE859.h5`, 6.30 MB):

| Quantity | Measured | At 10 000 experiments |
| --- | --- | --- |
| Ingestion, end to end | 30 ms per experiment | ~5 min for a full rebuild |
| Payload per experiment | 136 KB | **1.4 GB** |
| Retained uploads | ≈ source size | **~1.4 GB** |
| Index | ~2 KB per entity | tens of MB |

Two things are worth knowing. Ingestion at 30 ms per experiment means a
40-experiment upload completes in **about 1.3 seconds**, so it runs
synchronously with a progress bar and needs no job queue. And gzip only takes
6.30 MB to 5.99 MB on this data — **1.05×, not the 1.7× measured on an older
fixture** — because per-file HDF5 overhead across 44 separate payloads offsets
most of the saving on high-entropy sensor traces. Budget ~140 KB per experiment
rather than assuming compression will help much.

Total for 10 000 experiments: **under 3 GB**. The 50 GB minimum is generous.

---

## 8. Firewall, backups, maintenance

```bash
sudo ufw default deny incoming
sudo ufw allow from 10.0.0.0/8 to any port 22    # SSH from inside only
sudo ufw allow 443/tcp
sudo ufw enable
```

Back up the two irreplaceable things nightly. `sqlite3 .backup` is used rather
than `cp` because it is safe against a concurrent writer:

```bash
#!/bin/bash
# /usr/local/bin/photocat-backup — run from cron as root
set -euo pipefail
STAMP=$(date +%Y%m%d)
DEST=/backup/photocat/$STAMP
mkdir -p "$DEST"

sqlite3 /srv/photocat/data/index.sqlite ".backup '$DEST/index.sqlite'"
rsync -a --delete /srv/photocat/data/uploads/ "$DEST/uploads/"

find /backup/photocat -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
```

**Test the restore path once, before you need it**: copy a backup to a scratch
directory, run `rebuild_index` against it, and confirm the entity count matches.
A backup nobody has restored is a hypothesis.

Routine maintenance: unattended security upgrades, `ANALYZE` after any bulk
ingestion (`analyse_index`), and a monthly look at the admin page for dangling
references and drifted metadata keys.

---

## 9. Phase 0 checklist

Nothing in Phase 3 onwards should start until all of these hold.

- [ ] DNS resolves `photocat.` and `auth.` to the host
- [ ] TLS certificate installed; HTTP redirects to HTTPS
- [ ] `photocat` system user owns `/srv/photocat`, `data/` is `750`
- [ ] pyKES installed in the venv; `open_index` creates `index.sqlite`
- [ ] `photocat.service` runs and binds **127.0.0.1 only** (`ss -tlnp`)
- [ ] Authelia running; secrets in files, not in the configuration
- [ ] Logging in with a second factor reaches the application
- [ ] Logging out, then requesting the app directly, returns the login page
- [ ] **`Remote-User` visible in `st.context.headers` through the proxy**
- [ ] `curl https://…/payloads/anything.h5` returns 404, not a file
- [ ] Uploading a >200 MB file is accepted (`maxUploadSize`)
- [ ] Backup script runs from cron, and a restore has been tested once

The two that catch people out are the WebSocket headers — without them the page
loads and silently stops working — and the `internal` directive on
`/payloads/`, without which every measurement in the database is downloadable by
anyone who guesses a filename.
