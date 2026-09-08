# Provisioning the database server

Everything needed to get from a bare machine to a TLS-protected,
Authelia-gated host ready to serve the photocatalysis database: the system
user, the directory layout, the authentication route, the certificate and the
firewall.

This document is about the **machine**. The code that runs on it — how it gets
there, what pins the version that is live, and how a bad one is undone — is
[deployment.md](deployment.md). The split is deliberate: the parts here are
painful to change once real data is on the machine, and the parts there are
meant to change every week.

The machine is a DigitalOcean droplet running Ubuntu, with nginx and Docker
already installed and a domain the group owns resolving to it.

The decisions this document implements:

| Decision | Choice |
| --- | --- |
| Authentication | **Authelia**, in front of nginx, with TOTP |
| Name collisions | Both kept, newcomer suffixed `__v2` |
| Editing and deletion | **Owner + admin only** |
| Editing a shared ancestor | Owner + admin, descendants recomputed |
| Entity types | experiment, catalyst_batch, finished_semiconductor, precursor_semiconductor, precursor_chemical, commercial_chemical, stock_solution, modified_catalyst_batch, other_entity |
| Original uploads | **Kept indefinitely** |
| TLS | **Let's Encrypt** via certbot, renewed by its own timer |
| How the code is deployed | a container image built by CI, pinned by digest |

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

The image adds to this. It is about 600 MB unpacked, and keeping one previous
image on disk so a rollback needs no network makes it 1.2 GB; with a build
cache and the pre-update snapshots, budget **10 GB more** than the table above.
On a 25 GB droplet you will meet `no space left on device` inside a year, and
it will happen during an update.

**Before starting, obtain:** a domain that resolves to the droplet — three
names under it, for the application, staging and the login page — and the list
of people who should have accounts. TLS comes from Let's Encrypt via certbot
(§6): the earlier advice to prefer an institutional CA applied to a host that
was not reachable from the public internet, and this one is.

---

## 2. Layout and system user

The application runs as an unprivileged user that owns the data and nothing
else. Never as root, and never as a login user.

The user id is fixed at 10001, and that is not cosmetic: the container runs as
that id and the bind-mounted data directory must be owned by it, or the
application dies on its first write. An image whose user changed per host could
not be a prebuilt artefact.

```bash
sudo groupadd --gid 10001 photocat
sudo useradd --system --uid 10001 --gid 10001 --home /srv/photocat \
             --shell /usr/sbin/nologin photocat

sudo mkdir -p /srv/photocat/{data,staging-data,tmp,staging-tmp,backups,venv}
sudo mkdir -p /srv/photocat/authelia/{config,secrets}
sudo chown -R 10001:10001 /srv/photocat/{data,staging-data,tmp,staging-tmp}
sudo chmod 750 /srv/photocat/data /srv/photocat/staging-data

# Check it now rather than from a traceback later. SQLite needs write
# permission on the *directory*, not just the file, to create
# index.sqlite-wal; a writable file in a read-only directory fails with
# "attempt to write a readonly database", which reads like corruption.
sudo -u '#10001' test -w /srv/photocat/data && echo "writable by the container"
```

| Path | Holds | Backed up |
| --- | --- | --- |
| `/srv/photocat/.env` | **the pinned image tag and digest** | from git, per deployment |
| `/srv/photocat/staging-data` | a copy of the above, for staging | no — refreshed on demand |
| `/srv/photocat/tmp` | uploads being staged before ingestion | no |
| `/srv/photocat/data/index.sqlite` | the searchable index | **yes** |
| `/srv/photocat/data/uploads` | every uploaded file, verbatim | **yes** |
| `/srv/photocat/data/payloads` | per-experiment HDF5, derived | no — rebuildable |
| `/srv/photocat/venv` | operator tooling only | no |
| `/srv/photocat/backups` | nightly and pre-update snapshots | it *is* the backup |

`payloads/` is deliberately outside the backup: it is reconstructed from
`uploads/` by `rebuild_index`, which is the whole reason uploads are kept.

---

## 3. The application

The code, how it gets here and how it is replaced are
[deployment.md](deployment.md). It is a container image built by CI and pinned
by tag and digest, with a second instance following the main branch, and this
section only covers what the machine has to provide for it.

```bash
sudo apt update
sudo apt install -y nginx sqlite3 rsync
```

`sqlite3` and `rsync` are the backup script's tools and belong on the host
whichever way the application runs. There is deliberately no Python
environment here: `pip install pyKES` does **not** install this application —
version 0.2.4 was released before any of the database work existed — and the
image builds from the repository instead.

A virtual environment is still worth having for operator tooling, so
`photocat-rebuild` can be run without the container:

```bash
sudo apt install -y python3-venv python3-dev build-essential
sudo -u photocat python3 -m venv /srv/photocat/venv
sudo -u photocat /srv/photocat/venv/bin/pip install \
    'pyKES @ git+https://github.com/jschneidewind/pyKES@v0.3.0'
```

Verify the data layer can see the index before going further:

```bash
sudo -u photocat env PHOTOCAT_DATA_ROOT=/srv/photocat/data \
  /srv/photocat/venv/bin/python -c "
from pyKES.database.index_schema import IndexPaths, open_index, read_index_schema_version
connection = open_index(IndexPaths(root='/srv/photocat/data'))
print('index schema', read_index_schema_version(connection))"
```

---

## 4. Running it

The compose stack, the operator scripts and the update runbook are in
[deployment.md](deployment.md); the files themselves are committed under
`compose.yaml` and `deploy/`.

Two things about it belong here, because both contradict what a reader of this
document would reasonably assume.

**The application binds `0.0.0.0`, not `127.0.0.1`.** Inside a container
`127.0.0.1` is the container's own loopback and host nginx cannot reach it. The
confinement moves to the published port, which compose binds to
`127.0.0.1:8501` on the host — and that prefix is load-bearing, because Docker
writes its iptables rules ahead of ufw's (§8).

**The upload ceiling and the theme now ship inside the package.**
`.streamlit/config.toml` used to exist only at the repository root, outside
`src/pyKES`, so it was not in the wheel: a wheel or container install rendered
in Streamlit's light default against a stylesheet built for a dark surface, and
silently dropped `maxUploadSize` back to the 200 MB default — which rejects
exactly the batches this database exists for, since at roughly 140 KB per
experiment a large plate dataset exceeds it.

`deploy/systemd/photocat.service` is the documented way to run it *without*
Docker, kept correct because it is the fallback if the group decides containers
are not worth it. Note that the unit this document used to carry could not
start at all: it ran `streamlit run Home.py` from `/srv/photocat/app`, and no
install produces a `Home.py` at that path — in a wheel the entry script is
inside the package. `photocat-app` resolves it at run time, so one service
definition works for a checkout, a wheel and an image alike.

## 5. Authelia

Authelia is a forward-authentication service: nginx asks it about every request
and refuses to proxy anything it does not approve. The application never sees
an unauthenticated request, and neither do the payload files.

It runs as a container in the same compose stack. This document used to have
you fetch a release binary by hand, which gave the group no upgrade path; the
configuration is otherwise unchanged, and is committed at
`deploy/authelia/configuration.yml`.

### 5.1 Secrets

Never in the configuration file — Authelia reads them from files, and the
directory is mounted read-only.

```bash
for secret in jwt session storage; do
  sudo sh -c "openssl rand -hex 48 > /srv/photocat/authelia/secrets/${secret}.secret"
done
sudo chmod 600 /srv/photocat/authelia/secrets/*.secret
```

The config directory has to be *writable*, because Authelia keeps its own
SQLite database and the filesystem notifier there and rewrites `users.yml` on a
password change — and it does not run as 10001, so find out what it does use:

```bash
docker run --rm --entrypoint id ghcr.io/authelia/authelia:4.39.12
sudo chown -R <that-uid>:<that-gid> /srv/photocat/authelia/config
```

### 5.2 What differs from a bare-metal install

Three lines, each with a failure mode that looks like something else.

* `address: 'tcp://0.0.0.0:9091'`. The container's own loopback is unreachable
  from host nginx; the confinement is compose's `127.0.0.1:9091` publish.
* No `log.file_path`, so `docker compose logs authelia` works and rotation is
  the logging driver's problem.
* Staging carries `subject: ["group:admins"]`, because it holds a copy of the
  real data and a wrong belief about which instance you are looking at is the
  mistake that costs work.

### 5.3 Users

Generate each password hash with Authelia itself, never by hand:

```bash
photocat run --rm authelia authelia crypto hash generate argon2 --password 'chosen'
```

`/srv/photocat/authelia/config/users.yml`:

```yaml
users:
  jschneidewind:
    displayname: "Jacob Schneidewind"
    password: "$argon2id$v=19$m=65536,t=3,p=4$..."
    email: jacob@example.de
    groups:
      - admins
      - users
  ae:
    displayname: "…"
    password: "$argon2id$v=19$..."
    email: ae@example.de
    groups:
      - users
```

The `admins` group is what the application reads to decide who may edit
another person's entry, and who may reach staging. Everyone else is in `users`:
they can read everything and edit their own entries. The group name is
`PHOTOCAT_ADMIN_GROUP` in the application, so pointing it at an institutional
directory's own group does not need a release.

**After the first login, check the address in Authelia's log.** The
brute-force lockout bans by client address, and if `X-Forwarded-For` is not
reaching it, that address is `172.17.0.1` — the Docker bridge gateway — so the
lockout hits everybody or nobody.

```bash
photocat logs authelia | grep -i remote_ip
```

## 6. nginx and TLS

nginx and certbot stay on the host while the application and Authelia run in
containers: nginx is already installed, this configuration reaches both over
loopback, `certbot --nginx` plus its packaged timer is a solved problem, and
TLS stays up while the stack is being recreated — so a user sees a maintenance
page rather than a refused connection.

The configuration is committed at `deploy/nginx/photocat.conf`. Install it and
take the certificate:

```bash
sudo cp deploy/nginx/photocat.conf /etc/nginx/sites-available/photocat
sudo cp deploy/nginx/snippets/photocat-maintenance.conf /etc/nginx/snippets/
sudo ln -s /etc/nginx/sites-available/photocat /etc/nginx/sites-enabled/

sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx --redirect -d photocat.example.de \
     -d photocat-test.example.de -d auth.example.de
sudo certbot renew --dry-run

sudo nginx -t && sudo systemctl reload nginx
```

Six things in that file have a failure mode that looks like something else, and
each is commented where it appears:

* **The WebSocket headers.** Streamlit needs `proxy_http_version 1.1`, the
  `Upgrade` and `Connection` headers and a generous `proxy_read_timeout`.
  Without them the page loads and then appears frozen, which reads as an
  application bug and is not one.
* **`internal` on `/payloads/`.** It is what makes the path unreachable except
  by internal redirect. Without it the entire dataset is simply on the web.
  (Not yet used: downloads still go through Streamlit, which is correct and
  authenticated but passes the bytes through Python.)
* **The ACME challenge location comes before the redirect, and outside
  `auth_request`.** Authelia would answer a challenge with a 302 to the login
  page, and renewal would fail silently about sixty days later.
* **`/_stcore/health` is outside the auth gate.** Behind it, every external
  health check gets a 302, so a monitor can never tell "the application is
  down" from "you are not logged in". It returns the literal string `ok`.
* **`listen 443 ssl http2`, not a separate `http2 on;`.** That directive only
  exists from nginx 1.25.1, and Ubuntu 24.04 LTS ships 1.24, where it is an
  unknown directive and nginx refuses to start.
* **No IPv6 listeners.** A droplet has IPv6 only if it was enabled at
  creation; without it nginx will not start. Add them once `ip -6 addr` shows
  an address.

### The identity header, and the one thing to verify early

The application reads the signed-in user from the `Remote-User` header via
`st.context.headers`. **That reflects the `/_stcore/stream` WebSocket request,
not the initial page request**, so the header must be set on the location that
proxies the WebSocket — which is the same `location /`, and is exactly why it
is not split into a separate block.

Verify it before trusting anything else:

```python
# scratch.py, run through the proxy, not directly against port 8501
import streamlit as st
st.write("headers:", dict(st.context.headers))
```

`Remote-User` must be present. If it is not, the application cannot attribute
uploads and the ownership rule cannot be enforced — and it now *refuses to
serve* rather than falling back to an admin identity called `developer`, which
is what it used to do. That fallback is off unless `PHOTOCAT_ALLOW_DEV_LOGIN`
asks for it.

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

Two firewalls, and they are not redundant. The DigitalOcean Cloud Firewall
drops traffic before it reaches the droplet at all, so it survives anything
misconfigured on the host; ufw is what protects against a service that binds
an interface it should not.

```bash
sudo ufw default deny incoming
sudo ufw allow 22/tcp
# Port 80 is not optional: certbot's HTTP-01 challenge needs it, and the
# redirect to HTTPS lives there. The earlier rules in this document opened
# only 443, which made both unreachable.
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

In the DigitalOcean control panel, allow inbound 80 and 443 from anywhere and
22 only from the addresses the group actually uses. The box is on the public
internet now, so also: key-only SSH, no root login, and
`unattended-upgrades` enabled.

**Docker writes its own iptables rules ahead of ufw's.** A published port with
no address prefix is therefore reachable from the internet with
`ufw default deny incoming` still in force and still reporting it. Every
`ports:` entry in `compose.yaml` binds `127.0.0.1` explicitly, and the check
that this is actually true belongs in every verification pass:

```bash
sudo ss -tlnp | grep -E '8501|8502|9091'   # 127.0.0.1 only
curl -sf --max-time 5 http://<droplet-address>:8501/ && echo "EXPOSED"
```

### Backups

`photocat-backup`, committed at `deploy/bin/photocat-backup`, from cron
nightly. `sqlite3 .backup` rather than `cp`, because the index is in WAL mode
and copying `index.sqlite` without its `-wal` and `-shm` gives a file missing
the most recent commits.

It backs up the two irreplaceable things — `index.sqlite` and `uploads/` — and
three that a rebuild cannot reconstruct on its own: the corrections made on the
entry page, which have no journal to replay from and which a rebuild reverts;
the entity type each upload was ingested as, for uploads stored before that was
recorded; and the version pin that was live.

`payloads/` is excluded from the nightly backup, and deliberately: it is
reconstructed from `uploads/`, which is the whole reason uploads are kept. A
**pre-update** snapshot is a different artefact with a different job — it takes
payloads too, as hardlinks against the last snapshot, which costs almost
nothing and is what makes a rollback a file restore rather than a five-minute
rebuild.

```bash
sudo crontab -e
# 15 2 * * *  /usr/local/bin/photocat-backup
```

**Test the restore path once, before you need it.** This is now a test that
means something: restore a snapshot into a scratch directory and rebuild it
there.

```bash
sudo mkdir -p /srv/photocat/rehearsal
sudo cp /srv/photocat/backups/<snapshot>/index.sqlite /srv/photocat/rehearsal/
sudo cp -r /srv/photocat/backups/<snapshot>/uploads /srv/photocat/rehearsal/
sudo -u photocat env PHOTOCAT_DATA_ROOT=/srv/photocat/rehearsal \
  /srv/photocat/venv/bin/photocat-rebuild --yes
```

Confirm the entry counts by kind and the reference count match production. Note
that this instruction has been in this document from the beginning and could
not have worked: `uploads.stored_path` was absolute, so a rebuild against a
restored copy read the *live* upload store and a passing test proved nothing
about the backup. Paths are relative now.

A backup nobody has restored is a hypothesis.

### Routine maintenance

* `unattended-upgrades` for OS security patches.
* `ANALYZE` after any bulk ingestion — `photocat-rebuild` runs it, and the
  Admin page has a button.
* A monthly look at the Admin page for dangling references and drifted
  metadata keys, and at `docker system df`.
* **Never `docker image prune -a`.** It removes untagged *and* unused-tagged
  images, which is exactly the previous image a rollback needs. See
  [deployment.md](deployment.md) §5 for a safe prune.

---

## 9. Provisioning checklist

Nothing that stores real data should start until all of these hold.

- [ ] DNS resolves `photocat.`, `photocat-test.` and `auth.` to the droplet
- [ ] DigitalOcean Cloud Firewall allows 80, 443 and 22 (22 restricted)
- [ ] `ufw` allows 22, **80** and 443; key-only SSH; no root login
- [ ] Let's Encrypt certificate installed; `certbot renew --dry-run` passes;
      `certbot.timer` is active
- [ ] `photocat` owns `/srv/photocat/data` as uid **10001**, mode 750, and
      `sudo -u '#10001' test -w /srv/photocat/data` succeeds
- [ ] `ss -tlnp` shows 8501, 8502 and 9091 on **127.0.0.1 only**
- [ ] the same ports are **not** reachable from off-host (the Docker/ufw check)
- [ ] Authelia running; secrets in files, not in the configuration; its config
      directory writable by the id the image actually uses
- [ ] the address in Authelia's log after a login is the user's, not
      `172.17.0.1`
- [ ] logging in with a second factor reaches the application
- [ ] logging out, then requesting the app directly, returns the login page
- [ ] **`Remote-User` visible in `st.context.headers` through the proxy**, and
      Home names the signed-in user rather than warning about the proxy
- [ ] staging reachable at `photocat-test.` and refused to a non-admin
- [ ] `curl https://…/payloads/anything.h5` returns 404, not a file
- [ ] uploading a file over 200 MB is accepted
- [ ] `load_entity_schemas()` inside the container returns **9**, not 0
- [ ] `photocat-backup` runs from cron, and a restore has been rehearsed once
- [ ] a deploy and a rollback have each been performed once, before real data
      arrives

The two that catch people out are unchanged: the WebSocket headers, without
which the page loads and silently stops working, and `internal` on
`/payloads/`, without which every measurement in the database is downloadable
by anyone who guesses a filename. The third, new with the droplet, is the
Docker-before-ufw rule — a missing `127.0.0.1` prefix puts Streamlit on the
public internet while the firewall still reports it closed.
