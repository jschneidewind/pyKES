# Deploying the database app, and changing it afterwards

[server_provisioning.md](server_provisioning.md) gets a machine ready. This
document is about the code that runs on it: how an edit on a laptop becomes the
running version, what pins the version that is live, how a bad one is undone,
and how none of that touches the only thing on the machine that cannot be
recreated.

The machine is a DigitalOcean droplet running Ubuntu, with nginx and Docker
already installed and a domain that resolves to it. One host runs everything.

## Names to substitute

Every committed file uses placeholders. Replace them once, on the droplet.

| Placeholder | What it is |
| --- | --- |
| `photocat.example.de` | the application |
| `photocat-test.example.de` | staging, admins only |
| `auth.example.de` | Authelia's login page |
| `example.de` | the cookie domain, in the Authelia configuration |
| `ghcr.io/jschneidewind/pykes-photocat` | the image, in `.env` |
| `/srv/photocat` | the deployment root, if it moves |

---

## 1. What is deployed, and what a version of it is

The database app lives inside pyKES: `src/pyKES/database_app/Home.py` and its
`pages/`, discovered by Streamlit from inside the installed package. So the
unit of deployment is a pyKES build, not a separate app repository.

**`pip install pyKES` does not get you this application.** Version 0.2.4 was
released before any of the database work existed, and every database commit
landed after it with the version untouched, so the tree on PyPI and the tree in
this repository were two different programs sharing a number. That is why the
image builds from the repository, and why the version is now 0.3.0.

Four numbers identify a running deployment, and the Home and Admin pages show
all of them:

| Identifier | Where it comes from | Why it matters |
| --- | --- | --- |
| image tag | `PHOTOCAT_TAG` in `.env` | what a person reads in `docker compose ps` |
| image digest | `PHOTOCAT_DIGEST` in `.env` | what actually runs, so a moved tag cannot change it |
| git commit | baked in at build time | which source produced it |
| pyKES version | the installed distribution | **stamped into every payload the server writes** |

The last has scientific consequences: `write_payload` records it in each
derived HDF5, so it is the answer to "which code produced this trace".

And the line that everything else follows from:

> The code is replaceable. `index.sqlite` and `uploads/` are not.

Everything below exists to keep those two out of the path of a code change.

---

## 2. Why an image built in CI

Five ways to get a change onto the droplet, and what each costs.

| | A code change arrives by | Rollback | Reproducible? | Cost |
| --- | --- | --- | --- | --- |
| **A** pinned PyPI release + venv | `uv pip sync requirements/X.Y.Z.txt` | reinstall in place, not atomic | only with a committed hashed closure | every UI tweak is a public release of a *kinetics library*; cannot deploy a branch |
| **B** git checkout + editable venv | `git pull && systemctl restart` | another checkout, another sync | no — unpinned deps re-resolve | the live version is a mutable working tree; no atomic switch |
| **C** release directories + symlink | build a new directory, repoint `current` | **repoint the symlink: seconds, no network** | per release, with a lock | a bespoke deploy script; N virtual environments |
| **D** Docker, built on the droplet | `git pull && compose up -d --build` | rebuild the old commit | environment yes, artefact no | minutes of build on two cores; a second place code lives |
| **E** Docker, built in CI → GHCR | `compose pull && up -d` a pinned digest | **one line in `.env`, under a minute** | yes, digest-addressed | a CI round trip; ~280 MB to pull; prune discipline |

**E, and the reasons are specific to this repository rather than general.**

A rollback restores the whole environment, not just the code. On the venv path
it does not: `pip install pyKES==0.2.4` fetches a wheel that does not contain
this application at all, and leaves every transitive dependency wherever the
failed upgrade put it. Only `streamlit` and `pandas` are bounded, so two
installs a month apart can be different programs.

`.streamlit/config.toml` used to live only at the repository root, outside
`src/pyKES`, so it was not in the wheel — and a wheel-only deployment silently
rendered in Streamlit's light default against a stylesheet built for a dark
surface, with the upload ceiling back at 200 MB, rejecting exactly the batches
this database exists for. It now ships inside the package, which fixes every
deployment shape; but an image built from the repository could never have had
the problem.

Staging and production run the same image, so staging tests what will ship.
And CI can run the tests before publishing, which nothing did before.

The honest costs. A change cannot be tried on the droplet until CI has built it
— about ninety seconds warm, three to four minutes cold. The image is roughly
600 MB unpacked and 280 MB to pull, driven by pyarrow (49 MB, required by
Streamlit), scipy and numpy, none of them removable. Every runbook line
changes: `journalctl -u photocat` becomes `photocat logs -f app`, and when a
container will not *start*, `exec` is unavailable and you need
`photocat run --rm --entrypoint sh app` — a different command for the case
where you least want to look one up. And the container user's id must match
the owner of the bind-mounted data directory, which is the trap that bites
first (§4).

### Named and rejected

**DigitalOcean App Platform**, and managed hosting generally: architecturally
impossible, not merely unwanted. The filesystem is ephemeral, so `index.sqlite`,
`uploads/` and `payloads/` cannot live there; you would need object storage and
a managed database, which is a rewrite of the data layer. Forward authentication
is not expressible either, and it builds a container anyway — Docker with less
control.

**Kubernetes or k3s**: [photocatalytic_database.md](photocatalytic_database.md)
§10 already draws this line. One host, one process, one SQLite file is the
opposite of the problem an orchestrator solves.

**Ansible**: the right tool for recording *provisioning* — nginx, Authelia,
certbot, ufw, unattended upgrades — and worth considering if the droplet is
ever rebuilt. For deploying one host it puts a language between the operator
and the machine.

**Nix**: the strongest reproducibility on this list, and it would eliminate the
Python-version failure mode outright. Rejected on who is on call: a chemist,
and `buildPythonPackage` with h5py and scipy overrides is a week of learning
before the first deploy.

**Auto-deploy on push** — a webhook, watchtower, `git pull` from cron: fine for
staging and wrong for production. There is no point at which a person decided
this version is good, and a broken commit restarts the app while somebody is
uploading. Staging does exactly this on a timer, deliberately.

**rsync from a laptop**: nobody can say what is running, and `git log` becomes
fiction. It is what people do when releasing is painful, so the fix is to make
releasing painless.

---

## 3. The shape of the deployment

| Layer | Runs where | Replaced by an update? |
| --- | --- | --- |
| nginx, certbot, ufw | **host** | no — configuration, edited rarely |
| Authelia | container, config bind-mounted | on its own release cadence |
| the app, production | container, pinned tag and digest | **yes; this is what an update replaces** |
| the app, staging | container, follows `main` | continuously, on a timer |
| `index.sqlite`, `uploads/`, `payloads/` | host filesystem, bind-mounted | **never** |

**nginx and certbot stay on the host.** nginx is already installed, the
configuration reaches both containers over loopback, `certbot --nginx` plus its
packaged timer is a solved problem, and TLS stays up while the stack is
recreated — so a user sees a maintenance page rather than a refused connection.
Moving nginx into compose buys "one file" and costs a certificate volume, a
cross-boundary reload signal, and publishing 80 and 443 from a container.

**Authelia moves into a container**, because the provisioning guide installs a
release binary by hand and offers no upgrade path.

### Two settings that invert the provisioning guide

Both are counterintuitive and both are load-bearing.

**The app binds `0.0.0.0` inside the container.** §4 of the provisioning guide
says `127.0.0.1`, and inside a container that is the container's own loopback,
unreachable from host nginx. The confinement moves to the published port.

**The published port must be `127.0.0.1:8501:8501`.** Docker writes its own
iptables rules ahead of ufw's, so a bare `8501:8501` publishes Streamlit to the
entire internet with `ufw default deny incoming` still in force. On a droplet
with a public address this is the single most dangerous line in the
configuration.

### The data root has the same path inside and out

`uploads.stored_path` is stored relative to the upload store, so the archive is
relocatable. The container still mounts the data at `/srv/photocat/data`, the
path it has on the host, and that is worth the small oddity: a rebuild run
inside the container and one run from a host virtual environment agree about
where everything is, and staging can mount `/srv/photocat/staging-data` at that
same container path — so a copy of the production index serves from it without
anything in it being rewritten.

---

## 4. First installation

Assumes the droplet, the domain and Docker; §§1–2 and 5–8 of
[server_provisioning.md](server_provisioning.md) cover the rest.

```bash
# The container runs as 10001. The bind-mounted data directory must be owned
# by that id, or the app dies on its first write.
sudo groupadd --gid 10001 photocat
sudo useradd --system --uid 10001 --gid 10001 --home /srv/photocat \
             --shell /usr/sbin/nologin photocat

sudo mkdir -p /srv/photocat/{data,staging-data,tmp,staging-tmp,backups,bin}
sudo mkdir -p /srv/photocat/authelia/{config,secrets}
sudo chown -R 10001:10001 /srv/photocat/{data,staging-data,tmp,staging-tmp}
sudo chmod 750 /srv/photocat/data /srv/photocat/staging-data

# Verify it before starting anything. SQLite needs write permission on the
# directory, not just the file, to create index.sqlite-wal — a writable file
# in a read-only directory fails with "attempt to write a readonly database",
# which reads like corruption and is not.
sudo -u '#10001' test -w /srv/photocat/data && echo "writable by the container"
```

Then the files from this repository:

```bash
sudo cp compose.yaml /srv/photocat/
sudo cp .env.example /srv/photocat/.env        # then edit the pins
sudo chmod 600 /srv/photocat/.env
sudo cp deploy/bin/* /usr/local/bin/
sudo cp deploy/authelia/configuration.yml /srv/photocat/authelia/config/
sudo cp deploy/nginx/photocat.conf /etc/nginx/sites-available/photocat
sudo cp deploy/nginx/snippets/photocat-maintenance.conf /etc/nginx/snippets/
sudo ln -s /etc/nginx/sites-available/photocat /etc/nginx/sites-enabled/
sudo cp deploy/systemd/photocat-staging-update.* /etc/systemd/system/
```

Authelia's secrets, never in its configuration file:

```bash
for secret in jwt session storage; do
  sudo sh -c "openssl rand -hex 48 > /srv/photocat/authelia/secrets/${secret}.secret"
done
sudo chmod 600 /srv/photocat/authelia/secrets/*.secret

# The Authelia image does not run as 10001; chown /config to whatever it does
# use, or it cannot write its own database.
docker run --rm --entrypoint id ghcr.io/authelia/authelia:4.39.12
```

Certificates, replacing the guide's institutional-CA hedge:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx --redirect -d photocat.example.de \
     -d photocat-test.example.de -d auth.example.de
sudo certbot renew --dry-run
sudo systemctl status certbot.timer
```

The first index has to be asked for, because an absent data root is otherwise
indistinguishable from a mount that failed to attach:

```bash
# Explicit, and only ever run once. Creating an index is otherwise refused,
# because a data root that is not there is indistinguishable from a bind mount
# that failed to attach — and an application that quietly starts a second,
# empty archive is worse than one that will not start.
photocat run --rm -e PHOTOCAT_CREATE_INDEX=1 --entrypoint python app -c \
  "from pyKES.database.index_schema import IndexPaths, open_index; \
   open_index(IndexPaths(root='/srv/photocat/data')).close(); \
   print('index created')"

photocat up -d
sudo systemctl enable --now photocat-staging-update.timer
```

---

## 5. Changing the code

Three layers, and almost every change stops at the first.

### Layer 0 — the laptop (seconds)

No droplet involved. This is where nearly every change is validated, and no
deployment mechanism should be optimised at its expense.

```bash
python -m pyKES.database_app.seed_demo --root /tmp/photocat-demo --fresh
PHOTOCAT_DATA_ROOT=/tmp/photocat-demo PHOTOCAT_ALLOW_DEV_LOGIN=1 photocat-app
```

Sixty experiments, twelve batches, six semiconductors and three precursors,
linked four levels deep — enough to exercise every filter, the mapping
sub-key pickers and the multi-path merge. Streamlit reruns on save.

For the container version of the same loop, `compose.dev.yaml` bind-mounts the
checkout. It needs one non-obvious trick: `PYTHONPATH=/workspace/src`, so the
checkout precedes site-packages. Streamlit's file watcher deliberately ignores
anything under `site-packages`, so editing an installed package reloads
nothing however it is mounted. And `file_watcher_type: poll` rather than
`watchdog`, because inotify crosses a Linux bind mount but not Docker
Desktop's VirtioFS, where edits produce no events at all.

What reload does not cover: the thread-local SQLite connections and
module-level constants survive a rerun, so a data-layer change still needs a
restart — five to ten seconds, mostly the cold import of pandas and scipy.

### Layer 1 — staging (minutes, automatic)

Merge to `main`. CI runs the tests, builds, and pushes `:main`. A systemd timer
on the droplet pulls it within twenty minutes; `photocat pull app-staging &&
photocat up -d app-staging` does it now.

Staging answers at `photocat-test.example.de`, gated to the `admins` group, and
runs against a copy of the production data:

```bash
photocat-refresh-staging
```

This is what makes staging worth having: a migration or a rebuild rehearsed
against the group's real entries rather than against sixty synthetic ones. It
needs no path rewriting, because both containers see the data root at the same
place.

### Layer 2 — production (a deliberate act)

Tag a release. `v*` triggers two workflows: the PyPI release, and the image.

```bash
$EDITOR pyproject.toml CHANGELOG.md     # bump the version
git tag v0.3.1 && git push origin v0.3.1
```

The image run's summary prints the two lines to paste. Then, on the droplet:

```bash
sudoedit /srv/photocat/.env             # PHOTOCAT_TAG and PHOTOCAT_DIGEST
photocat-deploy
```

`photocat-deploy` does, in this order:

1. **refuses** if the rendered configuration mounts `/workspace`, which would
   mean a development override is somehow in effect;
2. reports recent ingestion activity and the newest upload rows, so you can
   see whether anything is mid-flight;
3. **drains** — `photocat-maintenance on` writes `return 503` into an included
   nginx snippet and reloads, so no new session starts while sessions already
   open finish — then waits a minute;
4. **snapshots**: `sqlite3 .backup` of the index, `uploads/`, and `payloads/`
   as hardlinks against the last snapshot. The nightly backup rightly excludes
   the derived payloads; a pre-update snapshot takes them, because that is what
   makes a rollback a file restore instead of a five-minute rebuild. It also
   exports the corrections a rebuild would revert;
5. records the live digest and copies `.env` to `.env.previous`;
6. **pulls before swapping**, so the container is down for the recreate rather
   than for the download as well;
7. `up -d`, then polls `/_stcore/health` and prints the version that came up.
   An unhealthy container aborts with maintenance still on.

Rolling back:

```bash
photocat-rollback                # the image only, under a minute
photocat-rollback --with-data    # and the pre-update snapshot
```

Only pass `--with-data` if the update ran a migration or a rebuild. Restoring
the data unnecessarily discards whatever was uploaded since the snapshot.

**One case where `--with-data` is not optional**: a release that changes
`INDEX_SCHEMA_VERSION`. `photocat-app` stamps the index with the version it
writes at startup, and `SUPPORTED_SCHEMA_VERSIONS` is an exact-match tuple, so
once the new image has opened the index the *previous* image can no longer
read it. A code-only rollback then fails at the health poll with maintenance
still on, which is a discovery to make on staging rather than during a
rollback. Whenever the release notes mention a schema version, roll back with
the data.

The image half is fast because the previous image is still on disk — which is
why the prune policy must **never** be `docker image prune -a`. That is exactly
the command that deletes your rollback target:

```bash
# /etc/cron.weekly/photocat-docker-prune
docker image   prune --force --filter "until=720h"
docker builder prune --force --keep-storage 3GB
docker container prune --force --filter "until=168h"
```

---

## 6. Schema changes

One rule, worth memorising:

> If the change alters what gets **written** into `entities.metadata`,
> `results`, `contributions`, `edges`, or a payload file — rebuild. If it only
> alters what gets **read or displayed** — restart.

Three tiers follow from it.

**Tier 1, automatic.** A new metadata key, a new spreadsheet column, a new
result label: nothing to do, because metadata lives in JSON. A new typed column
listed in `ADDED_COLUMNS`: `add_missing_columns` applies it, and `photocat-app`
now does that once in the foreground at startup, so it appears in the log and
fails the service rather than happening inside whichever request opens the
first connection.

**Tier 2, additive plus a registry rebuild.** The mapping feature was one: the
`sub_keys` column exists but is NULL for every key that predates it, so the
dopant picker stays empty until somebody presses **Rebuild Key Registry** on
the Admin page. Restart, then press it.

**Tier 3, a full rebuild.** A non-additive layout change, a change to how keys
are escaped or metadata prepared, a change to result extraction, a renamed
reference role, a new payload layout.

Three tier-3 triggers are invisible from the file that causes them, and all
three live in the schema YAMLs:

* editing a `derived:` block — derived scalars are computed by
  `prepare_metadata`, which runs **only at ingestion**;
* adding a `type: reference` field — the resulting `edges` rows are stored, so
  existing entries never grow the new edge;
* but editing `accepts:` or a `select` option is **tier 1**, because those are
  re-evaluated on every read.

The schema-version check refuses an index it cannot read and names the repair.
Run it with the application stopped:

```bash
photocat-maintenance on
photocat-backup pre-rebuild
photocat run --rm --entrypoint photocat-rebuild app --dry-run   # what it will
                                                                # re-read, as what
photocat stop app
photocat run --rm --entrypoint photocat-rebuild app --yes
photocat up -d app
photocat-maintenance off
```

Both `photocat-rebuild` invocations go through the container. The operator
scripts (`photocat`, `photocat-maintenance`, `photocat-backup`) live on the
host, but the package does not: nothing outside the image installs it, so a
bare `photocat-rebuild` is not a command on this droplet. `PHOTOCAT_DATA_ROOT`
is set in the compose environment rather than in your shell, too, so even
where the command exists it would look for `~/.photocat` and stop. If you
built the optional operator virtual environment from
[server_provisioning.md](server_provisioning.md) §3, its form is
`sudo -u photocat env PHOTOCAT_DATA_ROOT=/srv/photocat/data
/srv/photocat/venv/bin/photocat-rebuild --dry-run` — the data root spelled
out, every time.

The dry run is safe to run against an index the version check refuses, and it
is safe to stop after it. The version stamp is what the check reads, and only
a *committed* rebuild writes it — so a dry run, a `--backfill-paths`, or a
rebuild that fails part-way all leave the refusal in place, and the
application still will not start until the real rebuild has run. That is the
intended order: find out what the rebuild will do while the app is still
serving, then stop it and do it.

Rehearse it on staging first, against refreshed data, and compare the entry
counts by kind and the reference count before and after — those two numbers are
what a bad rebuild shows up in.

**What a faithful rebuild still rewrites.** Everything the database is asked
about comes back identical — every entry and its metadata, the reference
graph, every inherited value, the upload log and the key registry. Two clocks
do not: `entities.created_at`/`updated_at` are taken from the upload's own
timestamp rather than from the moment each row was first written, so all the
entries of one batch share one time instead of keeping the second they were
each inserted at, and the registry's `first_seen`/`last_seen` move to the
rebuild. Nothing computes on any of them — they order the admin log and the
entry page — but do not expect a `created_at` comparison to be a round trip.

**What a rebuild still cannot restore:** corrections made on the entry page.
`update_entity_metadata` writes to the entity row and there is no journal to
replay, so a rebuild reverts them. `photocat-rebuild` exports them first, and
`photocat-backup` writes them to `corrections.json`; they have to be
re-applied by hand. A corrections journal is a design question, not a fix, and
is not built.

---

## 7. Restarting, honestly

`photocat up -d app` destroys and recreates the container. Streamlit gets
SIGTERM and closes every `/_stcore/stream` WebSocket at once.

**Survives**: the search itself, because the filters and the open entry live in
the query string — an accidental benefit of "a search is a URL", and a real
one. And the Authelia session, which is a separate container: nobody has to log
in again.

**Lost**: every `st.session_state` — a half-filled contribution form, a chosen
comparison subset, a staged upload — and every `st.cache_data` entry, so the
first Browse load afterwards is slower.

**The data is safe.** `ingest_hdf5_upload` commits once, at the end, so SQLite
rolls back an interrupted ingestion. What is left behind is a verbatim copy in
`uploads/` with no row referring to it: harmless, ignored by a rebuild,
overwritten if the same file is uploaded again, and listed by
`photocat-rebuild` so the disk cost is visible.

### Why not blue/green

The mechanism looks like it should work — two containers, an nginx upstream, a
reload — and it does not, for two reasons that are not worth fixing here.

Streamlit's out-of-band endpoints are session-scoped:
`st.file_uploader` POSTs to `/_stcore/upload_file/<session_id>/<file_id>` on a
fresh HTTP connection, which would land on the new upstream, which has never
heard of that session. In-flight uploads and every download button in an old
session would fail — the two operations that matter most on Contribute and
Entry. Fixing it needs session-id-aware routing.

And both containers would write the same `index.sqlite`. WAL makes that safe
mechanically, but during a tier-3 deploy — exactly the deploy you would most
want to be graceful — two incompatible versions of the ingestion code would be
writing at once. Separate data roots would make it not a handover.

**The policy, proportionate to twenty people on one host:** an announced
window, the maintenance toggle as the drain, and a published expectation — you
will be disconnected and the page will reload, anything typed and not
submitted is gone, you will not have to log in again, and nothing in the
database is at risk. Urgent security patches skip the window.

---

## 8. Verifying an update

```bash
photocat ps                                        # healthy
photocat logs -n 20 app | grep '^photocat:'        # the startup line
curl -fsS http://127.0.0.1:8501/_stcore/health     # ok
sudo ss -tlnp | grep 8501                          # 127.0.0.1 only
curl -s -o /dev/null -w '%{http_code}\n' https://photocat.example.de/          # 302
curl -s -o /dev/null -w '%{http_code}\n' https://photocat.example.de/payloads/x.h5  # 404

# From off-host: this must FAIL. If it answers, a ports entry is missing its
# 127.0.0.1 prefix and Streamlit is on the public internet.
curl -sf --max-time 5 http://<droplet-address>:8501/ && echo "EXPOSED"

# Nine, not zero. Zero is what silent schema loss looks like, and it does not
# raise: it removes validation, the templates and every reference declaration.
photocat exec -T app python -c \
  "from pyKES.database.entity_schema import load_entity_schemas as l; print(len(l()))"
```

Then in a browser, because three things no `curl` reaches:

- Home names the version you deployed, and says who you are signed in as. If
  it says *running without the authenticating proxy*, `Remote-User` is not
  arriving and uploads would be misattributed;
- Browse returns rows with one metadata filter applied — this exercises
  `st.context.headers` on the WebSocket, facet generation, and the JSON-path
  query at once;
- a dopant filter offers names and a slider, which is what proves mapping
  fields were parsed rather than stored as text.

---

## 9. Related documents

* [server_provisioning.md](server_provisioning.md) — the machine: user, layout,
  Authelia, nginx, firewall, backups.
* [database_app.md](database_app.md) — the application and its pages.
* [database_index.md](database_index.md) — the data layer and the rebuild.
* [releasing.md](releasing.md) — the PyPI release a `v*` tag also triggers.
