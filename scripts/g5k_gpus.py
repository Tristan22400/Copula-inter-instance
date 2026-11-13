#!/usr/bin/env python3
"""Find free Grid5000 GPUs across sites for your group, and submit where the wait is shortest.

Runs on any Grid5000 frontend (stdlib only, system python3). From a laptop whose
checkout is an sshfs mount of the cluster storage it re-runs itself over ssh on
that frontend (or on G5K_FRONTEND). See docs/workflows.md, "Other Grid5000 sites".

    python3 scripts/g5k_gpus.py status                        # free GPUs per cluster you can use
    python3 scripts/g5k_gpus.py status --gpus 2 --walltime 24 --besteffort
    python3 scripts/g5k_gpus.py submit --walltime 40 -- scripts/train.sh model=copula_nano
    python3 scripts/g5k_gpus.py submit --dry-run --walltime 12 -- scripts/eval_checkpoint.sh ckpt=nano
    python3 scripts/g5k_gpus.py jobs                          # your jobs on every site
    python3 scripts/g5k_gpus.py setup-site nancy rennes sophia [--copy-secrets]

Data sources, all read-only except `submit`/`setup-site`:
  - https://api.grid5000.fr/stable/accesses: priority label (p1..p4/besteffort) of
    each group on each nodeset; your level is the max over your unix groups.
  - <site>/internal/oarapi/resources/details: one row per core, tagged with its GPU id.
  - `oarstat -J -f` on each site frontend (over ssh): which resources each running
    job holds, whether it is besteffort, and when it ends.
  - <site>/status?waiting=yes: OAR's Gantt placement of waiting jobs per node,
    i.e. how long a currently free GPU stays free.

A job cannot span sites, and each site has its own OAR server, home and job ids.
The Grenoble group storage (/srv/storage/...) is mounted on every site, so the
same checkout and venv serve all sites once `setup-site` has replicated the few
home-directory paths the venv and jobs depend on.
"""

from __future__ import annotations

import argparse
import grp
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API = "https://api.grid5000.fr/stable"
ABACA_SITES = ["grenoble", "nancy", "rennes", "sophia"]
DEFAULT_QUEUE_SITES = ["lille", "lyon", "nantes", "luxembourg"]
# Production-queue walltime cap per priority label (Grid5000 "Production" page).
LEVEL_CAP_H = {"p1": 168.0, "p2": 96.0, "p3": 48.0, "p4": 24.0}
# Default queue: weekday jobs may not exceed 14 h (Grid5000 usage policy).
DEFAULT_QUEUE_CAP_H = 14.0
QUEUE_LEVEL = {"admin": 100, "p1": 40, "p2": 30, "p3": 20, "p4": 10, "default": 25, "besteffort": 0}
OCCUPYING_STATES = {"Running", "Launching", "toLaunch", "Finishing", "Resuming", "Suspended", "toAckReservation"}
# Same GPU models as FORBIDDEN_GPU_REGEX in scripts/_env.sh (known not to work for this project).
DEFAULT_EXCLUDE = r"TITAN\s*RTX|TitanRTX|Quadro\s*RTX\s*8000|(^|[^0-9A-Za-z])L4($|[^0-9A-Za-z])"
FORWARD_ENV = ("HF_HOME", "UV_CACHE_DIR")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
INF = float("inf")


# ----------------------------------------------------------------------------- helpers


def on_g5k_frontend() -> bool:
    return Path("/usr/bin/oarsub").exists() and socket.getfqdn().endswith("grid5000.fr")


def local_site() -> str:
    host = socket.gethostname()  # e.g. fgrenoble
    return host[1:] if host.startswith("f") else host


def api_get(path: str) -> Any:
    with urllib.request.urlopen(API + path, timeout=90) as r:
        return json.load(r)


def run_on_site(
    site: str, cmd: list[str], cwd: str | None = None, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    """Run cmd on the site frontend: locally on this frontend's site, else over ssh."""
    if site == local_site():
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    remote = shlex.join(cmd) if cwd is None else f"cd {shlex.quote(cwd)} && {shlex.join(cmd)}"
    return subprocess.run([*SSH, site, remote], capture_output=True, text=True, timeout=timeout)


def parse_walltime(text: str) -> float:
    """Hours from '48', '1.5', '36h', '90m', '12:30' or '12:30:00'."""
    t = text.strip().lower()
    if t.endswith("h"):
        return float(t[:-1])
    if t.endswith("m"):
        return float(t[:-1]) / 60
    if ":" in t:
        parts = [float(p) for p in t.split(":")] + [0.0, 0.0]
        return parts[0] + parts[1] / 60 + parts[2] / 3600
    return float(t)


def fmt_walltime(hours: float) -> str:
    s = int(round(hours * 3600))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_h(seconds: float) -> str:
    if seconds == INF:
        return "-"
    if seconds <= 0:
        return "now"
    h = seconds / 3600
    return f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d"


def my_groups() -> set[str]:
    names = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    return names


# ----------------------------------------------------------------------------- data model


@dataclass
class Gpu:
    site: str
    gid: int
    host: str
    cluster: str
    nodeset: str | None
    model: str
    mem_gb: float
    cc: float
    max_wt_h: float  # node limit; inf if none
    exotic: bool
    production: bool
    state: str  # Alive / standby / Absent / Suspected / Dead / maintenance
    jobs: set[int] = field(default_factory=set)
    blocked_at: float = INF  # start of an advance reservation already holding this GPU


@dataclass
class SiteData:
    site: str
    gpus: dict[int, Gpu]
    jobs: dict[int, dict[str, Any]]
    node_waiting: dict[str, list[tuple[float, str, int]]]  # host -> [(scheduled_at, queue, job uid)]
    error: str = ""


def fetch_site(site: str) -> SiteData:
    now = time.time()
    try:
        res = api_get(f"/sites/{site}/internal/oarapi/resources/details.json?limit=200000")["items"]
    except Exception as exc:  # report the site as unreachable, keep the others
        return SiteData(site, {}, {}, {}, error=f"resources: {exc}")
    gpus: dict[int, Gpu] = {}
    res_gpu: dict[int, int] = {}
    for r in res:
        if r.get("type") != "default" or not r.get("gpu") or not r.get("gpu_model"):
            continue
        gid = int(r["gpu"])
        res_gpu[int(r["id"])] = gid
        if gid in gpus:
            continue
        state = r.get("state", "?")
        if state == "Absent" and (r.get("available_upto") or 0) > now:
            state = "standby"  # energy-saving node OAR wakes up on demand
        if r.get("maintenance") == "YES" or r.get("drain") == "YES":
            state = "maintenance"
        mwt = r.get("max_walltime") or 0
        gpus[gid] = Gpu(
            site=site,
            gid=gid,
            host=r["host"],
            cluster=r["cluster"],
            nodeset=r.get("nodeset"),
            model=r["gpu_model"],
            mem_gb=(r.get("gpu_mem") or 0) / 1024,
            cc=float(r.get("gpu_compute_capability") or 0),
            max_wt_h=mwt / 3600 if mwt > 0 else INF,
            exotic=r.get("exotic") == "YES",
            production=r.get("production") == "YES",
            state=state,
        )
    try:
        out = run_on_site(site, ["oarstat", "-J", "-f"])
        jobs = {int(k): v for k, v in json.loads(out.stdout or "{}").items()}
    except Exception as exc:
        return SiteData(site, gpus, {}, {}, error=f"oarstat: {exc}")
    for jid, j in jobs.items():
        occupying = j.get("state") in OCCUPYING_STATES
        reservation = j.get("state") == "Waiting" and j.get("reservation") == "Scheduled"
        for rid in j.get("assigned_resources") or []:
            held = res_gpu.get(int(rid))
            if held is None:
                continue
            if occupying:
                gpus[held].jobs.add(jid)
            elif reservation:
                start = j.get("scheduled_start") or j.get("start_time") or now
                gpus[held].blocked_at = min(gpus[held].blocked_at, float(start))
    node_waiting: dict[str, list[tuple[float, str, int]]] = defaultdict(list)
    try:
        st = api_get(f"/sites/{site}/status?job_details=yes&waiting=yes&disks=no")
        for host, node in st.get("nodes", {}).items():
            for rv in node.get("reservations") or []:
                if rv.get("state") == "waiting" and rv.get("scheduled_at"):
                    node_waiting[host].append((float(rv["scheduled_at"]), rv.get("queue", "?"), int(rv["uid"])))
    except Exception as exc:
        return SiteData(site, gpus, jobs, {}, error=f"status: {exc}")
    return SiteData(site, gpus, jobs, dict(node_waiting))


def load_access() -> dict[str, tuple[str, int, str]]:
    """nodeset -> (label, level, group) at your best level over your groups."""
    acc = api_get("/accesses")
    best: dict[str, tuple[str, int, str]] = {}
    for g in sorted(my_groups() & set(acc)):
        for nodeset, a in acc[g].items():
            if nodeset not in best or a["level"] > best[nodeset][1]:
                best[nodeset] = (a["label"], a["level"], g)
    return best


# ----------------------------------------------------------------------------- per-cluster summary


@dataclass
class Request:
    gpus: int
    walltime_h: float
    besteffort: bool
    default_queue: bool
    exclude: str
    min_vram: float
    min_cc: float


@dataclass
class Candidate:
    site: str
    target: str  # nodeset (production) or cluster (default queue)
    prop_key: str  # 'nodeset' or 'cluster'
    label: str  # queue: p1..p4, besteffort, default
    level: int
    group: str
    models: list[str]
    mem_gb: float
    cc: float
    exotic: bool
    cap_h: float
    total: int = 0
    usable: int = 0
    free: int = 0
    be: int = 0  # held only by besteffort jobs (preemptable by p1..p4)
    ready_hosts: int = 0  # hosts with >= gpus GPUs available for the whole walltime
    now_hosts: int = 0  # hosts with >= gpus GPUs available right now (may be reclaimed sooner)
    release_s: float = INF  # estimated wait until some host has enough GPUs (ignores queued jobs)
    ahead: int = 0  # waiting jobs placed on this target at >= your level
    maint_s: float = INF  # time to the next maintenance placeholder on this target
    allowed_models: list[str] = field(default_factory=list)  # set when only some of the target's models qualify
    walltime_h: float = 0.0  # requested walltime

    @property
    def fits(self) -> bool:
        return self.cap_h + 1e-9 >= self.walltime_h

    def rank_key(self, prefer_site: str) -> tuple[Any, ...]:
        return (
            not self.fits,
            self.ready_hosts == 0,
            self.now_hosts == 0,
            self.release_s if self.ready_hosts == 0 else 0.0,
            self.label == "besteffort",
            -self.cc,
            -self.mem_gb,
            self.site != prefer_site,
        )


def summarize(sites: list[SiteData], access: dict[str, tuple[str, int, str]], req: Request) -> list[Candidate]:
    now = time.time()
    excl = re.compile(req.exclude) if req.exclude else None
    cands: list[Candidate] = []
    for sd in sites:
        groups: dict[str, list[Gpu]] = defaultdict(list)
        for g in sd.gpus.values():
            if g.nodeset:
                if g.nodeset not in access:
                    continue
                groups[g.nodeset].append(g)
            elif not g.production and req.default_queue:
                groups[g.cluster].append(g)
        for target, all_gpus in groups.items():
            gpus = [
                g
                for g in all_gpus
                if not (excl and excl.search(g.model)) and g.mem_gb + 0.5 >= req.min_vram and g.cc + 1e-9 >= req.min_cc
            ]
            if not gpus:
                continue
            if target in access and gpus[0].nodeset:
                label, level, group = access[target]
                prop_key = "nodeset"
            else:
                label, level, group, prop_key = "default", QUEUE_LEVEL["default"], "", "cluster"
            if label == "besteffort" and not req.besteffort:
                continue
            node_cap = min(g.max_wt_h for g in gpus)
            cap = min(LEVEL_CAP_H.get(label, INF), node_cap)
            if label == "default":
                cap = min(cap, DEFAULT_QUEUE_CAP_H)
            c = Candidate(
                site=sd.site,
                target=target,
                prop_key=prop_key,
                label=label,
                level=level,
                group=group,
                models=sorted({g.model for g in gpus}),
                mem_gb=min(g.mem_gb for g in gpus),
                cc=min(g.cc for g in gpus),
                exotic=any(g.exotic for g in gpus),
                cap_h=cap,
                total=len(gpus),
                walltime_h=req.walltime_h,
            )
            c.allowed_models = c.models if len(c.models) < len({g.model for g in all_gpus}) else []
            hosts: dict[str, list[Gpu]] = defaultdict(list)
            for g in gpus:
                hosts[g.host].append(g)
            seen_ahead: set[int] = set()
            for host, hg in hosts.items():
                waits = sd.node_waiting.get(host, [])
                for at, q, uid in waits:
                    if q == "admin":
                        c.maint_s = min(c.maint_s, at - now)
                    elif QUEUE_LEVEL.get(q, 0) >= level and uid not in seen_ahead:
                        seen_ahead.add(uid)
                host_window = min((at for at, q, _ in waits), default=INF) - now
                avail_now = 0
                avail_full = 0
                releases: list[float] = []
                for g in hg:
                    if g.state not in ("Alive", "standby"):
                        continue
                    c.usable += 1
                    js = [sd.jobs.get(j, {}) for j in g.jobs]
                    only_be = bool(js) and all("besteffort" in (j.get("types") or []) for j in js)
                    if not g.jobs:
                        c.free += 1
                    elif only_be:
                        c.be += 1
                    takeable = not g.jobs or (only_be and label != "besteffort")
                    window = min(host_window, g.blocked_at - now)
                    if takeable:
                        avail_now += 1
                        releases.append(0.0)
                        if window >= req.walltime_h * 3600:
                            avail_full += 1
                    else:
                        ends = [
                            float(j.get("start_time") or now) + float(j.get("walltime") or 0) - now
                            for j in js
                            if "besteffort" not in (j.get("types") or []) or label == "besteffort"
                        ]
                        releases.append(max([0.0, *ends]))
                c.ready_hosts += avail_full >= req.gpus
                c.now_hosts += avail_now >= req.gpus
                if len(releases) >= req.gpus:
                    c.release_s = min(c.release_s, sorted(releases)[req.gpus - 1])
            c.ahead = len(seen_ahead)
            cands.append(c)
    return cands


def collect(args: argparse.Namespace) -> tuple[list[SiteData], list[Candidate], Request]:
    sites = args.sites or (ABACA_SITES + (DEFAULT_QUEUE_SITES if args.default_queue else []))
    req = Request(
        gpus=args.gpus,
        walltime_h=parse_walltime(args.walltime),
        besteffort=args.besteffort,
        default_queue=args.default_queue,
        exclude=args.exclude,
        min_vram=args.min_vram,
        min_cc=args.min_cc,
    )
    with ThreadPoolExecutor(len(sites) + 1) as ex:
        acc_f = ex.submit(load_access)
        data = list(ex.map(fetch_site, sites))
        access = acc_f.result()
    for sd in data:
        if sd.error:
            print(f"[warn] {sd.site}: {sd.error}", file=sys.stderr)
    cands = summarize(data, access, req)
    cands.sort(key=lambda c: c.rank_key(args.prefer_site))
    return data, cands, req


# ----------------------------------------------------------------------------- commands


def cmd_status(args: argparse.Namespace) -> int:
    _, cands, req = collect(args)
    if args.json:
        print(json.dumps([c.__dict__ for c in cands], default=str, indent=1))
        return 0
    print(
        f"request: host=1/gpu={req.gpus}, walltime={fmt_walltime(req.walltime_h)}, "
        f"exclude=/{req.exclude}/ min_vram={req.min_vram:g}GB min_cc={req.min_cc:g}"
        f"{', +besteffort' if req.besteffort else ''}{', +default queue' if req.default_queue else ''}"
    )
    hdr = (
        f"{'site':9s} {'target':12s} {'queue':10s} {'gpu model':28s} {'GB':>4s} {'cap':>5s} "
        f"{'tot':>4s} {'free':>4s} {'be':>3s} {'ready':>5s} {'now':>3s} {'wait~':>6s} {'ahead':>5s} {'maint':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    shown = cands if args.all else [c for c in cands if c.fits]
    for c in shown:
        flag = "" if c.fits else "  (walltime > cap)"
        print(
            f"{c.site:9s} {c.target:12s} {c.label:10s} {'/'.join(c.models)[:28]:28s} {c.mem_gb:4.0f} "
            f"{f'{c.cap_h:.0f}h' if c.cap_h != INF else 'none':>5s} {c.total:4d} {c.free:4d} {c.be:3d} {c.ready_hosts:5d} "
            f"{c.now_hosts:3d} {fmt_h(0 if c.ready_hosts else c.release_s):>6s} {c.ahead:5d} {fmt_h(c.maint_s):>6s}{flag}"
        )
    hidden = len(cands) - len(shown)
    if hidden:
        print(f"({hidden} targets hidden: their walltime cap is below the request; --all shows them)")
    print(
        "\ncolumns: free = idle GPUs; be = GPUs held only by besteffort jobs (a p1..p4 job preempts them);\n"
        "ready = hosts with >= gpus GPUs takeable now and not promised to a queued job within the walltime;\n"
        "now = hosts with >= gpus takeable now; wait~ = when running jobs' walltimes free enough GPUs\n"
        "(upper bound on running jobs, ignores queued ones); ahead = queued jobs placed there at >= your level;\n"
        "maint = next maintenance placeholder; cap = min(queue limit, node max walltime)."
    )
    best = next((c for c in cands if c.fits), None)
    if best:
        print(
            f"\nbest: {best.site}/{best.target} queue {best.label} "
            f"({'starts now' if best.ready_hosts else 'expected wait ' + fmt_h(best.release_s)})"
        )
    return 0


def oarsub_args(c: Candidate, req: Request, name: str, project: str | None) -> list[str]:
    prop = f"{c.prop_key}='{c.target}'"
    if c.allowed_models:
        prop += " AND gpu_model IN (" + ",".join(f"'{m}'" for m in c.allowed_models) + ")"
    a: list[str] = []
    if c.label == "besteffort":
        a += ["-q", "besteffort", "-t", "besteffort", "-t", "idempotent"]
    else:
        a += ["-q", c.label]
    if c.exotic:
        a += ["-t", "exotic"]
    proj = project or c.group
    if proj:
        a += ["--project", proj]
    a += ["-p", prop, "-l", f"host=1/gpu={req.gpus},walltime={fmt_walltime(req.walltime_h)}", "-n", name]
    a += ["-O", f"logs/g5k/{name}.{c.site}.%jobid%.out", "-E", f"logs/g5k/{name}.{c.site}.%jobid%.err"]
    return a


def script_defaults(script: Path) -> dict[str, str]:
    """Read -n and -l (gpu=, walltime=) from the script's #OAR lines."""
    out: dict[str, str] = {}
    for line in script.read_text().splitlines():
        m = re.match(r"#OAR\s+(-\w)\s+(.*)", line.strip())
        if not m:
            continue
        flag, val = m.groups()
        if flag == "-n":
            out["name"] = val.strip()
        elif flag == "-l":
            if gm := re.search(r"gpu=(\d+)", val):
                out["gpus"] = gm.group(1)
            if wm := re.search(r"walltime=([\d:]+)", val):
                out["walltime"] = wm.group(1)
    return out


def job_info(site: str, jid: int) -> dict[str, Any]:
    out = run_on_site(site, ["oarstat", "-J", "-f", "-j", str(jid)])
    try:
        return dict(json.loads(out.stdout).get(str(jid), {}))
    except json.JSONDecodeError:
        return {}


def wait_for_schedule(site: str, jid: int, timeout_s: int) -> tuple[str, float]:
    """(state, seconds until start) once OAR has scheduled the job, or (state, inf) after timeout."""
    deadline = time.time() + timeout_s
    state = "?"
    while True:
        j = job_info(site, jid)
        state = j.get("state", "?")
        if state in OCCUPYING_STATES:
            return state, 0.0
        if state in ("Error", "Terminated"):
            return state, INF
        if j.get("scheduled_start"):
            return state, max(0.0, float(j["scheduled_start"]) - time.time())
        if time.time() > deadline:
            return state, INF
        time.sleep(8)


def cmd_submit(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parent.parent
    if not args.command:
        print("give the job after --, e.g. -- scripts/train.sh model=copula_nano", file=sys.stderr)
        return 2
    script = (repo / args.command[0]).resolve()
    if not script.is_file() or not os.access(script, os.X_OK):
        print(f"{script} must be an executable file in the checkout", file=sys.stderr)
        return 2
    defaults = script_defaults(script)
    args.gpus = args.gpus or int(defaults.get("gpus", 1))
    args.walltime = args.walltime or defaults.get("walltime")
    if not args.walltime:
        print("--walltime is required (the script has no #OAR -l walltime=)", file=sys.stderr)
        return 2
    name = args.name or defaults.get("name") or script.stem
    _, cands, req = collect(args)
    cands = [c for c in cands if c.fits]
    if not cands:
        print("no target accepts this walltime with your access; lower --walltime or add --besteffort", file=sys.stderr)
        return 1
    env = [f"{k}={os.path.realpath(os.environ[k])}" for k in FORWARD_ENV if os.environ.get(k)] + list(args.env)
    rel = str(script.relative_to(repo))
    job_cmd = shlex.join(["env", *env, f"./{rel}", *args.command[1:]] if env else [f"./{rel}", *args.command[1:]])
    max_wait = args.max_wait * 60
    kept: tuple[Candidate, int, float] | None = None
    for c in cands[: args.tries]:
        cmd = ["oarsub", *oarsub_args(c, req, name, args.project), job_cmd]
        print(f"\n[{c.site}/{c.target} {c.label}] {shlex.join(cmd)}")
        if args.dry_run:
            continue
        out = run_on_site(c.site, ["bash", "-c", f"mkdir -p logs/g5k && {shlex.join(cmd)}"], cwd=str(repo))
        m = re.search(r"OAR_JOB_ID=(\d+)", out.stdout)
        if not m:
            print(f"  rejected: {(out.stdout + out.stderr).strip()[-600:]}")
            continue
        jid = int(m.group(1))
        state, wait = wait_for_schedule(c.site, jid, args.schedule_timeout)
        print(f"  job {c.site}:{jid} state={state} expected start in {fmt_h(wait)}")
        if kept is None or wait < kept[2]:
            if kept is not None:
                run_on_site(kept[0].site, ["oardel", str(kept[1])])
                print(f"  cancelled {kept[0].site}:{kept[1]} (later start)")
            kept = (c, jid, wait)
        else:
            run_on_site(c.site, ["oardel", str(jid)])
            print(f"  cancelled {c.site}:{jid} (later start than {kept[0].site}:{kept[1]})")
        if kept[2] <= max_wait:
            break
    if args.dry_run:
        print("\ndry run: nothing submitted")
        return 0
    if kept is None:
        print("\nno submission was accepted", file=sys.stderr)
        return 1
    c, jid, wait = kept
    print(
        f"\nkept {c.site}:{jid} ({c.target}, queue {c.label}), expected start {fmt_h(wait)}\n"
        f"  logs:   {repo}/logs/g5k/{name}.{c.site}.{jid}.out\n"
        f"  watch:  ssh {c.site} oarstat -j {jid} -f\n"
        f"  cancel: ssh {c.site} oardel {jid}"
    )
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    sites = args.sites or (ABACA_SITES + DEFAULT_QUEUE_SITES)
    user = os.environ.get("USER", "")

    def one(site: str) -> list[str]:
        try:
            out = run_on_site(site, ["oarstat", "-J", "-f", "-u", user])
            jobs = json.loads(out.stdout or "{}")
        except Exception as exc:
            return [f"{site:9s} error: {exc}"]
        rows = []
        for jid, j in sorted(jobs.items(), key=lambda kv: int(kv[0])):
            start = j.get("start_time") or j.get("scheduled_start")
            when = time.strftime("%m-%d %H:%M", time.localtime(start)) if start else "-"
            hosts = ",".join(h.split(".")[0] for h in j.get("assigned_network_address") or [])
            rows.append(
                f"{site:9s} {jid:>9s} {str(j.get('name'))[:22]:22s} {j.get('state', '?'):9s} "
                f"{j.get('queue', '?'):10s} {when:11s} {hosts}"
            )
        return rows

    with ThreadPoolExecutor(len(sites)) as ex:
        for rows in ex.map(one, sites):
            for r in rows:
                print(r)
    return 0


def cmd_setup_site(args: argparse.Namespace) -> int:
    """Replicate the home-directory paths jobs depend on, on other sites' homes (shared group storage)."""
    home = Path.home()
    repo = Path(__file__).resolve().parent.parent
    # 1. home symlinks pointing into group storage (e.g. ~/thoth_storage)
    links = [p for p in home.iterdir() if p.is_symlink() and os.readlink(p).startswith("/srv/storage/")]
    # 2. the venv's interpreter, if it lives in this site's home (uv-managed python)
    py_real = Path(os.path.realpath(repo / ".venv/bin/python"))
    uv_python = home / ".local/share/uv/python"
    storage_root = Path(os.path.realpath(repo)).parent
    py_copy = storage_root / ".g5k-sites/uv-python"
    need_py = str(py_real).startswith(str(home) + "/")
    if need_py:
        print(f"venv interpreter {py_real} is in this site's home; mirroring {uv_python} -> {py_copy}")
        py_copy.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", f"{uv_python}/", f"{py_copy}/"], check=True)
    secrets = [home / s for s in args.secret_files] if args.copy_secrets else []
    rc = 0
    for site in args.sites:
        # OAR batch jobs source ~/.bashrc and warn when it is missing.
        steps = ["touch ~/.bashrc"]
        steps += [f"ln -sfn {shlex.quote(os.readlink(p))} ~/{shlex.quote(p.name)}" for p in links]
        if need_py:
            steps += [
                "mkdir -p ~/.local/share/uv",
                f"if [ -e ~/.local/share/uv/python ] && [ ! -L ~/.local/share/uv/python ]; then "
                f"echo 'skip: ~/.local/share/uv/python exists as a real dir'; "
                f"else ln -sfn {shlex.quote(str(py_copy))} ~/.local/share/uv/python; fi",
            ]
        steps.append(
            f"{shlex.quote(str(repo / '.venv/bin/python'))} -c 'import sys, torch; print(sys.version.split()[0], torch.__version__)'"
        )
        print(f"\n[{site}]")
        out = subprocess.run([*SSH, site, " && ".join(steps)], capture_output=True, text=True, timeout=300)
        print((out.stdout + out.stderr).strip() or "ok")
        rc |= out.returncode
        for s in secrets:
            if not s.is_file():
                print(f"  {s} missing here; skipped")
                continue
            rel = s.relative_to(home)
            subprocess.run([*SSH, site, f"mkdir -p ~/{shlex.quote(str(rel.parent))}"], check=False)
            cp = subprocess.run(["scp", "-q", "-p", "-o", "BatchMode=yes", str(s), f"{site}:{rel}"])
            subprocess.run([*SSH, site, f"chmod 600 ~/{shlex.quote(str(rel))}"], check=False)
            print(f"  copied ~/{rel} (mode 600)" if cp.returncode == 0 else f"  failed to copy ~/{rel}")
    return rc


# ----------------------------------------------------------------------------- entry point


def reexec_on_frontend(argv: list[str]) -> int:
    """From a laptop: run this script on the frontend that serves the sshfs-mounted checkout."""
    repo = Path(__file__).resolve().parent.parent
    host = os.environ.get("G5K_FRONTEND", "")
    rpath = os.environ.get("G5K_REPO", "")
    if not (host and rpath):
        for line in Path("/proc/mounts").read_text().splitlines():
            src, mnt, fstype = line.split()[:3]
            if fstype == "fuse.sshfs" and Path(mnt) == repo:
                h, _, p = src.partition(":")
                host, rpath = host or h.split("@")[-1], rpath or p
    if not (host and rpath):
        print("not on a Grid5000 frontend and no sshfs mount found; set G5K_FRONTEND and G5K_REPO", file=sys.stderr)
        return 2
    remote = f"cd {shlex.quote(rpath)} && /usr/bin/python3 scripts/g5k_gpus.py {shlex.join(argv)}"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, remote]).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_req(sp: argparse.ArgumentParser, walltime_default: str | None) -> None:
        sp.add_argument("--gpus", type=int, default=None if walltime_default is None else 1, help="GPUs on one host")
        sp.add_argument("--walltime", default=walltime_default, help="hours: 48, 36h, 90m, 12:30:00")
        sp.add_argument("--besteffort", action="store_true", help="also use besteffort-only targets (preemptable)")
        sp.add_argument("--default-queue", action="store_true", help="also use non-production clusters (default queue)")
        sp.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="regex of GPU models to skip ('' for none)")
        sp.add_argument("--min-vram", type=float, default=20.0, help="GB per GPU")
        sp.add_argument("--min-cc", type=float, default=7.0, help="CUDA compute capability")
        sp.add_argument("--sites", nargs="+", help=f"default {' '.join(ABACA_SITES)}")
        sp.add_argument("--prefer-site", default="grenoble", help="tie-break: the site holding the data")

    st = sub.add_parser("status", help="free GPUs per cluster you can use")
    add_req(st, "24")
    st.add_argument("--all", action="store_true", help="also show targets whose walltime cap is too short")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_status)

    sb = sub.add_parser("submit", help="submit where the job starts soonest")
    add_req(sb, None)
    sb.add_argument("--name")
    sb.add_argument("--project", help="OAR project (default: the group that grants the access level)")
    sb.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE for the job")
    sb.add_argument("--max-wait", type=float, default=20, help="minutes; accept the first target starting sooner")
    sb.add_argument("--tries", type=int, default=4, help="targets to try, in ranked order")
    sb.add_argument("--schedule-timeout", type=int, default=90, help="seconds to wait for OAR to schedule a job")
    sb.add_argument("--dry-run", action="store_true", help="print the oarsub commands only")
    sb.add_argument("command", nargs=argparse.REMAINDER, help="-- <script relative to the repo> [args]")
    sb.set_defaults(func=cmd_submit)

    jb = sub.add_parser("jobs", help="your jobs on every site")
    jb.add_argument("--sites", nargs="+")
    jb.set_defaults(func=cmd_jobs)

    ss = sub.add_parser("setup-site", help="make the venv and storage paths work on other sites (one time)")
    ss.add_argument("sites", nargs="+")
    ss.add_argument("--copy-secrets", action="store_true", help="also copy --secret-files to the remote homes")
    ss.add_argument(
        "--secret-files",
        nargs="+",
        default=[".netrc", ".config/copula-inter/tabpfn_token", ".config/tabpfn_token"],
        help="paths relative to ~ (missing ones are skipped); <checkout>/.env needs no copy",
    )
    ss.set_defaults(func=cmd_setup_site)

    if not on_g5k_frontend():
        return reexec_on_frontend(sys.argv[1:])
    args = p.parse_args()
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
