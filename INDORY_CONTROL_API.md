# Indory Control API

Server-client contract between `indory_server` (Spring Boot orchestrator) and
`indory_lerobot` (policy runtime). This document is the authoritative spec for
the long-lived daemon mode that replaces the legacy subprocess / stdout-capture
flow.

> Status: **spec only**, no implementation yet. When the daemon lands in
> `src/lerobot/control/`, this document becomes its contract.

## 1. Goals

- `indory_server` launches `indory_lerobot` once as a long-lived daemon,
  not once per TASK.
- `indory_server` invokes TASKs (`GRABBING_PARCEL`, `DROPPING_OFF`, ...) over
  a request-response channel and receives structured status + event streams
  without parsing stdout.
- The wire format stays consistent with the existing Pi adapter
  (`xlerobot_v1.1`) — **msgpack + ZMQ** — so no new serializer dependency.
- The DROP leader publisher (Mac → super, port 8892) keeps its existing
  channel; control events use a separate port to avoid collision.

## 2. Non-goals (intentionally out of scope)

- Replacing the existing `lerobot-teleoperate` / `lerobot-record` CLI flows.
  Direct operators keep using those.
- Replacing the Pi-side `xlerobot_v1.1` ZMQ contract (state / command / RPC
  / camera). The control API is the **north-bound** interface from
  `indory_server`, not the south-bound interface to Pi.
- TLS / mTLS / OAuth. First version assumes a private network (Tailscale,
  firewall) like the existing `8856` command port. Auth is a shared bearer
  token, not a full identity layer.

## 3. Transport

| Channel | Endpoint | ZMQ pattern | Purpose |
|---|---|---|---|
| **Control** | `tcp://*:8891` | `REQ` ↔ `REP` | Command / response. One request per REQ, one response per REP. |
| **Events** | `tcp://*:8893` | `PUB` → `SUB` | Structured event stream. Subscribe by `run_id` filter at the client side. |

Why two ports:

- The existing DROP supervisor binds `8892` as `PULL` to receive Mac leader
  actions. A control `PUB` on `8892` would either collide with the leader
  PULL or confuse existing DROP wiring. `8893` is reserved for control
  events so the DROP path stays untouched.
- REQ/REP is the right pattern for synchronous commands (`start_run`,
  `stop_run`, `get_run_status`). Push events go over PUB/SUB so subscribers
  don't have to poll.

Wire format: **MessagePack**, `use_bin_type=True`, single frame per REQ/REP
message, multi-frame OK for PUB events (topic prefix optional but
recommended: `b"run.<run_id>"`).

Default bind host: `0.0.0.0`. Override with `INDORY_CONTROL_BIND_HOST` /
`INDORY_EVENTS_BIND_HOST`.

## 4. Authentication

- Shared bearer token in environment variable `INDORY_CONTROL_API_TOKEN`.
- When unset, the daemon binds but **rejects every request** with
  `AUTH_REQUIRED`. This is the recommended production mode — leaving auth
  off is only for local debug.
- When set, every request must include `auth_token` matching the daemon's
  token. `AUTH_FAILED` (`error.code`) on mismatch.
- Tokens are compared with `hmac.compare_digest` to avoid timing side
  channels. Logging must mask the token (`token=***`).

## 5. Schema versioning

- Every request, response, and event carries `"schema": "indory_control_v1"`.
- The daemon refuses a request whose `schema` does not start with
  `indory_control_v1` and replies with `UNSUPPORTED_SCHEMA`.
- Future minor additions increment `indory_control_v1.1`, `indory_control_v1.2`, …
  Major redesigns jump to `indory_control_v2`.

## 6. Request envelope

```python
{
    "op":          "<operation>",                    # required
    "schema":      "indory_control_v1",              # required
    "id":          "<client-supplied correlation id>",  # required, unique per REQ
    "auth_token":  "<bearer>",                       # required when daemon has a token

    # operation-specific fields below
}
```

The daemon matches each REQ to one REP by FIFO order. The `id` is **not**
used for routing — it is only echoed in the response for client-side
correlation. Clients SHOULD generate `id` as a UUIDv4.

## 7. Response envelope

```python
{
    "ok":      True | False,            # required
    "schema":  "indory_control_v1",    # required
    "id":      "<same as request>",    # required

    "result":  { ... },                # present when ok=True, op-specific
    "error":   {                       # present when ok=False
        "code":    "<UPPER_SNAKE>",
        "message": "<human readable>",
        "details": { ... }            # optional, op-specific context
    }
}
```

The daemon never closes the REP socket on error — it always sends a REP
with `ok=False`. Clients SHOULD treat a closed socket as
`TRANSPORT_LOST` and reconnect.

## 8. Operations

### 8.1 `health`

Liveness + version + minimal capability summary. Intended for Spring's
`@Scheduled` health checks and for the operator Mac to confirm the daemon
is up before launching the leader publisher.

Request:
```python
{ "op": "health", "schema": "indory_control_v1", "id": "..." }
```

Response (`ok=True`):
```python
{
    "ok":     True,
    "schema": "indory_control_v1",
    "id":     "...",
    "result": {
        "status":          "ok",                       # "ok" | "degraded" | "starting" | "shutting_down"
        "daemon_version":  "0.5.1+indory.control.v1",
        "indory_lerobot_version": "0.5.1",
        "uptime_s":        12345.6,
        "active_runs":     0,                          # currently RUNNING runs
        "max_concurrent_runs": 1,                      # see §10
        "loaded_policies": ["act_indory_100k"]         # optional, names only
    }
}
```

### 8.2 `list_capabilities`

Static description of what the daemon can do. Spring uses this to validate
operator requests before issuing `start_run`.

Response (`ok=True`):
```python
{
    "ok":     True,
    "schema": "indory_control_v1",
    "id":     "...",
    "result": {
        "stages": [
            {
                "name":           "GRABBING_PARCEL",
                "description":    "ACT live runner guarded by success detector",
                "skill_runner":   "scripts.indory_act_live",
                "params_schema":  { ... },              # JSON Schema-like dict
                "required_params": ["checkpoint", "dataset_root", "repo_id", "task"],
                "default_params":  { ... },
                "stoppable":       True,
                "emits_success":   True
            },
            {
                "name":           "DROPPING_OFF",
                "description":    "Supervised DROP teleop; gated by success detector before forwarding",
                "skill_runner":   "scripts.indory_drop_supervised_teleop",
                "params_schema":  { ... },
                "required_params": ["remote_ip"],
                "default_params":  { ... },
                "stoppable":       True,
                "emits_success":   True
            }
        ],
        "camera_transports":     ["zmq", "rtp_udp", "cam_bridge"],
        "policy_types":          ["act", "pi05", "pi0", "groot"],
        "supported_checkpoints": ["/.../*.safetensors", "/.../pretrained_model"],
        "max_concurrent_runs":   1,
        "command_lease_ms":      300
    }
}
```

### 8.3 `start_run`

Start a TASK in the background. Returns immediately with the assigned
`run_id`; progress arrives on the events channel.

Request:
```python
{
    "op":         "start_run",
    "schema":     "indory_control_v1",
    "id":         "...",
    "run_id":     "<optional client id>",   # if omitted, daemon assigns a uuid
    "stage":      "GRABBING_PARCEL",
    "params": {
        # all keys from list_capabilities().params_schema
        "checkpoint":          "/home/indory/indory_lerobot/outputs/train/.../pretrained_model",
        "dataset_root":        "/home/indory/indory_lerobot/data/.../head_patched",
        "repo_id":             "hanbin5/indory_xlerobot_pick_delivery",
        "task":                "place the blue parcel into the basket in front of you",
        "remote_ip":           "100.127.146.68",
        "fps":                 15.0,
        "duration_s":          180.0,
        "max_relative_target": 45.0,
        "command_lease_ms":    300,
        "camera_transport":    "cam_bridge",
        "cam_bridge_base_url": "ws://127.0.0.1:8870",
        "success_detector":    "sky-blue-parcel-color",
        "success_detector_kwargs": { ... },
        "send":                True,
        "device":              "cuda",
        # ... plus detector / arm_recenter / gripper_latch / debug_web_* knobs
    },
    "timeout_s":  180.0,                  # optional, daemon wall-clock cap; >0 enforced
    "metadata":   { ... }                 # optional, opaque, echoed in events
}
```

Validation rules:

- `params` MUST validate against `list_capabilities().stages[name].params_schema`.
  Mismatch → `INVALID_PARAMS` with the validator's details.
- `run_id` collision → `RUN_ALREADY_ACTIVE` (or `RUN_ID_REUSED` if the previous
  run with that id is in a terminal state; daemons MAY allow reuse for
  idempotency, but Spring's TASK state machine SHOULD mint new ids per
  attempt).
- A run with the same `stage` is already active → `RUN_ALREADY_ACTIVE`.
- `max_concurrent_runs` exceeded → `BUSY`.

Response (`ok=True`):
```python
{
    "ok":     True,
    "schema": "indory_control_v1",
    "id":     "...",
    "result": {
        "run_id":  "<assigned or echoed>",
        "stage":   "GRABBING_PARCEL",
        "status":  "starting",            # starting | running | succeeded | failed | stopped
        "pid":     12345,                 # worker process pid (forked, not the daemon)
        "started_at_ns": 1782138890123456789
    }
}
```

### 8.4 `stop_run`

Request a graceful stop. The daemon sends SIGINT to the worker, waits
`stop_grace_s` (default 5s), then SIGKILL.

Request:
```python
{
    "op":      "stop_run",
    "schema":  "indory_control_v1",
    "id":      "...",
    "run_id":  "<target>",
    "reason":  "operator_cancel" | "stage_advance" | "timeout" | "external_failure",
    "force":   False                     # True skips SIGINT, sends SIGKILL immediately
}
```

Response (`ok=True`):
```python
{
    "ok":     True,
    "schema": "indory_control_v1",
    "id":     "...",
    "result": {
        "run_id":        "<target>",
        "stopped":       True,
        "final_status":  "stopped" | "failed",
        "stopped_at_ns": 1782138890123456789
    }
}
```

`stop_run` returns even if the run was already in a terminal state; the
result reflects the latest known status.

### 8.5 `get_run_status`

Synchronous snapshot. Use this for Spring's `GET /api/tasks/{id}`.

Request:
```python
{
    "op":     "get_run_status",
    "schema": "indory_control_v1",
    "id":     "...",
    "run_id": "<target>"
}
```

Response (`ok=True`):
```python
{
    "ok":     True,
    "schema": "indory_control_v1",
    "id":     "...",
    "result": {
        "run_id":         "<id>",
        "stage":          "GRABBING_PARCEL",
        "status":         "running",
        "started_at_ns":  ...,
        "step":           42,                       # last reported step
        "elapsed_s":      2.81,
        "last_event_seq": 117,                      # monotonic from events channel
        "last_event":     "run_step",
        "last_log_line":  "step=42/2700 ...",       # tail of last log event, truncated
        "pid":            12345,
        "exit_code":      null,                     # only present when terminal
        "error":          null                      # only present when failed
    }
}
```

### 8.6 `list_runs`

Enumerate runs. Optional filter:

Request:
```python
{
    "op":     "list_runs",
    "schema": "indory_control_v1",
    "id":     "...",
    "filter": {
        "status_in":   ["starting", "running"],
        "stage":       "GRABBING_PARCEL",
        "since_ns":    1782138890000000000,
        "limit":       50
    }
}
```

Response (`ok=True`):
```python
{
    "ok":     True,
    "schema": "indory_control_v1",
    "id":     "...",
    "result": {
        "runs": [ <run_summary>, ... ]
    }
}
```

### 8.7 `shutdown`

Stop the daemon. Used by `run_indory_web.sh` shutdown hooks.

Request:
```python
{ "op": "shutdown", "schema": "indory_control_v1", "id": "..." }
```

Response: `ok=True`. The daemon then closes REP, then closes PUB after a
short drain delay (default 2s).

## 9. Event stream (PUB :8893)

Subscribe by connecting a SUB socket to `tcp://<host>:8893` and
`SUBSCRIBE` to a topic prefix:

- `b""` — receive **all** events from all runs.
- `b"run.<run_id>."` — receive only events for one run.
- `b"stage.GRABBING_PARCEL."` — receive by stage.

The daemon MAY also support a JSON-encoded in-payload filter for clients
that cannot do topic matching, but topic matching is the primary contract.

Event envelope:

```python
{
    "schema":   "indory_control_v1",
    "event":    "<event name>",
    "run_id":   "<run>",
    "seq":      117,                              # monotonic per-run
    "ts_ns":    1782138890123456789,
    "payload":  { ... }                           # event-specific
}
```

Event types:

| Event | When emitted | Payload |
|---|---|---|
| `run_queued` | right after `start_run` accepted | `{stage, params_summary}` |
| `run_started` | worker process spawned, before first step | `{pid, stage, started_at_ns}` |
| `run_step` | each skill step (configurable rate, default ≤10Hz) | `{step, elapsed_s, max_joint_delta, base_cmd, status, sent_action_summary}` |
| `run_log` | worker emits a stdout/stderr line worth surfacing | `{stream: "stdout"|"stderr", line, level}` |
| `run_progress` | detector / camera / stage-specific intermediate | `{kind: "detector_fired"|"arm_recentered"|"camera_stale"|..., metadata}` |
| `run_success` | detector confirmed success and policy/stop sequence finished | `{reason, metadata, elapsed_s, step}` |
| `run_stopped` | `stop_run` honored | `{reason, final_status, stopped_at_ns}` |
| `run_failed` | worker exited non-zero, timed out, or hit fatal detector state | `{exit_code, stderr_tail, error_code, message, fatal_at_step}` |
| `daemon_status` | periodic heartbeat, default 5s | `{active_runs, uptime_s, status}` |

Clients SHOULD treat missed `seq` values as a gap and may request
`get_run_status` to backfill.

## 10. Concurrency model

- Default `max_concurrent_runs = 1`. The Pi command lease (`lease_ms=300`)
  already enforces single-owner semantics on the south-bound ZMQ 8856.
  Running multiple GRAB/DROP stages against the same Pi in parallel would
  race for the lease and produce unsafe interleavings.
- The daemon enforces this server-side and returns `BUSY` on
  `start_run` when the limit is reached.
- The worker for each run is a **separate child process** spawned via
  `multiprocessing` or `subprocess.Popen`, NOT a thread. This keeps the
  ZMQ REP loop responsive and isolates crashes.
- Workers inherit the conda env (`lerobot`) and the working dir
  (`/home/indory/indory_lerobot`) used by the daemon, so Spring does not
  need to manage envs per call.

## 11. Lifecycle / state machine

```
              start_run accepted
                       │
                       ▼
                  ┌─────────┐
                  │ queued  │
                  └────┬────┘
                       │ worker spawned
                       ▼
                  ┌─────────┐
                  │starting │
                  └────┬────┘
                       │ first run_step OR run_started emitted
                       ▼
   stop_run ◄──┐    ┌─────────┐    ┌──► run_success
                │    │ running │    │
                └───►└────┬────┘◄───┘
                       │
                       │ exit_code != 0 OR fatal detector OR timeout
                       ▼
                  ┌─────────┐
                  │ failed  │
                  └─────────┘

stop_run path:  running ──► stopped (terminal)
all terminal states emit one last event with the final status.
```

Terminal states (`succeeded`, `failed`, `stopped`) keep the run record in
memory for `list_runs` until retention (default 1 hour) then drop it.
Clients SHOULD treat `get_run_status` after retention as
`UNKNOWN_RUN`.

## 12. Error codes

| Code | Meaning | Retryable by client |
|---|---|---|
| `AUTH_REQUIRED` | daemon has no token configured and request omitted one | no (fix config) |
| `AUTH_FAILED` | token mismatch | no (fix token) |
| `INVALID_REQUEST` | malformed envelope, missing required field | no |
| `UNSUPPORTED_SCHEMA` | `schema` not in `indory_control_v1.*` | no |
| `UNKNOWN_OP` | `op` not in §8 | no |
| `INVALID_PARAMS` | `params` failed `params_schema` validation | no |
| `RUN_ALREADY_ACTIVE` | a run with the same id / stage is in flight | no |
| `RUN_ID_REUSED` | (optional) client reused a terminal-state id | maybe |
| `BUSY` | `max_concurrent_runs` reached | yes (after backoff) |
| `UNKNOWN_RUN` | `run_id` not found or retention expired | maybe |
| `WORKER_SPAWN_FAILED` | could not start child process | yes |
| `INTERNAL` | unexpected daemon error | yes |
| `TRANSPORT_LOST` | client side: REP socket closed without reply | yes (reconnect) |

Error envelope:

```python
{
    "ok":     False,
    "schema": "indory_control_v1",
    "id":     "...",
    "error":  {
        "code":    "INVALID_PARAMS",
        "message": "params.camera_transport must be one of zmq|rtp_udp|cam_bridge",
        "details": {
            "field":  "camera_transport",
            "value":  "webrtc",
            "allowed": ["zmq", "rtp_udp", "cam_bridge"]
        }
    }
}
```

## 13. Configuration

| Env var | Default | Purpose |
|---|---|---|
| `INDORY_CONTROL_API_TOKEN` | unset → reject all | shared bearer token |
| `INDORY_CONTROL_BIND_HOST` | `0.0.0.0` | REQ/REP bind host |
| `INDORY_CONTROL_PORT` | `8891` | REQ/REP port |
| `INDORY_EVENTS_BIND_HOST` | `0.0.0.0` | PUB bind host |
| `INDORY_EVENTS_PORT` | `8893` | PUB port |
| `INDORY_CONTROL_MAX_CONCURRENT_RUNS` | `1` | enforced server-side |
| `INDORY_CONTROL_DEFAULT_STOP_GRACE_S` | `5` | SIGINT → SIGKILL delay |
| `INDORY_CONTROL_RUN_RETENTION_S` | `3600` | terminal-state retention |
| `INDORY_CONTROL_EVENT_HEARTBEAT_S` | `5` | `daemon_status` cadence |
| `INDORY_CONTROL_LOG_LEVEL` | `INFO` | one of `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `INDORY_CONTROL_LOG_TOKEN` | `***` | how to mask `auth_token` in logs |

Spring side (`application-local.yml`):

```yaml
indoory:
  control:
    enabled:        true
    host:           "127.0.0.1"          # where the daemon binds
    port:           8891
    events_port:    8893
    auth_token:     "${INDORY_CONTROL_API_TOKEN}"
    request_timeout_s: 30
    reconnect_backoff_s: [1, 2, 5, 10]
```

## 14. End-to-end example: GRAB

Spring orchestrator:

```text
1. Spring boots, sees `indoory.control.enabled=true`, ensures the daemon
   is running (ProcessBuilder("python -m lerobot.control.daemon") or
   systemd unit). Daemon binds :8891 + :8893.
2. Operator clicks "Run GRAB on dataset X".
3. Spring validates request against `list_capabilities`.
4. Spring issues `start_run` (REQ).
5. Daemon validates, spawns worker for `indory_act_live.py` with the
   requested params, returns REP with `run_id` + `pid`.
6. Daemon emits `run_queued`, `run_started`, then `run_step` events on
   :8893. Spring's Spring `RosControlAdapter` (or new
   `RosIndooryControlAdapter`) subscribes and forwards to its own
   TaskEventRepository / WebSocket fan-out.
7. Worker hits `success_detector` → emits `run_success`.
8. Spring sees success → advances TASK stage → issues `stop_run` for
   the worker to release lease and write final log.
9. Daemon returns REP for `stop_run` with `final_status=stopped`.
```

## 15. End-to-end example: DROP

DROP is **multi-machine**: Mac leader publisher + super-side supervisor.
Only the **super-side supervisor** is the control-API run. Spring does
NOT manage the Mac publisher over this API; that stays as SSH.

```text
1. Spring `start_run` with `stage=DROPPING_OFF`.
2. Daemon spawns `indory_drop_supervised_teleop.py` (worker).
3. Worker binds `tcp://*:8892` PULL and connects to the Pi adapter.
4. Spring (separately, via its existing `DropTeleopSessionProperties`)
   SSHes into the operator Mac (`hanbin5@100.81.219.12`) and launches
   `indory_mac_leader_publisher.py --server-url tcp://super:8892`.
5. Mac publisher PUSHes leader actions to worker's 8892 PULL.
6. Worker runs the success detector and (when `--send` is true)
   forwards gated actions to the Pi adapter on 8856.
7. On success → `run_success` event. Spring cancels SSH'd Mac publisher
   and stops the worker via `stop_run`.
```

The control API MUST NOT touch port 8892 or any Mac-side processes; the
SSH orchestration stays in Spring's `LerobotSkillRunnerAdapter`.

## 16. Backward compatibility / migration

- The current `LerobotSkillRunnerAdapter` (subprocess + stdout) keeps
  working as a fallback while the daemon lands.
- A side-by-side flag `indoory.lerobot.legacy_subprocess_mode=true`
  forces the old path even when `indoory.control.enabled=true`.
- During transition, both modes may be selectable per TASK by Spring.

## 17. Open questions / TBD

- **TLS**: future `tls://` variant of `INDORY_CONTROL_BIND_HOST`. Not in v1.
- **mTLS**: needed when the operator Mac and the controller host are on
  different trust boundaries. Not in v1.
- **HTTP `/healthz`**: useful for systemd / k8s probes. v1.1 candidate.
- **Run cancellation token**: do we need a separate `cancel` op distinct
  from `stop_run`? Current design collapses both. Reopen if Spring needs
  to distinguish operator-cancel vs stage-advance.
- **Multi-tenant**: one daemon per Spring is the v1 assumption. Multi-tenant
  would need per-client auth + run scoping.
- **Replay of past runs**: do we want `INDORY_CONTROL_PERSIST_RUNS=true`
  with disk-backed log retention for offline review? Out of scope for v1.
- **Policy hot-reload**: should `start_run` allow loading a new checkpoint
  without restarting the daemon? Probably yes (warm-load weights), but
  v1 keeps it simple — each `start_run` re-loads from disk.

## 18. Cross-references

- `README.md` — high-level fork overview, runtime roles.
- `INDORY_ZMQ_ROLES.md` — the Pi-side adapter contract (8855/8856/8857/8866/8867).
  This control API is **independent** of that contract and lives above it.
- `scripts/indory_act_live.py` — GRAB worker implementation.
- `scripts/indory_drop_supervised_teleop.py` — DROP super-side worker.
- `scripts/indory_mac_leader_publisher.py` — DROP Mac-side (managed by
  Spring over SSH, not by this API).
