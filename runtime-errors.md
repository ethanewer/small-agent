# Runtime Errors: 2026-03-07 Benchmark Run

## Summary

3 of 15 tasks failed with `RuntimeError` before the agent was invoked.
All three share the same root cause: **Docker ran out of network address pools**.

| Task | Error |
|---|---|
| `gcode-to-text` | `all predefined address pools have been fully subnetted` |
| `financial-document-processor` | `all predefined address pools have been fully subnetted` |
| `filter-js-from-html` | `all predefined address pools have been fully subnetted` |

## Root Cause

The benchmark ran with `--n-concurrent 5`, meaning up to 5 Docker Compose
environments were active simultaneously. Each environment creates its own
Docker bridge network, which consumes a subnet from Docker's default address
pool. The pool was exhausted before these three tasks could create their
networks.

The failures occurred inside Harbor's environment setup
(`harbor.environments.docker.docker.DockerEnvironment.start`), during the
`docker compose up --detach --wait` step. The agent's `setup()` and `run()`
methods were never called — both `agent_setup` and `agent_execution` are `null`
in the result JSON for all three tasks.

## Not an Agent Bug

No changes are needed in `agents/qwen/` or `harbor/agent.py`. The qwen agent
code was never reached. This is a local Docker infrastructure issue.

## Mitigations

1. **Reduce concurrency** — use `--n-concurrent 3` or `4` to keep fewer
   Docker networks alive at once.
2. **Expand Docker's address pool** — edit `~/.docker/daemon.json`:
   ```json
   {
     "default-address-pools": [
       {"base": "172.17.0.0/12", "size": 24},
       {"base": "192.168.0.0/16", "size": 24}
     ]
   }
   ```
   Then restart Docker.
3. **Prune stale networks before a run** — `docker network prune -f` to
   reclaim subnets leaked by previous runs.
