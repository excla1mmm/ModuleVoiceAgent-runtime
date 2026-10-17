"""Latency statistics per pipeline stage and counters, exposed on /health (JSON) and /metrics (Prometheus).

Service-wide numbers are for the operator; per-tenant numbers (active calls, calls, rejected calls, bundles)
let the operator see which PBX is loading the runtime and let each PBX see its own usage.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque

from .usage import COUNTERS as USAGE_COUNTERS

STAGES = ("latency_ms", "stt_ms", "llm_first_ms", "first_audio_ms")
WINDOW = 2000
TENANT_LABEL = "tenant"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(p / 100 * (len(ordered) - 1))]


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stages: dict[str, deque] = {s: deque(maxlen=WINDOW) for s in STAGES}
        self._counters: dict[str, int] = defaultdict(int)
        self._tenant_counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.active_calls = 0
        self.tenant_active: dict[str, int] = defaultdict(int)

    def observe(self, stage: str, value_ms: float) -> None:
        if stage in self._stages:
            with self._lock:
                self._stages[stage].append(float(value_ms))

    def inc(self, counter: str, value: int = 1) -> None:
        with self._lock:
            self._counters[counter] += value

    def inc_tenant(self, tenant: str, counter: str, value: int = 1) -> None:
        with self._lock:
            self._tenant_counters[tenant][counter] += value

    # Call slots are taken and released on the event loop only, so no lock is needed here.
    def call_started(self, tenant: str) -> None:
        self.active_calls += 1
        self.tenant_active[tenant] += 1

    def call_ended(self, tenant: str) -> None:
        self.active_calls -= 1
        self.tenant_active[tenant] -= 1
        if self.tenant_active[tenant] <= 0:
            self.tenant_active.pop(tenant, None)

    def tenant_snapshot(self, tenant: str) -> dict:
        with self._lock:
            counters = dict(self._tenant_counters.get(tenant, {}))
        return {"active_calls": self.tenant_active.get(tenant, 0), "counters": counters}

    def snapshot(self) -> dict:
        with self._lock:
            stages = {s: list(v) for s, v in self._stages.items()}
            counters = dict(self._counters)
            tenants = {t: dict(c) for t, c in self._tenant_counters.items()}
        for tenant in self.tenant_active:
            tenants.setdefault(tenant, {})
        return {
            "active_calls": self.active_calls,
            "counters": counters,
            "stages": {s: {"count": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95)}
                       for s, v in stages.items()},
            "tenants": {t: {"active_calls": self.tenant_active.get(t, 0), "counters": c}
                        for t, c in sorted(tenants.items())},
        }

    def prometheus(self, extra_gauges: dict[str, float], tenant_gauges: dict[str, dict[str, float]] | None = None,
                   usage: dict[str, dict[str, float]] | None = None) -> str:
        snap = self.snapshot()
        lines = ["# TYPE voice_agent_active_calls gauge", f"voice_agent_active_calls {snap['active_calls']}"]
        for name, value in sorted(extra_gauges.items()):
            lines += [f"# TYPE voice_agent_{name} gauge", f"voice_agent_{name} {value}"]
        for name, value in sorted(snap["counters"].items()):
            lines += [f"# TYPE voice_agent_{name}_total counter", f"voice_agent_{name}_total {value}"]
        for stage, data in snap["stages"].items():
            metric = f"voice_agent_{stage.removesuffix('_ms')}_milliseconds"
            lines.append(f"# TYPE {metric} summary")
            for q, key in ((0.5, "p50"), (0.95, "p95")):
                if data[key] is not None:
                    lines.append(f'{metric}{{quantile="{q}"}} {data[key]}')
            lines.append(f"{metric}_count {data['count']}")
        tenants = snap["tenants"]
        if tenants:
            lines.append("# TYPE voice_agent_tenant_active_calls gauge")
            for tenant, data in tenants.items():
                lines.append(f'voice_agent_tenant_active_calls{{{TENANT_LABEL}="{_label(tenant)}"}} {data["active_calls"]}')
            names = sorted({name for data in tenants.values() for name in data["counters"]})
            for name in names:
                lines.append(f"# TYPE voice_agent_tenant_{name}_total counter")
                for tenant, data in tenants.items():
                    if name in data["counters"]:
                        lines.append(f'voice_agent_tenant_{name}_total{{{TENANT_LABEL}="{_label(tenant)}"}} '
                                     f'{data["counters"][name]}')
        for name, per_tenant in sorted((tenant_gauges or {}).items()):
            lines.append(f"# TYPE voice_agent_tenant_{name} gauge")
            for tenant, value in sorted(per_tenant.items()):
                lines.append(f'voice_agent_tenant_{name}{{{TENANT_LABEL}="{_label(tenant)}"}} {value}')
        if usage:  # all-time billing counters per tenant, from the usage meter
            for counter in USAGE_COUNTERS:
                metric = f"voice_agent_usage_{counter}_total"
                lines.append(f"# TYPE {metric} counter")
                lines += [f'{metric}{{{TENANT_LABEL}="{_label(tenant)}"}} {totals[counter]}'
                          for tenant, totals in sorted(usage.items())]
        return "\n".join(lines) + "\n"
