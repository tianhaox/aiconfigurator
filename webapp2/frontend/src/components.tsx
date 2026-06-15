import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, MemGroup, OpRow, Phase } from "./api";

export const SOURCE_COLORS: Record<string, string> = {
  silicon: "#4ade80",
  empirical: "#fbbf24",
  sol: "#60a5fa",
  mixed: "#c084fc",
  unknown: "#9ca3af",
};
export const MEM_COLORS = ["#60a5fa", "#4ade80", "#fbbf24", "#f87171", "#c084fc", "#94a3b8"];

export function fmtVal(value: number): string {
  return value >= 100 ? value.toFixed(0) : value.toFixed(2);
}

export function fmtKvPerToken(bytes: number | null): string {
  if (!bytes) return "—";
  const kib = bytes / 1024;
  return kib >= 1024 ? `${(kib / 1024).toFixed(2)} MiB` : `${kib.toFixed(1)} KiB`;
}

export function Field({ label, v, on }: { label: string; v: number; on: (n: number) => void }) {
  return (
    <div>
      <label>{label}</label>
      <input type="number" value={v} onChange={(e) => on(+e.target.value)} />
    </div>
  );
}

export function ScalarCard({ c }: { c: Card }) {
  return (
    <div className="card">
      <div className="card-value">
        {fmtVal(c.value)}
        {c.unit && <span className="card-unit">{c.unit}</span>}
      </div>
      <div className="card-label">{c.label}</div>
    </div>
  );
}

export function PhaseChart({ phase }: { phase: Phase }) {
  const rows = phase.rows;
  if (!rows.length) return null;
  const height = Math.max(180, rows.length * 26 + 40);
  return (
    <div className="panel">
      <h3>{phase.name}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} layout="vertical" margin={{ left: 20, right: 30 }}>
          <XAxis type="number" stroke="#94a3b8" fontSize={11} unit="ms" />
          <YAxis
            type="category"
            dataKey="op"
            width={180}
            stroke="#94a3b8"
            fontSize={11}
            tickFormatter={(s: string) => s.replace(/^(context|generation)_/, "")}
          />
          <Tooltip
            cursor={{ fill: "rgba(148,163,184,0.1)" }}
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
            formatter={(v: number, _n, p: any) => [`${v.toFixed(3)} ms`, (p.payload as OpRow).source]}
          />
          <Bar dataKey="ms" radius={[0, 3, 3, 0]}>
            {rows.map((r, i) => (
              <Cell key={i} fill={SOURCE_COLORS[r.source] || SOURCE_COLORS.unknown} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MemoryPanel({ group }: { group: MemGroup }) {
  const rows = [...group.rows].sort((a, b) => b.gib - a.gib);
  if (!rows.length) return null;
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>{group.label}</h3>
        <span className="kv-stat">
          KV cache: <b>{fmtKvPerToken(group.kv_bytes_per_token)}</b> / token
        </span>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(160, rows.length * 38 + 30)}>
        <BarChart data={rows} layout="vertical" margin={{ left: 20, right: 50 }}>
          <XAxis type="number" stroke="#94a3b8" fontSize={11} unit=" GiB" />
          <YAxis type="category" dataKey="name" width={90} stroke="#94a3b8" fontSize={12} />
          <Tooltip
            cursor={{ fill: "rgba(148,163,184,0.1)" }}
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
            formatter={(v: number) => `${v.toFixed(3)} GiB`}
          />
          <Bar
            dataKey="gib"
            radius={[0, 3, 3, 0]}
            label={{ position: "right", fill: "#cbd5e1", fontSize: 11, formatter: (v: number) => v.toFixed(1) }}
          >
            {rows.map((_, i) => (
              <Cell key={i} fill={MEM_COLORS[i % MEM_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
