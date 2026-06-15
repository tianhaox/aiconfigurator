import { useRef, useState } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SweepResult, generateTask, getSweep, startSweepYaml } from "../api";

const EXAMPLE_YAML = `serving_mode: agg
model_path: Qwen/Qwen3-32B
system_name: h200_sxm
backend_name: trtllm
total_gpus: 8
isl: 4000
osl: 1000
ttft: 1000.0
tpot: 50.0
agg_num_gpu_candidates: [2, 4, 8]
agg_tp_candidates: [1, 2, 4, 8]`;

export default function SweepPage() {
  const [yamlText, setYamlText] = useState(EXAMPLE_YAML);
  const [prompt, setPrompt] = useState("");
  const [genInfo, setGenInfo] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SweepResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  async function generate() {
    if (!prompt.trim()) return;
    setGenerating(true);
    setGenInfo(null);
    setError(null);
    try {
      const r = await generateTask(prompt);
      setYamlText(r.yaml);
      setGenInfo(
        r.valid
          ? `✓ valid (${r.attempts} attempt${r.attempts > 1 ? "s" : ""})`
          : `⚠ generated but failed validation: ${r.error}`
      );
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setGenerating(false);
    }
  }

  async function run() {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const { job_id } = await startSweepYaml(yamlText);
      const poll = async () => {
        const job = await getSweep(job_id);
        if (job.status === "running") {
          pollRef.current = window.setTimeout(poll, 1500);
          return;
        }
        setRunning(false);
        if (job.status === "error") setError(job.error || "sweep failed");
        else setResult(job.result);
      };
      poll();
    } catch (e) {
      setRunning(false);
      setError(String(e instanceof Error ? e.message : e));
    }
  }

  const feasible = (result?.points || []).filter((p) => !p.on_pareto);
  const pareto = (result?.points || []).filter((p) => p.on_pareto);

  return (
    <div className="layout">
      <aside className="form wide">
        <label>✨ Describe the experiment (LLM → Task YAML)</label>
        <textarea
          className="nl"
          rows={3}
          placeholder="e.g. DeepSeek-V3 on h200 trtllm fp8, agg, ISL 4k OSL 1k, sweep TP 1-8 over 8/16 GPUs"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button className="go alt" onClick={generate} disabled={generating || !prompt.trim()}>
          {generating ? "Generating…" : "Generate YAML"}
        </button>
        {genInfo && <div className={genInfo.startsWith("✓") ? "gen-ok" : "gen-warn"}>{genInfo}</div>}

        <label style={{ marginTop: 14 }}>Task definition (YAML) — editable</label>
        <textarea className="yaml" rows={16} value={yamlText} onChange={(e) => setYamlText(e.target.value)} />

        <button className="go" onClick={run} disabled={running}>
          {running ? "Running sweep…" : "Run sweep"}
        </button>
        {error && <div className="error">{error}</div>}
      </aside>

      <main className="results">
        {running && <div className="empty">Running sweep — this can take seconds to minutes…</div>}
        {!running && !result && <div className="empty">Edit the YAML (or generate it) and run a sweep.</div>}
        {result?.empty && <div className="empty">No feasible configurations met the SLA.</div>}
        {result && !result.empty && (
          <>
            <div className="sweep-head">
              <span>
                <b>{result.n_feasible}</b> feasible · <b>{result.n_pareto}</b> on Pareto front
              </span>
            </div>
            <div className="panel">
              <h3>
                Pareto: {result.y_col} vs {result.x_col}
              </h3>
              <ResponsiveContainer width="100%" height={360}>
                <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 20 }}>
                  <CartesianGrid stroke="#334155" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name={result.x_col}
                    stroke="#94a3b8"
                    fontSize={11}
                    label={{ value: result.x_col, position: "bottom", fill: "#94a3b8", fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name={result.y_col}
                    stroke="#94a3b8"
                    fontSize={11}
                    label={{ value: result.y_col, angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 11 }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
                    formatter={(v: number) => v.toFixed(2)}
                    labelFormatter={() => ""}
                    content={({ payload }: any) => {
                      if (!payload || !payload.length) return null;
                      const p = payload[0].payload;
                      return (
                        <div style={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, padding: 8, fontSize: 12 }}>
                          <div>{p.label}</div>
                          <div>{result.x_col}: {p.x.toFixed(2)}</div>
                          <div>{result.y_col}: {p.y.toFixed(2)}</div>
                        </div>
                      );
                    }}
                  />
                  <Scatter name="feasible" data={feasible} fill="#475569" />
                  <Scatter name="pareto" data={pareto} fill="#4ade80" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <h3>Pareto-optimal configurations</h3>
              <div className="table-wrap">
                <table className="results-table">
                  <thead>
                    <tr>
                      {(result.columns || []).map((c) => (
                        <th key={c}>{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(result.pareto || []).map((row, i) => (
                      <tr key={i}>
                        {(result.columns || []).map((c) => (
                          <td key={c}>
                            {typeof row[c] === "number"
                              ? Number.isInteger(row[c])
                                ? row[c]
                                : (row[c] as number).toFixed(2)
                              : String(row[c] ?? "—")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
