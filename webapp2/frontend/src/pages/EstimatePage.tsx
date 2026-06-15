import { useEffect, useMemo, useState } from "react";
import { Databases, EstimateResponse, Model, postEstimate } from "../api";
import { Field, MemoryPanel, PhaseChart, ScalarCard, SOURCE_COLORS } from "../components";

const DB_MODES = ["SILICON", "HYBRID", "EMPIRICAL", "SOL"];
const MODES = [
  { id: "static", label: "Static" },
  { id: "agg", label: "Aggregated" },
  { id: "disagg", label: "Disagg" },
];

type Worker = { tp: number; dp: number; moe_tp: number; moe_ep: number; bs: number; workers: number };

function WorkerForm({
  title,
  w,
  set,
  isMoe,
  showWorkers,
}: {
  title?: string;
  w: Worker;
  set: (w: Worker) => void;
  isMoe: boolean;
  showWorkers: boolean;
}) {
  return (
    <div className="worker">
      {title && <div className="worker-title">{title}</div>}
      <div className="row">
        <Field label="TP" v={w.tp} on={(x) => set({ ...w, tp: x })} />
        {isMoe && <Field label="MoE DP" v={w.dp} on={(x) => set({ ...w, dp: x })} />}
      </div>
      {isMoe && (
        <div className="row">
          <Field label="MoE TP" v={w.moe_tp} on={(x) => set({ ...w, moe_tp: x })} />
          <Field label="MoE EP" v={w.moe_ep} on={(x) => set({ ...w, moe_ep: x })} />
        </div>
      )}
      <div className="row">
        <Field label="Batch" v={w.bs} on={(x) => set({ ...w, bs: x })} />
        {showWorkers && <Field label="Workers" v={w.workers} on={(x) => set({ ...w, workers: x })} />}
      </div>
    </div>
  );
}

export default function EstimatePage({ dbs, models }: { dbs: Databases; models: Model[] }) {
  const [mode, setMode] = useState("static");
  const [system, setSystem] = useState("");
  const [backend, setBackend] = useState("");
  const [version, setVersion] = useState("");
  const [dbMode, setDbMode] = useState("SILICON");
  const [modelPath, setModelPath] = useState("Qwen/Qwen3-32B");
  const [isl, setIsl] = useState(1024);
  const [osl, setOsl] = useState(1024);

  const [single, setSingle] = useState<Worker>({ tp: 4, dp: 8, moe_tp: 1, moe_ep: 8, bs: 1, workers: 1 });
  const [prefill, setPrefill] = useState<Worker>({ tp: 4, dp: 1, moe_tp: 1, moe_ep: 8, bs: 1, workers: 1 });
  const [decode, setDecode] = useState<Worker>({ tp: 4, dp: 1, moe_tp: 1, moe_ep: 8, bs: 32, workers: 1 });

  const [result, setResult] = useState<EstimateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!system) setSystem(dbs["h200_sxm"] ? "h200_sxm" : Object.keys(dbs)[0] || "");
  }, [dbs]); // eslint-disable-line react-hooks/exhaustive-deps

  const isMoe = useMemo(
    () => models.find((m) => m.path === modelPath)?.is_moe ?? false,
    [models, modelPath]
  );

  const backends = useMemo(() => Object.keys(dbs[system] || {}), [dbs, system]);
  useEffect(() => {
    if (backends.length && !backends.includes(backend)) {
      setBackend(backends.includes("trtllm") ? "trtllm" : backends[0]);
    }
  }, [backends]); // eslint-disable-line react-hooks/exhaustive-deps

  const versions = useMemo(() => dbs[system]?.[backend] || [], [dbs, system, backend]);
  useEffect(() => {
    if (versions.length) setVersion(versions[versions.length - 1]);
  }, [versions]);

  function moeFields(w: Worker) {
    return isMoe
      ? { attention_dp_size: w.dp, moe_tp_size: w.moe_tp, moe_ep_size: w.moe_ep }
      : { attention_dp_size: 1, moe_tp_size: null, moe_ep_size: null };
  }

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const base = {
        mode,
        model_path: modelPath,
        system_name: system,
        backend_name: backend,
        backend_version: version || null,
        database_mode: dbMode,
        isl,
        osl,
      };
      const req =
        mode === "disagg"
          ? {
              ...base,
              prefill_tp_size: prefill.tp,
              prefill_batch_size: prefill.bs,
              prefill_num_workers: prefill.workers,
              prefill_attention_dp_size: isMoe ? prefill.dp : 1,
              prefill_moe_tp_size: isMoe ? prefill.moe_tp : null,
              prefill_moe_ep_size: isMoe ? prefill.moe_ep : null,
              decode_tp_size: decode.tp,
              decode_batch_size: decode.bs,
              decode_num_workers: decode.workers,
              decode_attention_dp_size: isMoe ? decode.dp : 1,
              decode_moe_tp_size: isMoe ? decode.moe_tp : null,
              decode_moe_ep_size: isMoe ? decode.moe_ep : null,
            }
          : { ...base, batch_size: single.bs, tp_size: single.tp, ...moeFields(single) };
      setResult(await postEstimate(req));
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="layout">
      <aside className="form">
        <div className="modes">
          {MODES.map((m) => (
            <button key={m.id} className={mode === m.id ? "mode active" : "mode"} onClick={() => setMode(m.id)}>
              {m.label}
            </button>
          ))}
        </div>

        <label>Model {isMoe && <span className="tag">MoE</span>}</label>
        <select value={modelPath} onChange={(e) => setModelPath(e.target.value)}>
          {models.map((m) => (
            <option key={m.path} value={m.path}>
              {m.is_moe ? "◆ " : ""}
              {m.path}
            </option>
          ))}
        </select>

        <label>System</label>
        <select value={system} onChange={(e) => setSystem(e.target.value)}>
          {Object.keys(dbs).map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>

        <div className="row">
          <div>
            <label>Backend</label>
            <select value={backend} onChange={(e) => setBackend(e.target.value)}>
              {backends.map((b) => (
                <option key={b}>{b}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Version</label>
            <select value={version} onChange={(e) => setVersion(e.target.value)}>
              {versions.map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </div>
        </div>

        <label>Database mode</label>
        <select value={dbMode} onChange={(e) => setDbMode(e.target.value)}>
          {DB_MODES.map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>

        <div className="row">
          <Field label="ISL" v={isl} on={setIsl} />
          <Field label="OSL" v={osl} on={setOsl} />
        </div>

        {mode === "disagg" ? (
          <>
            <WorkerForm title="Prefill worker" w={prefill} set={setPrefill} isMoe={isMoe} showWorkers />
            <WorkerForm title="Decode worker" w={decode} set={setDecode} isMoe={isMoe} showWorkers />
          </>
        ) : (
          <WorkerForm w={single} set={setSingle} isMoe={isMoe} showWorkers={false} />
        )}

        <button className="go" onClick={run} disabled={loading || !system}>
          {loading ? "Estimating…" : "Estimate"}
        </button>
        {error && <div className="error">{error}</div>}

        <div className="legend">
          {Object.entries(SOURCE_COLORS)
            .filter(([k]) => k !== "unknown")
            .map(([k, c]) => (
              <span key={k}>
                <i style={{ background: c }} /> {k}
              </span>
            ))}
        </div>
      </aside>

      <main className="results">
        {!result && !loading && <div className="empty">Configure on the left and hit Estimate.</div>}
        {result && (
          <>
            <div className="cards">
              {result.cards.map((c) => (
                <ScalarCard key={c.label} c={c} />
              ))}
            </div>
            <div className="grid">
              {result.phases.map((p) => (
                <PhaseChart key={p.name} phase={p} />
              ))}
            </div>
            {result.memory_groups.length > 0 && (
              <div className={result.memory_groups.length > 1 ? "grid" : ""}>
                {result.memory_groups.map((g) => (
                  <MemoryPanel key={g.label} group={g} />
                ))}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
