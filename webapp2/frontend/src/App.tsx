import { useEffect, useState } from "react";
import { Databases, Model, fetchDatabases, fetchModels } from "./api";
import EstimatePage from "./pages/EstimatePage";
import SweepPage from "./pages/SweepPage";
import ConfigPage from "./pages/ConfigPage";

const TABS = [
  { id: "estimate", label: "Estimate" },
  { id: "sweep", label: "Sweep" },
  { id: "config", label: "Config" },
];

export default function App() {
  const [tab, setTab] = useState("estimate");
  const [dbs, setDbs] = useState<Databases>({});
  const [models, setModels] = useState<Model[]>([]);

  useEffect(() => {
    fetchDatabases().then(setDbs).catch(() => {});
    fetchModels().then(setModels).catch(() => {});
  }, []);

  return (
    <div className="app">
      <header>
        <h1>aiconfigurator</h1>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === "estimate" && <EstimatePage dbs={dbs} models={models} />}
      {tab === "sweep" && <SweepPage />}
      {tab === "config" && <ConfigPage />}
    </div>
  );
}
