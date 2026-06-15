import { useEffect, useState } from "react";
import { ConfigStatus, fetchConfig, saveConfig } from "../api";

export default function ConfigPage() {
  const [cfg, setCfg] = useState<ConfigStatus | null>(null);
  const [provider, setProvider] = useState("anthropic");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchConfig()
      .then((c) => {
        setCfg(c);
        setProvider(c.provider);
        setModel(c.model);
        setBaseUrl(c.base_url || "");
      })
      .catch((e) => setError(String(e)));
  }, []);

  const isCustom = provider === "custom";

  const models = cfg?.provider_models[provider] || [];

  useEffect(() => {
    // When provider changes, default the model to that provider's first suggestion.
    if (models.length && !models.includes(model)) setModel(models[0]);
  }, [provider]); // eslint-disable-line react-hooks/exhaustive-deps

  async function save() {
    setSaving(true);
    setMsg(null);
    setError(null);
    try {
      const updated = await saveConfig(provider, model, apiKey || null, baseUrl);
      setCfg(updated);
      setApiKey("");
      setMsg(updated.key_set ? "Saved. API key is set." : "Saved (no API key on record yet).");
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="config-page">
      <div className="panel config-card">
        <h3>LLM provider</h3>
        <p className="hint">
          Used by the Sweep page’s “Generate YAML” feature. The API key is stored locally on the
          backend and never returned to the browser. Choose <b>custom</b> for any OpenAI-compatible
          endpoint (vLLM, SGLang, TGI, a proxy, etc.) — fill in its base URL below.
        </p>

        <label>Provider</label>
        <select value={provider} onChange={(e) => setProvider(e.target.value)}>
          {Object.keys(cfg?.provider_models || { anthropic: [], openai: [], custom: [] }).map((p) => (
            <option key={p}>{p}</option>
          ))}
        </select>

        {isCustom && (
          <>
            <label>Base URL</label>
            <input
              placeholder="https://my-host:8000/v1"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
            />
          </>
        )}

        <label>Model</label>
        <input
          list="model-list"
          placeholder={isCustom ? "served model name" : ""}
          value={model}
          onChange={(e) => setModel(e.target.value)}
        />
        <datalist id="model-list">
          {models.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>

        <label>
          API key{" "}
          {cfg && (
            <span className={cfg.key_set ? "badge ok" : "badge none"}>
              {cfg.key_set ? "set" : "not set"}
            </span>
          )}
        </label>
        <input
          type="password"
          placeholder={
            cfg?.key_set
              ? "•••••••• (leave blank to keep)"
              : isCustom
              ? "paste API key (optional for local servers)"
              : "paste API key"
          }
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
        />

        <button className="go" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>
        {msg && <div className="gen-ok">{msg}</div>}
        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
}
