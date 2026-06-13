import { ChangeEvent, DragEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type Decision = "BLOCK" | "REVIEW" | "ALLOW" | "DECLINED";

type FeatureContribution = {
  feature: string;
  suspicion_score: number;
  distance_component: number;
  value: number;
  fraud_centroid_value: number;
};

type ModelVotes = {
  rf: number;
  xgb: number;
  lgbm: number;
};

type PredictResponse = {
  status: Decision;
  reason?: string;
  confidence_score: number;
  euclidean_distance_to_fraud: number;
  feature_contributions: FeatureContribution[];
  top_suspicious_feature: string | null;
  newbalanceOrig: number;
  newbalanceDest?: number;
  type?: string;
  amount?: number;
  oldbalanceOrg?: number;
  session_id?: string;
  txn_uuid?: string;
  model_votes?: ModelVotes;
  heuristic_triggered?: string | null;
};

type LedgerEntry = {
  id: number;
  txn_uuid: string;
  session_id: string;
  timestamp: string;
  type: string;
  amount: number;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  risk_score: number;
  decision: string;
  heuristic_triggered?: string | null;
  model_votes?: ModelVotes;
};

type HealthData = {
  status: string;
  uptime: string;
  model_count: number;
  models_loaded: string[];
  database: string;
  total_requests: number;
  total_transactions_db: number;
  active_sessions: number;
};

type AlertEntry = {
  txn_uuid: string;
  timestamp: string;
  type: string;
  amount: number;
  decision: string;
  alert_type: string;
  session_id: string;
  risk_score: number;
};

type FormState = {
  type: string;
  amount: string;
  oldbalanceOrg: string;
  oldbalanceDest: string;
  session_id: string;
};

type BatchSummary = {
  total: number;
  summary: Record<string, number>;
  avg_confidence: number;
  results: PredictResponse[];
};

type ConfigState = {
  allow_threshold: number;
  block_threshold: number;
};

type TimelinePoint = { offset: number; ts: number };

const FEATURE_ALIASES: Record<string, string> = {
  amount: "Transaction Amount",
  oldbalanceOrg: "Sender Balance (Before)",
  newbalanceOrig: "Sender Balance (After)",
  oldbalanceDest: "Receiver Balance (Before)",
  newbalanceDest: "Receiver Balance (After)",
  type_encoded: "Transaction Type",
};

const TRANSACTION_TYPES = [
  { backendValue: "PAYMENT", displayLabel: "Merchant Payment (Online Purchase)" },
  { backendValue: "TRANSFER", displayLabel: "Peer-to-Peer Transfer (Digital)" },
  { backendValue: "CASH_OUT", displayLabel: "Cash Withdrawal (Digital-to-Fiat)" },
  { backendValue: "CASH_IN", displayLabel: "Cash Deposit (Fiat-to-Digital)" },
  { backendValue: "DEBIT", displayLabel: "Direct Bank Debit" },
];
const INITIAL_FORM: FormState = { type: "TRANSFER", amount: "", oldbalanceOrg: "", oldbalanceDest: "", session_id: "SESSION-99X" };
const HIGH_RISK_SAMPLE: FormState = { type: "CASH_OUT", amount: "80000", oldbalanceOrg: "80000", oldbalanceDest: "0", session_id: "SESSION-99X" };
const API = "http://localhost:8000";

function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx".replace(/[x]/g, () => Math.floor(Math.random() * 16).toString(16));
}
function randomBetween(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
function statusColor(s: Decision | string) {
  if (s === "BLOCK") return "bg-red-600 text-white";
  if (s === "REVIEW") return "bg-yellow-500 text-yellow-950";
  if (s === "DECLINED") return "bg-orange-600 text-white";
  return "bg-emerald-600 text-white";
}
function statusDot(s: Decision | string) {
  if (s === "BLOCK") return "bg-red-500";
  if (s === "REVIEW") return "bg-yellow-500";
  if (s === "DECLINED") return "bg-orange-500";
  return "bg-emerald-500";
}
function parseCSV(text: string): FormState[] {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const header = lines[0].toLowerCase().split(",").map((h) => h.trim());
  const typeIdx = header.indexOf("type");
  const amtIdx = header.indexOf("amount");
  const balOrigIdx = header.indexOf("oldbalanceorg");
  const balDestIdx = header.indexOf("oldbalancedest");
  return lines.slice(1).map((line) => {
    const cols = line.split(",").map((c) => c.trim());
    return { type: cols[typeIdx] || "PAYMENT", amount: cols[amtIdx] || "0", oldbalanceOrg: cols[balOrigIdx] || "0", oldbalanceDest: cols[balDestIdx] || "0", session_id: "SESSION-99X" };
  });
}

function AuditReceipt({ result, form, onExport }: { result: PredictResponse; form: FormState; onExport: (uuid: string) => void }) {
  const uuid = useMemo(() => result.txn_uuid || generateUUID(), [result]);
  const timestamp = useMemo(() => new Date().toLocaleString(), [result]);
  const amount = result.amount ?? Number(form.amount);
  const startBal = result.oldbalanceOrg ?? Number(form.oldbalanceOrg);
  const endBal = result.newbalanceOrig ?? startBal - amount;
  const isVelocity = result.reason === "VELOCITY_EXCEEDED";
  const badgeCls = isVelocity ? "text-red-400 border-red-500/40 bg-red-950/30"
    : result.status === "BLOCK" ? "text-red-400 border-red-500/40 bg-red-950/30"
    : result.status === "DECLINED" ? "text-orange-400 border-orange-500/40 bg-orange-950/30"
    : result.status === "REVIEW" ? "text-yellow-400 border-yellow-500/40 bg-yellow-950/30"
    : "text-emerald-400 border-emerald-500/40 bg-emerald-950/30";
  return (
    <div className="mt-5 rounded-xl border border-slate-700/60 bg-slate-950/90 p-5 font-mono text-sm backdrop-blur">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.25em] text-slate-500">Digital Audit Receipt</div>
          <div className="mt-1 text-xs text-slate-400">UUID: <span className="text-slate-200">{uuid}</span></div>
          <div className="text-xs text-slate-400">Time: <span className="text-slate-200">{timestamp}</span></div>
          <div className="text-xs text-slate-400">Session: <span className="text-slate-200">{result.session_id ?? form.session_id}</span></div>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" className="h-6 text-[10px] border-slate-600 text-slate-300 bg-slate-950 hover:bg-slate-800" onClick={() => onExport(uuid)}>
            Export JSON
          </Button>
          <Badge variant="outline" className={`text-xs font-bold ${badgeCls}`}>{isVelocity ? "RATE LIMIT" : result.status}</Badge>
        </div>
      </div>
      {result.heuristic_triggered && (
        <div className="mt-3 rounded-lg border border-yellow-500/40 bg-yellow-950/20 p-3">
          <div className="text-xs font-bold text-yellow-400">Heuristic Triggered: {result.heuristic_triggered}</div>
        </div>
      )}
      {isVelocity ? (
        <div className="mt-4 rounded-lg border border-red-500/40 bg-red-950/30 p-4">
          <div className="text-sm font-bold text-red-400 uppercase tracking-wider">RATE LIMIT / BOT DETECTED</div>
          <p className="mt-2 text-xs text-red-300/80">Session <span className="font-mono text-red-200">{result.session_id}</span> exceeded 3 requests in 10s.</p>
        </div>
      ) : (
        <>
          <div className="mt-4 border-t border-dashed border-slate-700/60 pt-3">
            <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-2">Balance Ledger</div>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between"><span className="text-slate-400">Start Balance</span><span className="text-slate-200 tabular-nums">${startBal.toLocaleString("en-US", { minimumFractionDigits: 2 })}</span></div>
              <div className="flex justify-between"><span className="text-slate-400">&minus; Amount</span><span className="text-rose-400 tabular-nums">&minus;${amount.toLocaleString("en-US", { minimumFractionDigits: 2 })}</span></div>
              <div className="flex justify-between border-t border-slate-700/40 pt-1 font-semibold"><span className="text-slate-300">Ending Balance</span><span className="text-slate-100 tabular-nums">${endBal.toLocaleString("en-US", { minimumFractionDigits: 2 })}</span></div>
            </div>
          </div>
          <div className="mt-3 border-t border-dashed border-slate-700/60 pt-3 text-[11px] text-slate-500">
            {result.status === "DECLINED" ? `Reason: ${result.reason}` : `Fraud Probability: ${result.confidence_score.toFixed(2)}% — Distance: ${result.euclidean_distance_to_fraud.toFixed(4)}`}
          </div>
        </>
      )}
    </div>
  );
}

function RiskDistributionBar({ summary }: { summary: Record<string, number> }) {
  const total = Object.values(summary).reduce((a, b) => a + b, 0);
  if (total === 0) return null;
  const segs = [
    { key: "ALLOW", color: "bg-emerald-500", label: "Allow" },
    { key: "REVIEW", color: "bg-yellow-500", label: "Review" },
    { key: "BLOCK", color: "bg-red-500", label: "Block" },
    { key: "DECLINED", color: "bg-orange-500", label: "Declined" },
  ];
  return (
    <div className="mt-4">
      <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-2">Risk Distribution</div>
      <div className="flex h-6 w-full overflow-hidden rounded-md">
        {segs.map((s) => { const c = summary[s.key] ?? 0; const p = (c / total) * 100; return p > 0 ? <div key={s.key} className={`${s.color} flex items-center justify-center text-[10px] font-bold text-white`} style={{ width: `${p}%` }} title={`${s.label}: ${c}`}>{p > 8 ? c : ""}</div> : null; })}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-3 text-[10px] text-slate-400">
        {segs.map((s) => <span key={s.key} className="flex items-center gap-1"><span className={`inline-block h-2 w-2 rounded-full ${s.color}`} />{s.label}: {summary[s.key] ?? 0}</span>)}
      </div>
    </div>
  );
}

function SessionTimeline({ points }: { points: TimelinePoint[] }) {
  if (points.length < 2) return null;
  const data = points.map((p, i) => ({ idx: i + 1, seconds_ago: p.offset }));
  return (
    <div className="mt-4">
      <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-2">Session Activity ({points.length} req/min)</div>
      <div className="h-[80px] w-full rounded-lg border border-slate-700 bg-slate-950/60 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="idx" tick={{ fill: "#64748b", fontSize: 9 }} />
            <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
            <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: 11 }} formatter={(v: number) => [`${v.toFixed(1)}s ago`, "Time"]} />
            <Line type="monotone" dataKey="seconds_ago" stroke="#22d3ee" strokeWidth={2} dot={{ fill: "#0891b2", r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function DivergenceChart({ contributions }: { contributions: FeatureContribution[] }) {
  const data = useMemo(() => contributions.map((c) => {
    const d = c.distance_component;
    const normalized = Math.min((d / 3.0) * 100, 100);
    return {
      feature: FEATURE_ALIASES[c.feature] || c.feature,
      divergence: Number(d.toFixed(4)),
      fill: normalized > 60 ? "#22c55e" : normalized > 30 ? "#eab308" : "#ef4444",
    };
  }), [contributions]);
  if (data.length === 0) return null;
  return (
    <div className="mt-5">
      <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-2">Feature Divergence from Fraud Pattern</div>
      <div className="h-[200px] w-full rounded-lg border border-slate-700 bg-slate-950/60 p-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
            <CartesianGrid stroke="#1e293b" horizontal={false} />
            <XAxis type="number" domain={[0, 3]} tick={{ fill: "#64748b", fontSize: 10 }} />
            <YAxis dataKey="feature" type="category" width={110} tick={{ fill: "#cbd5e1", fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: 11 }} formatter={(v: number) => [`${v.toFixed(4)}`, "Distance"]} />
            <Bar dataKey="divergence" radius={[0, 4, 4, 0]}>{data.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1.5 flex gap-4 text-[10px] text-slate-400">
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-red-500" />Risky (Close to fraud pattern)</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-yellow-500" />Moderate</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-emerald-500" />Safe (Far from fraud pattern)</span>
      </div>
    </div>
  );
}

function SystemHealth({ health }: { health: HealthData | null }) {
  if (!health) return null;
  const items = [
    { label: "Status", value: health.status, color: "text-emerald-400" },
    { label: "Uptime", value: health.uptime, color: "text-slate-100" },
    { label: "Models", value: `${health.model_count} loaded`, color: "text-cyan-400" },
    { label: "Database", value: health.database, color: "text-emerald-400" },
    { label: "Requests", value: health.total_requests.toLocaleString(), color: "text-slate-100" },
    { label: "DB Records", value: health.total_transactions_db.toLocaleString(), color: "text-slate-100" },
    { label: "Sessions", value: health.active_sessions.toString(), color: "text-cyan-400" },
  ];
  return (
    <Card className="border-slate-800 bg-slate-900/80 p-4 backdrop-blur">
      <h2 className="text-sm font-semibold text-cyan-100 mb-3">System Health</h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {items.map((it) => (
          <div key={it.label} className="rounded-lg border border-slate-700 bg-slate-950/60 p-2 text-center">
            <div className="text-[9px] uppercase tracking-wider text-slate-500">{it.label}</div>
            <div className={`mt-0.5 text-sm font-semibold tabular-nums ${it.color}`}>{it.value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function MLMetricsDashboard({ metrics }: { metrics: any | null }) {
  const [tab, setTab] = useState<"confusion" | "roc" | "precision">("confusion");
  const rocData = useMemo(() => {
    if (!metrics) return [];
    const models = ["rf", "xgb", "lgbm"] as const;
    const colors = { rf: "#22d3ee", xgb: "#a78bfa", lgbm: "#f472b6" };
    const result: { name: string; color: string; data: { fpr: number; tpr: number }[] }[] = [];
    for (const m of models) {
      const curve = metrics.models[m].roc_curve;
      result.push({ name: metrics.models[m].name, color: colors[m], data: curve.fpr.map((f, i) => ({ fpr: f, tpr: curve.tpr[i] })) });
    }
    const ensCurve = metrics.ensemble.roc_curve;
    result.push({ name: "Ensemble (Max Vote)", color: "#22c55e", data: ensCurve.fpr.map((f, i) => ({ fpr: f, tpr: ensCurve.tpr[i] })) });
    return result;
  }, [metrics]);
  if (!metrics) return null;
  const ens = metrics.ensemble;
  const cm = ens.confusion_matrix;
  const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];

  return (
    <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-cyan-100">ML Metrics Dashboard</h2>
        <div className="flex gap-1">
          {(["confusion", "roc", "precision"] as const).map((t) => (
            <Button key={t} size="sm" variant={tab === t ? "default" : "outline"}
              className={`h-6 text-[10px] ${tab === t ? "bg-cyan-600 text-white" : "border-slate-700 text-slate-300 bg-slate-950 hover:bg-slate-800"}`}
              onClick={() => setTab(t)}>
              {t === "confusion" ? "Confusion Matrix" : t === "roc" ? "ROC Curves" : "Precision/Recall"}
            </Button>
          ))}
        </div>
      </div>

      {tab === "confusion" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="text-xs text-slate-400 mb-2">Ensemble Confusion Matrix (Test: {metrics.test_size.toLocaleString()} samples)</h3>
            <div className="rounded-lg border border-slate-700 bg-slate-950/60 p-4">
              <table className="w-full text-xs">
                <thead><tr><th className="px-2 py-1 text-slate-500" /><th className="px-2 py-1 text-center text-slate-400">Pred Legit</th><th className="px-2 py-1 text-center text-slate-400">Pred Fraud</th></tr></thead>
                <tbody>
                  <tr><td className="px-2 py-1 text-slate-400 font-medium">Actual Legit</td><td className="px-2 py-1 text-center text-emerald-400 font-mono">{tn.toLocaleString()}</td><td className="px-2 py-1 text-center text-yellow-400 font-mono">{fp.toLocaleString()}</td></tr>
                  <tr><td className="px-2 py-1 text-slate-400 font-medium">Actual Fraud</td><td className="px-2 py-1 text-center text-orange-400 font-mono">{fn.toLocaleString()}</td><td className="px-2 py-1 text-center text-red-400 font-mono">{tp.toLocaleString()}</td></tr>
                </tbody>
              </table>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2 text-[10px]">
              <div className="text-center"><span className="text-slate-500">Accuracy</span><div className="text-slate-100 font-semibold">{((tn + tp) / (tn + fp + fn + tp) * 100).toFixed(2)}%</div></div>
              <div className="text-center"><span className="text-slate-500">Fraud Caught</span><div className="text-red-400 font-semibold">{tp.toLocaleString()}/{(tp + fn).toLocaleString()}</div></div>
              <div className="text-center"><span className="text-slate-500">False Alarms</span><div className="text-yellow-400 font-semibold">{fp.toLocaleString()}</div></div>
            </div>
          </div>
          <div>
            <h3 className="text-xs text-slate-400 mb-2">Per-Model ROC AUC</h3>
            <div className="space-y-2">
              {Object.entries(metrics.models).map(([key, m]) => (
                <div key={key} className="flex items-center gap-3 rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2">
                  <span className="text-xs text-slate-300 w-20">{m.name}</span>
                  <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-cyan-500 rounded-full" style={{ width: `${m.roc_auc * 100}%` }} />
                  </div>
                  <span className="text-xs font-mono text-cyan-400 w-14 text-right">{m.roc_auc}</span>
                </div>
              ))}
              <div className="flex items-center gap-3 rounded-lg border border-emerald-700/40 bg-emerald-950/20 px-3 py-2">
                <span className="text-xs text-emerald-300 w-20 font-semibold">Ensemble</span>
                <div className="flex-1 h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${ens.roc_auc * 100}%` }} />
                </div>
                <span className="text-xs font-mono text-emerald-400 w-14 text-right font-bold">{ens.roc_auc}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === "roc" && (
        <div>
          <h3 className="text-xs text-slate-400 mb-2">ROC Curves — All Models vs Ensemble</h3>
          <div className="h-[300px] w-full rounded-lg border border-slate-700 bg-slate-950/60 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 10 }} label={{ value: "False Positive Rate", position: "bottom", fill: "#64748b", fontSize: 10, offset: -5 }} />
                <YAxis type="number" dataKey="tpr" domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 10 }} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: 11 }} />
                {rocData.map((r) => (
                  <Line key={r.name} data={r.data} type="monotone" dataKey="tpr" stroke={r.color} strokeWidth={2} dot={false} name={r.name} />
                ))}
                <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} type="monotone" dataKey="tpr" stroke="#475569" strokeWidth={1} strokeDasharray="5 5" dot={false} name="Random" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex flex-wrap gap-3 text-[10px]">
            {rocData.map((r) => <span key={r.name} className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: r.color }} />{r.name} (AUC: {r.name === "Ensemble (Max Vote)" ? ens.roc_auc : metrics.models[r.name === "RandomForest" ? "rf" : r.name === "XGBoost" ? "xgb" : "lgbm"]?.roc_auc})</span>)}
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-slate-500" />Random baseline</span>
          </div>
        </div>
      )}

      {tab === "precision" && (
        <div>
          <h3 className="text-xs text-slate-400 mb-3">Precision / Recall / F1 — Fraud Class</h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[...Object.entries(metrics.models).map(([, m]) => ({ name: m.name, ...m.metrics.fraud })), { name: "Ensemble (Max Vote)", ...ens.metrics.fraud }].map((m) => (
              <div key={m.name} className={`rounded-lg border p-3 ${m.name === "Ensemble (Max Vote)" ? "border-emerald-700/40 bg-emerald-950/20" : "border-slate-700 bg-slate-950/60"}`}>
                <div className={`text-xs font-semibold mb-2 ${m.name === "Ensemble (Max Vote)" ? "text-emerald-300" : "text-slate-300"}`}>{m.name}</div>
                <div className="space-y-1.5 text-[11px]">
                  <div className="flex justify-between"><span className="text-slate-500">Precision</span><span className="text-slate-100 font-mono">{(m.precision * 100).toFixed(2)}%</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">Recall</span><span className="text-slate-100 font-mono">{(m.recall * 100).toFixed(2)}%</span></div>
                  <div className="flex justify-between"><span className="text-slate-500">F1</span><span className="text-slate-100 font-mono">{(m.f1 * 100).toFixed(2)}%</span></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

function ModelAgreement({ items }: { items: any[] }) {
  if (items.length === 0) return null;
  return (
    <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
      <h2 className="text-lg font-semibold text-cyan-100 mb-3">Model Agreement</h2>
      <div className="overflow-auto rounded-lg border border-slate-700">
        <table className="w-full text-[11px]">
          <thead className="bg-slate-900">
            <tr className="text-left text-slate-400 border-b border-slate-700">
              <th className="px-3 py-2">Type</th>
              <th className="px-3 py-2 text-right">Amount</th>
              <th className="px-3 py-2 text-right">RF</th>
              <th className="px-3 py-2 text-right">XGB</th>
              <th className="px-3 py-2 text-right">LGBM</th>
              <th className="px-3 py-2">Decision</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => (
              <tr key={i} className="border-t border-slate-800/50 hover:bg-slate-800/30">
                <td className="px-3 py-2 text-slate-300">{it.type}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-100">${it.amount.toLocaleString()}</td>
                <td className="px-3 py-2 text-right tabular-nums"><span className={it.rf > 50 ? "text-red-400" : "text-emerald-400"}>{it.rf.toFixed(1)}%</span></td>
                <td className="px-3 py-2 text-right tabular-nums"><span className={it.xgb > 50 ? "text-red-400" : "text-emerald-400"}>{it.xgb.toFixed(1)}%</span></td>
                <td className="px-3 py-2 text-right tabular-nums"><span className={it.lgbm > 50 ? "text-red-400" : "text-emerald-400"}>{it.lgbm.toFixed(1)}%</span></td>
                <td className="px-3 py-2"><span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-bold ${statusColor(it.decision)}`}>{it.decision}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function FeatureImportance({ importance }: { importance: Record<string, number> }) {
  const data = useMemo(() => {
    return Object.entries(importance).map(([feature, imp]) => ({
      feature: FEATURE_ALIASES[feature] || feature,
      importance: Number((imp * 100).toFixed(1)),
      fill: imp > 0.25 ? "#22d3ee" : imp > 0.1 ? "#a78bfa" : "#64748b",
    })).sort((a, b) => b.importance - a.importance);
  }, [importance]);
  return (
    <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
      <h2 className="text-lg font-semibold text-cyan-100 mb-3">Feature Importance (RandomForest)</h2>
      <div className="h-[200px] w-full rounded-lg border border-slate-700 bg-slate-950/60 p-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
            <CartesianGrid stroke="#1e293b" horizontal={false} />
            <XAxis type="number" domain={[0, 50]} tick={{ fill: "#64748b", fontSize: 10 }} label={{ value: "Importance %", position: "bottom", fill: "#64748b", fontSize: 10, offset: -5 }} />
            <YAxis dataKey="feature" type="category" width={110} tick={{ fill: "#cbd5e1", fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: 11 }} formatter={(v: number) => [`${v.toFixed(1)}%`, "Importance"]} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>{data.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

function AlertFeed({ alerts }: { alerts: AlertEntry[] }) {
  return (
    <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-cyan-100">Alert Feed</h2>
        <Badge variant="outline" className={`text-[10px] ${alerts.length > 0 ? "border-red-700/50 text-red-400" : "border-slate-700 text-slate-500"}`}>
          {alerts.length} alerts
        </Badge>
      </div>
      {alerts.length === 0 ? (
        <p className="text-xs text-slate-400">No alerts yet. High-risk events will appear here.</p>
      ) : (
        <div className="max-h-[250px] overflow-auto space-y-2">
          {alerts.map((a, i) => {
            const isVelocity = a.alert_type === "VELOCITY";
            return (
              <div key={i} className={`flex items-center gap-3 rounded-lg border px-3 py-2 text-xs ${isVelocity ? "border-red-500/30 bg-red-950/20" : "border-orange-500/30 bg-orange-950/20"}`}>
                <span className={`h-2 w-2 rounded-full shrink-0 ${isVelocity ? "bg-red-500 animate-pulse" : "bg-orange-500"}`} />
                <div className="flex-1 min-w-0">
                  <span className="font-semibold text-slate-100">{a.alert_type}</span>
                  <span className="text-slate-500 mx-1">&middot;</span>
                  <span className="text-slate-400">{a.type} ${a.amount.toLocaleString()}</span>
                </div>
                <span className="text-slate-500 shrink-0">{new Date(a.timestamp).toLocaleTimeString()}</span>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function App() {
  const [form, setForm] = useState<FormState>(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [stressLoading, setStressLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");
  const [stressSummary, setStressSummary] = useState("");
  const [liveTime, setLiveTime] = useState(new Date().toLocaleString());
  const [copiedUuid, setCopiedUuid] = useState("");

  const [ledger, setLedger] = useState<LedgerEntry[]>([]);
  const [ledgerLoading, setLedgerLoading] = useState(false);
  const [config, setConfig] = useState<ConfigState>({ allow_threshold: 0.35, block_threshold: 0.70 });
  const [configMsg, setConfigMsg] = useState("");
  const [batchResult, setBatchResult] = useState<BatchSummary | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);

  const [health, setHealth] = useState<HealthData | null>(null);
  const [metrics, setMetrics] = useState<any | null>(null);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [agreement, setAgreement] = useState<any[]>([]);
  const [featureImportance, setFeatureImportance] = useState<Record<string, number>>({});

  const fileRef = useRef<HTMLInputElement>(null);

  const handleInput = (field: keyof FormState) => (e: ChangeEvent<HTMLInputElement>) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const fetchLedger = useCallback(async () => {
    setLedgerLoading(true);
    try {
      const res = await fetch(`${API}/api/v1/history?limit=50`);
      if (res.ok) { const d = await res.json(); setLedger(d.items); }
    } catch { /* */ }
    setTimeout(() => setLedgerLoading(false), 400);
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/health`);
      if (res.ok) setHealth(await res.json());
    } catch { /* */ }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/health`);
      if (res.ok) {
        const d = await res.json();
        const mRes = await fetch(`${API}/api/v1/metrics`);
        if (mRes.ok) {
          const md = await mRes.json();
          setMetrics(md.training_metrics);
        }
      }
    } catch { /* */ }
  }, []);

  const fetchFeatureImportance = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/feature-importance`);
      if (res.ok) { const d = await res.json(); const map: Record<string, number> = {}; d.features.forEach((f: { feature: string; importance: number }) => { map[f.feature] = f.importance; }); setFeatureImportance(map); }
    } catch { /* */ }
  }, []);

  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/alerts?limit=20`);
      if (res.ok) { const d = await res.json(); setAlerts(d.items); }
    } catch { /* */ }
  }, []);

  const fetchAgreement = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/model-agreement?limit=10`);
      if (res.ok) { const d = await res.json(); setAgreement(d.items); }
    } catch { /* */ }
  }, []);

  const fetchTimeline = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/session/${form.session_id}/timeline`);
      if (res.ok) { const d = await res.json(); setTimeline(d.points); }
    } catch { /* */ }
  }, [form.session_id]);

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/v1/config`);
      if (res.ok) { const d = await res.json(); setConfig({ allow_threshold: d.allow_threshold, block_threshold: d.block_threshold }); }
    } catch { /* */ }
  }, []);

  const fetchAll = useCallback(() => {
    fetchLedger();
    fetchHealth();
    fetchMetrics();
    fetchFeatureImportance();
    fetchAlerts();
    fetchAgreement();
    fetchTimeline();
  }, [fetchLedger, fetchHealth, fetchMetrics, fetchFeatureImportance, fetchAlerts, fetchAgreement, fetchTimeline]);

  useEffect(() => { fetchConfig(); fetchAll(); }, [fetchConfig, fetchAll]);

  // Live clock tick
  useEffect(() => { const id = setInterval(() => setLiveTime(new Date().toLocaleString()), 1000); return () => clearInterval(id); }, []);

  // Auto-refresh ledger + alerts every 15 seconds
  useEffect(() => { const id = setInterval(() => { fetchLedger(); fetchAlerts(); fetchAgreement(); }, 15000); return () => clearInterval(id); }, [fetchLedger, fetchAlerts, fetchAgreement]);

  const onExport = useCallback(async (uuid: string) => {
    try {
      const res = await fetch(`${API}/api/v1/export/${uuid}`);
      if (res.ok) {
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `fraud-report-${uuid.slice(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch { /* */ }
  }, []);

  const runPredictRequest = async (payload: { type: string; amount: number; oldbalanceOrg: number; oldbalanceDest: number; session_id: string }) => {
    const response = await fetch(`${API}/api/v1/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error((await response.text()) || "Predict request failed.");
    return (await response.json()) as PredictResponse;
  };

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true); setError(""); setStressSummary("");
    try {
      const data = await runPredictRequest({ type: form.type, amount: Number(form.amount), oldbalanceOrg: Number(form.oldbalanceOrg), oldbalanceDest: Number(form.oldbalanceDest), session_id: form.session_id });
      setResult(data);
      fetchAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
      setResult(null);
    } finally { setLoading(false); }
  };

  const onGenerateHighRisk = () => { setForm(HIGH_RISK_SAMPLE); setError(""); setStressSummary(""); setResult(null); };

  const onStressTest = async () => {
    setStressLoading(true); setError(""); setStressSummary("");
    const backendValues = TRANSACTION_TYPES.map((t) => t.backendValue);
    const jobs = Array.from({ length: 50 }, (_, i) => runPredictRequest({ type: backendValues[randomBetween(0, backendValues.length - 1)], amount: randomBetween(500, 150000), oldbalanceOrg: randomBetween(0, 200000), oldbalanceDest: randomBetween(0, 100000), session_id: `STRESS-USER-${i + 1}` }));
    const startedAt = performance.now();
    try {
      const settled = await Promise.allSettled(jobs);
      const success = settled.filter((x) => x.status === "fulfilled").length;
      const failed = settled.length - success;
      const avgRisk = settled.filter((x): x is PromiseFulfilledResult<PredictResponse> => x.status === "fulfilled").reduce((acc, x) => acc + x.value.confidence_score, 0) / Math.max(success, 1);
      setStressSummary(`Completed ${settled.length} requests in ${(performance.now() - startedAt).toFixed(0)}ms | success: ${success} | failed: ${failed} | avg risk: ${avgRisk.toFixed(2)}%`);
      fetchAll();
    } catch (err) { setError(err instanceof Error ? err.message : "Stress test failed"); }
    finally { setStressLoading(false); }
  };

  const onConfigUpdate = async () => {
    setConfigMsg("");
    try {
      const res = await fetch(`${API}/api/v1/config`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(config) });
      if (!res.ok) { setConfigMsg((await res.json()).detail || "Failed"); return; }
      setConfigMsg("Thresholds updated");
      setTimeout(() => setConfigMsg(""), 2000);
    } catch { setConfigMsg("Connection error"); }
  };

  const onBatchUpload = async (file: File) => {
    setBatchLoading(true); setError(""); setBatchResult(null);
    try {
      const text = await file.text();
      const txns = parseCSV(text);
      if (txns.length === 0) { setError("CSV must have: type, amount, oldbalanceOrg, oldbalanceDest"); setBatchLoading(false); return; }
      const res = await fetch(`${API}/api/v1/batch`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ transactions: txns }) });
      if (!res.ok) throw new Error("Batch failed");
      setBatchResult(await res.json());
      fetchAll();
    } catch (err) { setError(err instanceof Error ? err.message : "Batch failed"); }
    finally { setBatchLoading(false); }
  };

  const onFileDrop = (e: DragEvent) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f?.name.endsWith(".csv")) onBatchUpload(f); };
  const onFileSelect = (e: ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) onBatchUpload(f); };

  const statusBadgeClass = useMemo(() => {
    if (!result) return "";
    if (result.reason === "VELOCITY_EXCEEDED") return "bg-red-700 text-white animate-pulse";
    return statusColor(result.status);
  }, [result]);

  const radarData = useMemo(() => (result?.feature_contributions ?? []).map((item) => ({ feature: FEATURE_ALIASES[item.feature] || item.feature, suspicion: Number(item.suspicion_score.toFixed(2)) })), [result]);

  return (
    <main className="min-h-screen bg-slate-950 bg-[radial-gradient(circle_at_top,_rgba(14,165,233,0.15),_transparent_55%),radial-gradient(circle_at_bottom,_rgba(20,184,166,0.12),_transparent_50%)] px-4 py-8 text-slate-100 sm:px-6 lg:px-8 print:hidden">
      <div className="mx-auto w-full max-w-7xl space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-cyan-100">Enterprise Fraud Engine</h1>
            <p className="mt-1 text-sm text-slate-400">Max-Vote Ensemble (RF + XGBoost + LightGBM) with SQLite persistence &amp; XAI tracing</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs text-slate-400">{health ? `${health.model_count} Models Loaded` : "Loading..."}</span>
            <span className="text-[10px] text-slate-600 border-l border-slate-700 pl-3 font-mono">{liveTime}</span>
          </div>
        </div>

        <SystemHealth health={health} />

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="space-y-6">
            <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
              <h2 className="text-xl font-semibold text-cyan-100">Transaction Analysis</h2>
              <form className="mt-4 space-y-4" onSubmit={onSubmit}>
                <div className="space-y-2">
                  <Label className="text-slate-100">Session ID / Device Hash</Label>
                  <Input type="text" required value={form.session_id} onChange={handleInput("session_id")} className="border-slate-700 bg-slate-950 text-slate-100" placeholder="SESSION-99X" />
                </div>
                <div className="space-y-2">
                  <Label className="text-slate-100">Transaction Type</Label>
                  <Select value={form.type} onValueChange={(v) => setForm((p) => ({ ...p, type: v }))}>
                    <SelectTrigger className="border-slate-700 bg-slate-950 text-slate-100"><SelectValue placeholder="Select type" /></SelectTrigger>
                    <SelectContent>
                      {TRANSACTION_TYPES.map((t) => (
                        <SelectItem key={t.backendValue} value={t.backendValue}>{t.displayLabel}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2"><Label className="text-slate-100">Amount</Label><Input type="number" min="0" step="0.01" required value={form.amount} onChange={handleInput("amount")} className="border-slate-700 bg-slate-950 text-slate-100" placeholder="80000" /></div>
                <div className="space-y-2"><Label className="text-slate-100">Sender Starting Balance</Label><Input type="number" min="0" step="0.01" required value={form.oldbalanceOrg} onChange={handleInput("oldbalanceOrg")} className="border-slate-700 bg-slate-950 text-slate-100" placeholder="80000" /></div>
                <div className="space-y-2"><Label className="text-slate-100">Receiver Starting Balance</Label><Input type="number" min="0" step="0.01" required value={form.oldbalanceDest} onChange={handleInput("oldbalanceDest")} className="border-slate-700 bg-slate-950 text-slate-100" placeholder="0" /></div>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Button type="submit" disabled={loading || stressLoading} className="bg-cyan-600 text-white hover:bg-cyan-500 font-semibold">{loading ? "Analyzing..." : "Analyze Transaction"}</Button>
                  <Button type="button" onClick={onGenerateHighRisk} disabled={loading || stressLoading} variant="outline" className="border-rose-700/60 text-rose-300 hover:bg-rose-950/50 font-semibold">Generate High-Risk Sample</Button>
                </div>
                <Button type="button" onClick={onStressTest} disabled={loading || stressLoading} className="w-full bg-slate-700 text-white hover:bg-slate-600 font-semibold">{stressLoading ? "Running 50 Concurrent Requests..." : "Concurrency Stress Test (50 Requests)"}</Button>
              </form>
              {stressSummary && <p className="mt-4 rounded-md border border-slate-700 bg-slate-950/80 px-3 py-2 text-xs text-slate-300">{stressSummary}</p>}
              {error && <p className="mt-4 text-sm text-red-400">{error}</p>}
            </Card>

            <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
              <h2 className="text-lg font-semibold text-cyan-100">Threshold Configurator</h2>
              <div className="mt-4 space-y-4">
                <div>
                  <div className="flex justify-between text-xs text-slate-300"><span>ALLOW_THRESHOLD</span><span className="font-mono text-emerald-400">{config.allow_threshold.toFixed(2)}</span></div>
                  <input type="range" min="0" max="1" step="0.05" value={config.allow_threshold} onChange={(e) => setConfig((p) => ({ ...p, allow_threshold: Number(e.target.value) }))} className="mt-1 w-full accent-emerald-500" />
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-300"><span>BLOCK_THRESHOLD</span><span className="font-mono text-red-400">{config.block_threshold.toFixed(2)}</span></div>
                  <input type="range" min="0" max="1" step="0.05" value={config.block_threshold} onChange={(e) => setConfig((p) => ({ ...p, block_threshold: Number(e.target.value) }))} className="mt-1 w-full accent-red-500" />
                </div>
                <Button onClick={onConfigUpdate} className="w-full bg-cyan-600 text-slate-950 hover:bg-cyan-500 font-bold">Apply Thresholds</Button>
                {configMsg && <p className="text-xs text-center text-cyan-400">{configMsg}</p>}
              </div>
            </Card>

            <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
              <h2 className="text-lg font-semibold text-cyan-100">Batch CSV Upload</h2>
              <div onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={onFileDrop} onClick={() => fileRef.current?.click()} className={`mt-4 flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors ${dragOver ? "border-cyan-400 bg-cyan-950/20" : "border-slate-700 bg-slate-950/50 hover:border-slate-500"}`}>
                <svg className="mb-2 h-8 w-8 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 16V4m0 0L8 8m4-4l4 4" /></svg>
                <span className="text-xs text-slate-400">{batchLoading ? "Processing..." : "Click or drop CSV here"}</span>
              </div>
              <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={onFileSelect} />
              {batchResult && (<><div className="mt-3 text-xs text-slate-300">Processed <span className="font-semibold text-slate-100">{batchResult.total}</span> transactions &mdash; Avg risk: <span className="font-semibold text-slate-100">{batchResult.avg_confidence.toFixed(2)}%</span></div><RiskDistributionBar summary={batchResult.summary} /></>)}
            </Card>
          </div>

          <div className="space-y-6">
            <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
              <h2 className="text-xl font-semibold text-cyan-100">XAI Visualizer</h2>
              {!result ? (
                <p className="mt-3 text-sm text-slate-300">Run an analysis to visualize feature-level anomaly contributions.</p>
              ) : (
                <>
                  <div className="mt-4 flex flex-wrap items-center gap-4">
                    <Badge className={`px-4 py-1.5 text-base font-bold tracking-wide ${statusBadgeClass}`}>{result.reason === "VELOCITY_EXCEEDED" ? "BLOCKED" : result.status}</Badge>
                    <div className="text-sm text-slate-300">
                      {result.reason === "VELOCITY_EXCEEDED" ? <span className="text-red-400 font-semibold">Rate Limit Exceeded</span> : <>Confidence: <span className="font-semibold text-white">{result.confidence_score.toFixed(2)}%</span></>}
                    </div>
                    {result.txn_uuid && <span className="text-[10px] text-slate-500 font-mono flex items-center gap-1">
                      UUID: {result.txn_uuid}
                      <button onClick={() => { navigator.clipboard.writeText(result.txn_uuid!); setCopiedUuid(result.txn_uuid!); setTimeout(() => setCopiedUuid(""), 1500); }} className="text-slate-600 hover:text-cyan-400 transition-colors" title="Copy UUID">
                        {copiedUuid === result.txn_uuid ? (
                          <svg className="h-3 w-3 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                        ) : (
                          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                        )}
                      </button>
                    </span>}
                  </div>
                  {(() => {
                    const h = result.heuristic_triggered;
                    const reasons: Record<string, string> = {
                      ALLOW: "Transaction cleared: No anomalous patterns detected.",
                      REVIEW: "Transaction flagged: Requires manual review.",
                      BLOCK: "Transaction halted: High-confidence anomaly detected by ML ensemble.",
                    };
                    const display =
                      h && !h.startsWith("System Default:")
                        ? h
                        : result.reason === "VELOCITY_EXCEEDED"
                          ? "Session blocked: Rate limit exceeded."
                          : reasons[result.status] || "Transaction processed.";
                    return <p className="mt-2 text-sm text-slate-300">{display}</p>;
                  })()}

                  {result.model_votes && (
                    <div className="mt-4 flex gap-2 flex-wrap">
                      {(["rf", "xgb", "lgbm"] as const).map((m) => (
                        <span key={m} className="rounded-md border border-slate-700 bg-slate-950/60 px-2 py-1 text-[10px]">
                          <span className="text-slate-500 uppercase">{m}:</span>{" "}
                          <span className={result.model_votes![m] > 50 ? "text-red-400 font-semibold" : "text-emerald-400 font-semibold"}>{result.model_votes![m].toFixed(1)}%</span>
                        </span>
                      ))}
                    </div>
                  )}

                  {result.reason === "VELOCITY_EXCEEDED" ? (
                    <div className="mt-5 rounded-lg border border-red-500/40 bg-red-950/30 p-4">
                      <div className="text-xs uppercase tracking-wider text-red-400 font-bold">RATE LIMIT / BOT DETECTED</div>
                      <p className="mt-2 text-sm text-red-300/80">Session <span className="font-mono text-red-200">{result.session_id}</span> triggered the velocity firewall.</p>
                    </div>
                  ) : result.status === "DECLINED" ? (
                    <div className="mt-5 rounded-lg border border-orange-500/30 bg-orange-950/20 p-4">
                      <div className="text-xs uppercase tracking-wider text-orange-400">Blocked &mdash; {result.reason}</div>
                      <p className="mt-2 text-sm text-slate-300">Transaction declined: amount exceeds sender balance.</p>
                    </div>
                  ) : (
                    <>
                      <div className="mt-5 rounded-lg border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-xs uppercase tracking-wider text-slate-400">Anomaly Distance</div>
                        <div className="mt-1 text-3xl font-bold text-cyan-300 tabular-nums">{result.euclidean_distance_to_fraud.toFixed(4)}</div>
                        <div className="mt-1 text-xs text-slate-400">Euclidean distance to fraud centroid.</div>
                      </div>
                      <div className="mt-5 h-[320px] w-full rounded-lg border border-slate-700 bg-slate-950/60 p-3">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart data={radarData} outerRadius="75%">
                            <PolarGrid stroke="#334155" />
                            <PolarAngleAxis dataKey="feature" tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                            <Tooltip formatter={(value: number) => [`${value.toFixed(2)}%`, "Suspicion"]} contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }} />
                            <Radar name="Suspicion" dataKey="suspicion" stroke="#22d3ee" fill="#0891b2" fillOpacity={0.45} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                      <DivergenceChart contributions={result.feature_contributions} />
                      <div className="mt-4 text-sm text-slate-300">Top suspicious: <span className="font-semibold text-cyan-200">{(FEATURE_ALIASES[result.top_suspicious_feature ?? ""] || result.top_suspicious_feature) ?? "n/a"}</span></div>
                      <div className="mt-2 text-sm text-slate-300">Ending Balance: <span className="font-semibold text-white tabular-nums">${result.newbalanceOrig.toLocaleString("en-US", { minimumFractionDigits: 2 })}</span></div>
                    </>
                  )}
                  <AuditReceipt result={result} form={form} onExport={onExport} />
                  <div className="mt-4 flex justify-end">
                    <Button onClick={() => window.print()} variant="outline" size="sm" className="border-slate-600 text-slate-300 bg-slate-950 hover:bg-slate-800 text-xs">
                      Print Official Receipt
                    </Button>
                  </div>
                </>
              )}
            </Card>

            <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
              <h2 className="text-lg font-semibold text-cyan-100">Session Timeline</h2>
              <SessionTimeline points={timeline} />
              {timeline.length < 2 && <p className="mt-2 text-xs text-slate-500">Send 2+ requests to see timeline.</p>}
            </Card>
          </div>
        </div>

        <MLMetricsDashboard metrics={metrics} />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ModelAgreement items={agreement} />
          <FeatureImportance importance={featureImportance} />
        </div>
        <AlertFeed alerts={alerts} />

        {ledger.length > 0 && (
          <Card className="border-slate-800 bg-slate-900/80 p-6 backdrop-blur">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-lg font-semibold text-cyan-100">Historical Audit Ledger</h2>
                <p className="mt-0.5 text-[10px] text-slate-500">SQLite persistent storage &mdash; {ledger.length} records &middot; auto-refresh every 15s</p>
              </div>
              <Button onClick={fetchAll} variant="outline" size="sm" className="h-6 text-[10px] border-slate-700 text-slate-300 bg-slate-950 hover:bg-slate-800" disabled={ledgerLoading}>
  {ledgerLoading ? (
    <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  ) : null}
  {ledgerLoading ? "Refreshing..." : "Refresh"}
</Button>
            </div>
            <div className="max-h-[350px] overflow-auto rounded-lg border border-slate-700">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 bg-slate-900 z-10">
                  <tr className="text-left text-slate-400 border-b border-slate-700">
                    <th className="px-3 py-2">Decision</th>
                    <th className="px-3 py-2">Type</th>
                    <th className="px-3 py-2 text-right">Amount</th>
                    <th className="px-3 py-2 text-right">Risk</th>
                    <th className="px-3 py-2">Heuristic</th>
                    <th className="px-3 py-2">Session</th>
                    <th className="px-3 py-2">Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {ledger.map((e) => (
                    <tr key={e.id} className="border-t border-slate-800/50 hover:bg-slate-800/30">
                      <td className="px-3 py-2"><span className="flex items-center gap-1.5"><span className={`h-1.5 w-1.5 rounded-full ${statusDot(e.decision)}`} /><span className="font-medium text-slate-100">{e.decision}</span></span></td>
                      <td className="px-3 py-2 text-slate-300">{e.type}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-100">${e.amount.toLocaleString()}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-100">{e.risk_score.toFixed(1)}%</td>
                      <td className="px-3 py-2 text-slate-400">{e.heuristic_triggered || "-"}</td>
                      <td className="px-3 py-2 text-slate-400 font-mono">{e.session_id}</td>
                      <td className="px-3 py-2 text-slate-500 text-[10px]">{new Date(e.timestamp).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>

      {/* Printable Receipt — hidden on screen, full-page on print */}
      <div className="hidden print:block print:absolute print:inset-0 print:bg-white print:text-black print:p-8 print:w-full print:h-screen print:z-50 print:overflow-auto">
        {result ? (
          <div className="max-w-2xl mx-auto">
            <div className="border-b-2 border-black pb-4 mb-6">
              <h1 className="text-2xl font-bold tracking-tight">Sentinel-XAI</h1>
              <p className="text-sm text-gray-600">Official Transaction Audit Record</p>
            </div>

            <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm mb-6">
              <div className="text-gray-500">Print Date/Time</div>
              <div className="font-semibold">{new Date().toLocaleString()}</div>
              <div className="text-gray-500">Transaction UUID</div>
              <div className="font-mono font-semibold">{result.txn_uuid}</div>
              <div className="text-gray-500">Session ID</div>
              <div className="font-mono">{result.session_id}</div>
              <div className="text-gray-500">Transaction Type</div>
              <div className="font-semibold">{result.type}</div>
            </div>

            <div className="border-t border-gray-300 pt-4 mb-6">
              <h2 className="text-lg font-bold mb-2">Financial Details</h2>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm">
                <div className="text-gray-500">Transaction Amount</div>
                <div className="font-semibold">${(result.amount ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2 })}</div>
                <div className="text-gray-500">Sender Balance (Before)</div>
                <div className="font-semibold">${(result.oldbalanceOrg ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2 })}</div>
                <div className="text-gray-500">Sender Balance (After)</div>
                <div className="font-semibold">${result.newbalanceOrig.toLocaleString("en-US", { minimumFractionDigits: 2 })}</div>
                <div className="text-gray-500">Receiver Balance (After)</div>
                <div className="font-semibold">${(result.newbalanceDest ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2 })}</div>
              </div>
            </div>

            <div className="border-t border-gray-300 pt-4">
              <h2 className="text-lg font-bold mb-2">AI Assessment</h2>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm">
                <div className="text-gray-500">Max-Vote Confidence Score</div>
                <div className="font-semibold">{result.confidence_score.toFixed(2)}%</div>
                <div className="text-gray-500">Final Routing Decision</div>
                <div className={`font-bold text-lg ${result.status === "ALLOW" ? "text-green-700" : result.status === "REVIEW" ? "text-yellow-700" : "text-red-700"}`}>{result.status}</div>
                <div className="text-gray-500">Heuristic Triggered</div>
                <div className="font-semibold">{result.heuristic_triggered || "None"}</div>
                <div className="text-gray-500">Fraud Distance</div>
                <div className="font-mono">{result.euclidean_distance_to_fraud.toFixed(4)}</div>
              </div>
            </div>

            <div className="border-t border-gray-300 mt-8 pt-4 text-[10px] text-gray-400 text-center">
              <p>Sentinel-XAI Fraud Detection System — Max-Vote Ensemble (RF + XGBoost + LightGBM)</p>
              <p className="mt-1">This document is a computer-generated official audit record.</p>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <p>No transaction data available. Run an analysis first, then print.</p>
          </div>
        )}
      </div>
    </main>
  );
}

export default App;
