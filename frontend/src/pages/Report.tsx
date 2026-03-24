import { useMemo } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { AppData } from "../data";
import {
  getMetric,
  pct,
  num,
  computeDrawdown,
  rollingWindow,
  sharpe,
  COLORS,
} from "../data";
import type { MetricsRow } from "../types";
import ChartCard from "../components/ChartCard";

const ttStyle = {
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 8,
};
const legendStyle = { fontSize: 11, color: "#94a3b8" };

interface DrawdownPeriod {
  start: string;
  trough: string;
  end: string | null;
  depthModel: number;
  depthEquity: number;
  depth6040: number;
  months: number;
}

export default function Report({ data }: { data: AppData }) {
  const { backtest, metrics } = data;

  const model = getMetric(metrics, "Model Portfolio")!;
  const equity = getMetric(metrics, "Equity Only")!;
  const ref6040 = getMetric(metrics, "60/40 Reference")!;

  const valid = useMemo(
    () => backtest.filter((r) => r.cum_port != null && r.port_return != null),
    [backtest]
  );

  const tick = { fill: "#64748b", fontSize: 11 };
  const interval = Math.floor(valid.length / 8);

  // Cumulative — only the 3 strategies
  const cumData = useMemo(
    () =>
      valid.map((r) => ({
        date: r.date.slice(0, 7),
        Model: +(r.cum_port / 100).toFixed(3),
        Equity: +(r.cum_equity / 100).toFixed(3),
        "60/40": +(r.cum_6040 / 100).toFixed(3),
      })),
    [valid]
  );

  // Drawdowns — the 3 strategies
  const ddModel = useMemo(() => computeDrawdown(valid.map((r) => r.cum_port)), [valid]);
  const ddEquity = useMemo(() => computeDrawdown(valid.map((r) => r.cum_equity)), [valid]);
  const dd6040 = useMemo(() => computeDrawdown(valid.map((r) => r.cum_6040)), [valid]);

  const ddData = useMemo(
    () =>
      valid.map((r, i) => ({
        date: r.date.slice(0, 7),
        Model: +(ddModel[i]! * 100).toFixed(2),
        Equity: +(ddEquity[i]! * 100).toFixed(2),
        "60/40": +(dd6040[i]! * 100).toFixed(2),
      })),
    [valid, ddModel, ddEquity, dd6040]
  );

  // Rolling 12m return
  const rolling12m = useMemo(() => {
    const compound = (arr: number[]) => arr.reduce((a, v) => a * (1 + v), 1) - 1;
    const rM = rollingWindow(valid.map((r) => r.port_return), 12, compound);
    const rE = rollingWindow(valid.map((r) => r.ret_equity), 12, compound);
    const r6 = rollingWindow(valid.map((r) => r.ret_6040), 12, compound);
    return valid.map((r, i) => ({
      date: r.date.slice(0, 7),
      Model: rM[i] != null ? +(rM[i]! * 100).toFixed(1) : null,
      Equity: rE[i] != null ? +(rE[i]! * 100).toFixed(1) : null,
      "60/40": r6[i] != null ? +(r6[i]! * 100).toFixed(1) : null,
    }));
  }, [valid]);

  // Rolling 12m vol
  const rollingVol = useMemo(() => {
    const vol12 = (rets: number[]) => {
      const mean = rets.reduce((s, v) => s + v, 0) / rets.length;
      return Math.sqrt(rets.reduce((s, v) => s + (v - mean) ** 2, 0) / rets.length) * Math.sqrt(12);
    };
    const vM = rollingWindow(valid.map((r) => r.port_return), 12, vol12);
    const vE = rollingWindow(valid.map((r) => r.ret_equity), 12, vol12);
    const v6 = rollingWindow(valid.map((r) => r.ret_6040), 12, vol12);
    return valid.map((r, i) => ({
      date: r.date.slice(0, 7),
      Model: vM[i] != null ? +(vM[i]! * 100).toFixed(1) : null,
      Equity: vE[i] != null ? +(vE[i]! * 100).toFixed(1) : null,
      "60/40": v6[i] != null ? +(v6[i]! * 100).toFixed(1) : null,
    }));
  }, [valid]);

  // Significant drawdown episodes (model DD < -5%)
  const ddPeriods = useMemo(() => {
    const cumM = valid.map((r) => r.cum_port);
    const cumE = valid.map((r) => r.cum_equity);
    const cum6 = valid.map((r) => r.cum_6040);
    const periods: DrawdownPeriod[] = [];
    let peak = cumM[0]!;
    let peakIdx = 0;
    let troughIdx = 0;
    let inDD = false;

    for (let i = 1; i < cumM.length; i++) {
      const v = cumM[i]!;
      if (v > peak) {
        if (inDD) {
          const depthM = (cumM[troughIdx]! - peak) / peak;
          if (depthM < -0.04) {
            // compute equity and 60/40 DD over the same peak→trough window
            const eqPeak = Math.max(...cumE.slice(peakIdx, troughIdx + 1));
            const r6Peak = Math.max(...cum6.slice(peakIdx, troughIdx + 1));
            periods.push({
              start: valid[peakIdx]!.date.slice(0, 7),
              trough: valid[troughIdx]!.date.slice(0, 7),
              end: valid[i]!.date.slice(0, 7),
              depthModel: depthM * 100,
              depthEquity: ((cumE[troughIdx]! - eqPeak) / eqPeak) * 100,
              depth6040: ((cum6[troughIdx]! - r6Peak) / r6Peak) * 100,
              months: troughIdx - peakIdx,
            });
          }
          inDD = false;
        }
        peak = v;
        peakIdx = i;
        troughIdx = i;
      } else {
        inDD = true;
        if (v < cumM[troughIdx]!) troughIdx = i;
      }
    }
    if (inDD) {
      const depthM = (cumM[troughIdx]! - peak) / peak;
      if (depthM < -0.04) {
        const eqPeak = Math.max(...cumE.slice(peakIdx, troughIdx + 1));
        const r6Peak = Math.max(...cum6.slice(peakIdx, troughIdx + 1));
        periods.push({
          start: valid[peakIdx]!.date.slice(0, 7),
          trough: valid[troughIdx]!.date.slice(0, 7),
          end: null,
          depthModel: depthM * 100,
          depthEquity: ((cumE[troughIdx]! - eqPeak) / eqPeak) * 100,
          depth6040: ((cum6[troughIdx]! - r6Peak) / r6Peak) * 100,
          months: troughIdx - peakIdx,
        });
      }
    }
    return periods.sort((a, b) => a.depthModel - b.depthModel);
  }, [valid]);

  // Downside / upside capture vs equity
  const downMonths = valid.filter((r) => r.ret_equity < 0);
  const upMonths = valid.filter((r) => r.ret_equity > 0);
  const modelDownCapture = downMonths.length > 0
    ? downMonths.reduce((s, r) => s + r.port_return, 0) / downMonths.reduce((s, r) => s + r.ret_equity, 0)
    : 0;
  const modelUpCapture = upMonths.length > 0
    ? upMonths.reduce((s, r) => s + r.port_return, 0) / upMonths.reduce((s, r) => s + r.ret_equity, 0)
    : 0;
  const r6DownCapture = downMonths.length > 0
    ? downMonths.reduce((s, r) => s + r.ret_6040, 0) / downMonths.reduce((s, r) => s + r.ret_equity, 0)
    : 0;
  const r6UpCapture = upMonths.length > 0
    ? upMonths.reduce((s, r) => s + r.ret_6040, 0) / upMonths.reduce((s, r) => s + r.ret_equity, 0)
    : 0;

  const metricRow = (label: string, fn: (m: MetricsRow) => string, highlight?: boolean) => (
    <tr key={label} className="border-b border-slate-800/40">
      <td className="py-2 pr-4 text-slate-400 text-xs">{label}</td>
      {[model, equity, ref6040].map((m) => (
        <td
          key={m.label}
          className={`py-2 px-4 text-right mono text-xs ${
            m.label === "Model Portfolio" ? "text-indigo-400 font-medium" : "text-slate-300"
          }`}
        >
          {fn(m)}
        </td>
      ))}
    </tr>
  );

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Performance Report</h1>
        <p className="text-sm text-slate-500 mt-1">
          Model vs Equity vs 60/40 · {valid[0]?.date.slice(0, 7)} to{" "}
          {valid[valid.length - 1]?.date.slice(0, 7)} · {model.n_months} months
        </p>
      </div>

      {/* Performance comparison table */}
      <ChartCard title="Performance Summary">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500">
                <th className="text-left py-2 pr-4 font-medium text-xs">Metric</th>
                <th className="text-right py-2 px-4 font-medium text-xs">Model</th>
                <th className="text-right py-2 px-4 font-medium text-xs">Equity</th>
                <th className="text-right py-2 px-4 font-medium text-xs">60/40</th>
              </tr>
            </thead>
            <tbody>
              {metricRow("CAGR", (m) => pct(m.cagr))}
              {metricRow("Total Return", (m) => pct(m.total_return))}
              {metricRow("Volatility", (m) => pct(m.volatility))}
              {metricRow("Sharpe", (m) => num(m.sharpe))}
              {metricRow("Sortino", (m) => num(m.sortino))}
              {metricRow("Calmar", (m) => num(m.calmar))}
              {metricRow("Max Drawdown", (m) => pct(m.max_drawdown))}
              {metricRow("Max DD Duration", (m) => `${m.max_dd_duration} mo`)}
              {metricRow("Hit Rate", (m) => pct(m.hit_rate))}
              {metricRow("Best Month", (m) => pct(m.best_month))}
              {metricRow("Worst Month", (m) => pct(m.worst_month))}
              {metricRow("Avg Up Month", (m) => pct(m.avg_up_month))}
              {metricRow("Avg Down Month", (m) => pct(m.avg_down_month))}
              {metricRow("Up/Down Ratio", (m) => num(m.up_down_ratio))}
            </tbody>
          </table>
        </div>
      </ChartCard>

      {/* Capture ratios */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { l: "Model Upside Capture", v: `${(modelUpCapture * 100).toFixed(0)}%`, c: "text-emerald-400" },
          { l: "Model Downside Capture", v: `${(modelDownCapture * 100).toFixed(0)}%`, c: "text-amber-400" },
          { l: "60/40 Upside Capture", v: `${(r6UpCapture * 100).toFixed(0)}%`, c: "text-emerald-400" },
          { l: "60/40 Downside Capture", v: `${(r6DownCapture * 100).toFixed(0)}%`, c: "text-amber-400" },
        ].map((m) => (
          <div key={m.l} className="bg-slate-900/60 border border-slate-800 rounded-xl p-3">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">{m.l}</p>
            <p className={`text-lg font-semibold mono ${m.c}`}>{m.v}</p>
            <p className="text-[10px] text-slate-600">vs equity</p>
          </div>
        ))}
      </div>

      {/* Cumulative returns — 3 strategies */}
      <ChartCard title="Cumulative Returns" subtitle="Growth of $1 — Model vs Equity vs 60/40">
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={cumData}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} scale="log" domain={["auto", "auto"]} tickFormatter={(v: number) => `$${v.toFixed(1)}`} />
            <Tooltip formatter={(v: number) => [`$${v.toFixed(2)}`]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={1} stroke="#334155" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="Model" stroke={COLORS.model} strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="Equity" stroke={COLORS.equity} strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="60/40" stroke={COLORS.ref6040} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Drawdowns — 3 strategies */}
      <ChartCard title="Drawdowns" subtitle="Peak-to-trough decline — Model vs Equity vs 60/40">
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={ddData}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v}%`} />
            <Tooltip formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={0} stroke="#334155" />
            <Area type="monotone" dataKey="Equity" stroke={COLORS.equity} fill={COLORS.equity} fillOpacity={0.08} strokeWidth={1} />
            <Area type="monotone" dataKey="60/40" stroke={COLORS.ref6040} fill={COLORS.ref6040} fillOpacity={0.08} strokeWidth={1} />
            <Area type="monotone" dataKey="Model" stroke={COLORS.negative} fill={COLORS.negative} fillOpacity={0.15} strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Drawdown episodes table */}
      <ChartCard title="Drawdown Episodes" subtitle="Model drawdowns deeper than 4% — compared across strategies over the same window">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500">
                <th className="text-left py-2 pr-3 font-medium text-xs">Period</th>
                <th className="text-right py-2 px-3 font-medium text-xs">Model DD</th>
                <th className="text-right py-2 px-3 font-medium text-xs">Equity DD</th>
                <th className="text-right py-2 px-3 font-medium text-xs">60/40 DD</th>
                <th className="text-right py-2 px-3 font-medium text-xs">DD Saved vs Equity</th>
                <th className="text-right py-2 font-medium text-xs">Months</th>
              </tr>
            </thead>
            <tbody className="mono text-xs">
              {ddPeriods.map((p, i) => (
                <tr key={i} className="border-b border-slate-800/30">
                  <td className="py-2 pr-3 text-slate-300 font-sans">
                    {p.start} → {p.trough}
                  </td>
                  <td className="py-2 px-3 text-right text-indigo-400 font-medium">
                    {p.depthModel.toFixed(1)}%
                  </td>
                  <td className="py-2 px-3 text-right text-amber-400">
                    {p.depthEquity.toFixed(1)}%
                  </td>
                  <td className="py-2 px-3 text-right text-violet-400">
                    {p.depth6040.toFixed(1)}%
                  </td>
                  <td className={`py-2 px-3 text-right font-medium ${
                    p.depthModel > p.depthEquity ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {(p.depthEquity - p.depthModel).toFixed(1)}%
                  </td>
                  <td className="py-2 text-right text-slate-400">{p.months}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>

      {/* Rolling 12m return */}
      <ChartCard title="Rolling 12-Month Return" subtitle="Trailing annual performance">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={rolling12m}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v}%`} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="Model" stroke={COLORS.model} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Equity" stroke={COLORS.equity} strokeWidth={1.5} dot={false} opacity={0.7} />
            <Line type="monotone" dataKey="60/40" stroke={COLORS.ref6040} strokeWidth={1.5} dot={false} opacity={0.7} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Rolling 12m volatility */}
      <ChartCard title="Rolling 12-Month Volatility" subtitle="Annualized standard deviation">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={rollingVol}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v}%`} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <Line type="monotone" dataKey="Model" stroke={COLORS.model} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Equity" stroke={COLORS.equity} strokeWidth={1.5} dot={false} opacity={0.7} />
            <Line type="monotone" dataKey="60/40" stroke={COLORS.ref6040} strokeWidth={1.5} dot={false} opacity={0.7} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="h-8" />
    </div>
  );
}
