import { useMemo } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
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
import ChartCard from "../components/ChartCard";

const COEFFICIENTS = [
  { label: "Inflation Impulse", coeff: 0.373 },
  { label: "VIX 1m Change", coeff: 0.217 },
  { label: "Inflation YoY", coeff: 0.156 },
  { label: "Equity Momentum 3m", coeff: 0.129 },
  { label: "VIX Term Structure", coeff: 0.12 },
  { label: "Equity Drawdown", coeff: 0.019 },
  { label: "Unemployment Rate", coeff: -0.078 },
  { label: "Credit Spread 3m Chg", coeff: -0.104 },
  { label: "Credit Spread Level", coeff: -0.123 },
  { label: "Real Fed Funds", coeff: -0.145 },
  { label: "Yield Curve Slope", coeff: -0.167 },
  { label: "Equity Vol 3m", coeff: -0.282 },
].sort((a, b) => Math.abs(b.coeff) - Math.abs(a.coeff));

const ttStyle = {
  background: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 8,
};

const legendStyle = { fontSize: 11, color: "#94a3b8" };

export default function Dashboard({ data }: { data: AppData }) {
  const { backtest, metrics } = data;

  const model = getMetric(metrics, "Model Portfolio")!;
  const ew = getMetric(metrics, "90/10")!;
  const equity = getMetric(metrics, "Equity Only")!;

  const valid = useMemo(
    () => backtest.filter((r) => r.cum_port != null && r.port_return != null),
    [backtest]
  );

  const cumData = useMemo(
    () =>
      valid.map((r) => ({
        date: r.date.slice(0, 7),
        Model: +(r.cum_port / 100).toFixed(3),
        "90/10": +(r.cum_ew / 100).toFixed(3),
        Equity: +(r.cum_equity / 100).toFixed(3),
        "T-Bills": +(r.cum_tbills / 100).toFixed(3),
        "60/40": +(r.cum_6040 / 100).toFixed(3),
      })),
    [valid]
  );

  const ddData = useMemo(() => {
    const ddM = computeDrawdown(valid.map((r) => r.cum_port));
    const ddE = computeDrawdown(valid.map((r) => r.cum_equity));
    const ddEw = computeDrawdown(valid.map((r) => r.cum_ew));
    return valid.map((r, i) => ({
      date: r.date.slice(0, 7),
      Model: +(ddM[i]! * 100).toFixed(2),
      Equity: +(ddE[i]! * 100).toFixed(2),
      "90/10": +(ddEw[i]! * 100).toFixed(2),
    }));
  }, [valid]);

  const weightData = useMemo(
    () =>
      valid
        .filter((r) => r.weight_equity != null)
        .map((r) => ({
          date: r.date.slice(0, 7),
          "Equity Weight": +(r.weight_equity * 100).toFixed(1),
          "Overlay Active":
            r.overlay !== "none"
              ? +(r.weight_equity * 100).toFixed(1)
              : null,
        })),
    [valid]
  );

  const probData = useMemo(
    () =>
      valid
        .filter((r) => r.prob_equity != null)
        .map((r) => ({
          date: r.date.slice(0, 7),
          "P(Equity)": +(r.prob_equity * 100).toFixed(1),
        })),
    [valid]
  );

  const rollingSharpeData = useMemo(() => {
    const rM = rollingWindow(valid.map((r) => r.port_return), 12, sharpe);
    const rE = rollingWindow(valid.map((r) => r.ret_equity), 12, sharpe);
    return valid.map((r, i) => ({
      date: r.date.slice(0, 7),
      Model: rM[i] != null ? +rM[i]!.toFixed(2) : null,
      Equity: rE[i] != null ? +rE[i]!.toFixed(2) : null,
    }));
  }, [valid]);

  const confusion = useMemo(() => {
    let tp = 0, fn = 0, fp = 0, tn = 0;
    for (const r of valid) {
      if (r.actual_label == null || r.pred_class == null) continue;
      if (r.actual_label === 0 && r.pred_class === 0) tp++;
      else if (r.actual_label === 0 && r.pred_class === 1) fn++;
      else if (r.actual_label === 1 && r.pred_class === 0) fp++;
      else if (r.actual_label === 1 && r.pred_class === 1) tn++;
    }
    const total = tp + fn + fp + tn;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    return { tp, fn, fp, tn, accuracy };
  }, [valid]);

  const tick = { fill: "#64748b", fontSize: 11 };
  const interval = Math.floor(cumData.length / 8);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Macro Regime Allocator</h1>
        <p className="text-sm text-slate-500 mt-1">
          {valid[0]?.date.slice(0, 7)} to {valid[valid.length - 1]?.date.slice(0, 7)} · {model.n_months} months
        </p>
      </div>

      <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
        {[
          { l: "CAGR", v: pct(model.cagr), s: `vs ${pct(ew.cagr)}` },
          { l: "Sharpe", v: num(model.sharpe), s: `vs ${num(ew.sharpe)}` },
          { l: "Sortino", v: num(model.sortino), s: `vs ${num(ew.sortino)}` },
          { l: "Max DD", v: pct(model.max_drawdown), s: `vs ${pct(ew.max_drawdown)}` },
          { l: "Volatility", v: pct(model.volatility), s: `vs ${pct(equity.volatility)}` },
          { l: "Total Return", v: pct(model.total_return), s: `vs ${pct(equity.total_return)}` },
        ].map((m) => (
          <div key={m.l} className="bg-slate-900/60 border border-slate-800 rounded-xl p-3">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">{m.l}</p>
            <p className="text-lg font-semibold text-white mono">{m.v}</p>
            <p className="text-[10px] text-slate-600">{m.s}</p>
          </div>
        ))}
      </div>

      {/* 1. Cumulative Returns */}
      <ChartCard title="Cumulative Returns" subtitle="Growth of $1 (log scale)">
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={cumData}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} scale="log" domain={["auto", "auto"]} tickFormatter={(v: number) => `$${v.toFixed(1)}`} />
            <Tooltip formatter={(v: number) => [`$${v.toFixed(2)}`]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={1} stroke="#334155" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="Model" stroke={COLORS.model} strokeWidth={2.5} dot={false} />
            <Line type="monotone" dataKey="90/10" stroke={COLORS.ew} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
            <Line type="monotone" dataKey="Equity" stroke={COLORS.equity} strokeWidth={1.5} dot={false} opacity={0.7} />
            <Line type="monotone" dataKey="60/40" stroke={COLORS.ref6040} strokeWidth={1.5} dot={false} opacity={0.7} />
            <Line type="monotone" dataKey="T-Bills" stroke={COLORS.tbills} strokeWidth={1} dot={false} opacity={0.5} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 2. Drawdowns */}
      <ChartCard title="Drawdowns" subtitle="Model vs Equity vs 90/10">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={ddData}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v}%`} />
            <Tooltip formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={0} stroke="#334155" />
            <Area type="monotone" dataKey="Equity" stroke={COLORS.equity} fill={COLORS.equity} fillOpacity={0.08} strokeWidth={1} />
            <Area type="monotone" dataKey="90/10" stroke={COLORS.ew} fill={COLORS.ew} fillOpacity={0.08} strokeWidth={1} />
            <Area type="monotone" dataKey="Model" stroke={COLORS.negative} fill={COLORS.negative} fillOpacity={0.15} strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 3. Equity Weight Over Time */}
      <ChartCard title="Equity Weight Over Time" subtitle="Red dots = crash overlay active">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={weightData}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={90} stroke="#64748b" strokeDasharray="4 2" />
            <Area type="stepAfter" dataKey="Equity Weight" stroke={COLORS.weight} fill={COLORS.weight} fillOpacity={0.12} strokeWidth={2} />
            <Area type="stepAfter" dataKey="Overlay Active" stroke={COLORS.overlay} fill="transparent" strokeWidth={0} dot={{ r: 3, fill: COLORS.overlay }} connectNulls={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 4. Probabilities Over Time */}
      <ChartCard title="P(Equity Outperforms)" subtitle="Model probability — above 50% = equity regime">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={probData}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} domain={[20, 80]} tickFormatter={(v: number) => `${v}%`} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`]} contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={50} stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="6 3" />
            <Area type="monotone" dataKey="P(Equity)" stroke={COLORS.prob} fill={COLORS.prob} fillOpacity={0.15} strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 5. Rolling 12m Sharpe */}
      <ChartCard title="Rolling 12-Month Sharpe Ratio" subtitle="Model vs Equity">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={rollingSharpeData}>
            <XAxis dataKey="date" tick={tick} tickLine={false} interval={interval} />
            <YAxis tick={tick} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={ttStyle} />
            <Legend wrapperStyle={legendStyle} />
            <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="Model" stroke={COLORS.model} strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="Equity" stroke={COLORS.equity} strokeWidth={1.5} dot={false} opacity={0.6} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* 6. Feature Coefficients */}
      <ChartCard title="Feature Coefficients" subtitle="Negative = favors equity, Positive = favors T-bills">
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={COEFFICIENTS} layout="vertical">
            <XAxis type="number" tick={tick} tickLine={false} domain={[-0.4, 0.4]} />
            <YAxis type="category" dataKey="label" tick={{ fill: "#94a3b8", fontSize: 11 }} tickLine={false} axisLine={false} width={160} />
            <Tooltip formatter={(v: number) => [v.toFixed(3), "Coefficient"]} contentStyle={ttStyle} />
            <ReferenceLine x={0} stroke="#475569" />
            <Bar dataKey="coeff" radius={[0, 4, 4, 0]}>
              {COEFFICIENTS.map((d) => (
                <Cell key={d.label} fill={d.coeff < 0 ? COLORS.positive : COLORS.negative} opacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex justify-center gap-8 mt-2 text-xs text-slate-500">
          <span><span className="inline-block w-3 h-3 rounded bg-emerald-500/80 mr-1.5" />Favors Equity</span>
          <span><span className="inline-block w-3 h-3 rounded bg-red-500/80 mr-1.5" />Favors T-Bills</span>
        </div>
      </ChartCard>

      {/* 7. Confusion Matrix */}
      <ChartCard title="Confusion Matrix" subtitle={`Accuracy: ${(confusion.accuracy * 100).toFixed(1)}%`}>
        <div className="flex justify-center">
          <table className="text-sm mono">
            <thead>
              <tr>
                <th />
                <th className="px-6 py-2 text-slate-500 font-medium text-center">Pred Equity</th>
                <th className="px-6 py-2 text-slate-500 font-medium text-center">Pred T-Bills</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="pr-4 py-2 text-slate-400 font-medium text-right">Actual Equity</td>
                <td className="px-6 py-3 text-center rounded-tl-lg bg-emerald-500/20 text-emerald-300 text-lg font-bold">{confusion.tp}</td>
                <td className="px-6 py-3 text-center rounded-tr-lg bg-red-500/15 text-red-300 text-lg">{confusion.fn}</td>
              </tr>
              <tr>
                <td className="pr-4 py-2 text-slate-400 font-medium text-right">Actual T-Bills</td>
                <td className="px-6 py-3 text-center rounded-bl-lg bg-red-500/15 text-red-300 text-lg">{confusion.fp}</td>
                <td className="px-6 py-3 text-center rounded-br-lg bg-emerald-500/20 text-emerald-300 text-lg font-bold">{confusion.tn}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </ChartCard>

      <div className="h-8" />
    </div>
  );
}
