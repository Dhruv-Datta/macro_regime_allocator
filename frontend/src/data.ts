import { useEffect, useState } from "react";
import Papa from "papaparse";
import type { BacktestRow, MetricsRow } from "./types";

async function loadCsv<T>(url: string): Promise<T[]> {
  const res = await fetch(url);
  const text = await res.text();
  const { data } = Papa.parse<T>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return data;
}

export interface AppData {
  backtest: BacktestRow[];
  metrics: MetricsRow[];
}

export function useAppData() {
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      loadCsv<BacktestRow>("/data/backtest.csv"),
      loadCsv<MetricsRow>("/data/metrics.csv"),
    ])
      .then(([backtest, metrics]) => setData({ backtest, metrics }))
      .catch((e) => setError(e.message));
  }, []);

  return { data, error };
}

export function pct(v: number, decimals = 1): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function num(v: number, decimals = 2): string {
  return v.toFixed(decimals);
}

export function getMetric(
  metrics: MetricsRow[],
  label: string
): MetricsRow | undefined {
  return metrics.find((m) => m.label === label);
}

export function computeDrawdown(cum: number[]): number[] {
  let peak = cum[0] ?? 100;
  return cum.map((v) => {
    if (v > peak) peak = v;
    return (v - peak) / peak;
  });
}

export function rollingWindow(
  data: number[],
  window: number,
  fn: (slice: number[]) => number
): (number | null)[] {
  return data.map((_, i) => {
    if (i < window - 1) return null;
    return fn(data.slice(i - window + 1, i + 1));
  });
}

export function sharpe(returns: number[]): number {
  const mean = returns.reduce((s, v) => s + v, 0) / returns.length;
  const std = Math.sqrt(
    returns.reduce((s, v) => s + (v - mean) ** 2, 0) / returns.length
  );
  return std === 0 ? 0 : (mean / std) * Math.sqrt(12);
}

export const COLORS = {
  model: "#6366f1",
  ew: "#64748b",
  equity: "#f59e0b",
  tbills: "#10b981",
  ref6040: "#8b5cf6",
  positive: "#22c55e",
  negative: "#ef4444",
  overlay: "#f43f5e",
  weight: "#3b82f6",
  prob: "#a78bfa",
};
