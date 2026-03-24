export interface BacktestRow {
  date: string;
  rebalance_date: string;
  pred_class: number;
  actual_label: number;
  prob_equity: number;
  prob_tbills: number;
  weight_equity: number;
  weight_tbills: number;
  overlay: string;
  ret_equity: number;
  ret_tbills: number;
  port_return: number;
  ew_return: number;
  ret_6040: number;
  train_size: number;
  cum_port: number;
  cum_ew: number;
  cum_equity: number;
  cum_tbills: number;
  cum_6040: number;
  turnover: number;
}

export interface MetricsRow {
  label: string;
  cagr: number;
  volatility: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  max_dd_duration: number;
  hit_rate: number;
  total_return: number;
  best_month: number;
  worst_month: number;
  avg_up_month: number;
  avg_down_month: number;
  up_down_ratio: number;
  win_streak: number;
  lose_streak: number;
  n_months: number;
}
