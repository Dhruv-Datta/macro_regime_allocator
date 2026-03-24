import { ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export default function ChartCard({
  title,
  subtitle,
  children,
  className,
}: Props) {
  return (
    <div
      className={`bg-slate-900/60 border border-slate-800 rounded-xl p-5 ${className ?? ""}`}
    >
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
        {subtitle && (
          <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
        )}
      </div>
      {children}
    </div>
  );
}
