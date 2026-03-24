import { useState } from "react";
import { useAppData } from "./data";
import Dashboard from "./pages/Dashboard";
import Report from "./pages/Report";

const TABS = ["Charts", "Report"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const { data, error } = useAppData();
  const [tab, setTab] = useState<Tab>("Charts");

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 text-lg font-medium">Failed to load data</p>
          <p className="text-slate-500 text-sm mt-2">{error}</p>
          <p className="text-slate-600 text-xs mt-4">
            Run <code className="text-slate-400">make prepare</code> first.
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
      </div>
    );
  }

  return (
    <div>
      {/* Tab bar */}
      <div className="sticky top-0 z-10 bg-slate-950 border-b border-slate-800">
        <div className="max-w-6xl mx-auto px-6 flex gap-1 pt-4">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                tab === t
                  ? "bg-slate-900/80 text-white border border-slate-800 border-b-transparent -mb-px"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {tab === "Charts" && <Dashboard data={data} />}
      {tab === "Report" && <Report data={data} />}
    </div>
  );
}
