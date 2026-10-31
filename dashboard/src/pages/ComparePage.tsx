import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api, type CompareData } from '../api';

export default function ComparePage() {
  const [params] = useSearchParams();
  const [runA, setRunA] = useState(params.get('a') || '');
  const [runB, setRunB] = useState(params.get('b') || '');
  const [data, setData] = useState<CompareData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const doCompare = async () => {
    if (!runA || !runB) return;
    setLoading(true);
    setError('');
    try {
      const result = await api.compareRuns(Number(runA), Number(runB));
      setData(result);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (runA && runB) doCompare();
  }, []);

  return (
    <div>
      <h1 className="page-title">Compare Runs</h1>

      <div className="filters" style={{ marginBottom: 20 }}>
        <input className="filter-input" placeholder="Run A (champion)" value={runA}
               onChange={e => setRunA(e.target.value)} style={{ maxWidth: 120 }} />
        <span style={{ color: 'var(--text-muted)' }}>vs</span>
        <input className="filter-input" placeholder="Run B (candidate)" value={runB}
               onChange={e => setRunB(e.target.value)} style={{ maxWidth: 120 }} />
        <button className="btn btn-primary" onClick={doCompare} disabled={!runA || !runB}>
          Compare
        </button>
      </div>

      {error && <div style={{ color: 'var(--fail)', marginBottom: 16 }}>{error}</div>}

      {loading && <div className="skeleton" style={{ height: 200 }} />}

      {data && (
        <div>
          <div style={{ display: 'flex', gap: 20, marginBottom: 20 }}>
            <div className="card" style={{ flex: 1 }}>
              <div className="card-title">Run A</div>
              <p>ID: {data.run_a.id}</p>
              <span className={`status-badge ${data.run_a.status}`}>{data.run_a.status}</span>
              <p style={{ marginTop: 8, color: 'var(--text-secondary)' }}>{data.run_a.branch}</p>
            </div>
            <div className="card" style={{ flex: 1 }}>
              <div className="card-title">Run B</div>
              <p>ID: {data.run_b.id}</p>
              <span className={`status-badge ${data.run_b.status}`}>{data.run_b.status}</span>
              <p style={{ marginTop: 8, color: 'var(--text-secondary)' }}>{data.run_b.branch}</p>
            </div>
          </div>

          <table className="data-table">
            <thead>
              <tr><th>Metric</th><th>Run A</th><th>Run B</th><th>Delta</th></tr>
            </thead>
            <tbody>
              {data.metrics.map(m => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td className="mono">{m.value_a?.toFixed(4) ?? '—'}</td>
                  <td className="mono">{m.value_b?.toFixed(4) ?? '—'}</td>
                  <td className={`mono ${
                    m.delta && m.delta > 0 ? 'delta-positive' :
                    m.delta && m.delta < 0 ? 'delta-negative' : 'delta-neutral'
                  }`}>
                    {m.delta !== null ? `${m.delta > 0 ? '+' : ''}${m.delta.toFixed(4)}` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
