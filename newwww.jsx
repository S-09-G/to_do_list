/**
 * Guided Clustering Page
 * 
 * Lets managers define K clusters by selecting features + weights,
 * then assigns all users via cosine similarity.
 * 
 * Lives in: frontend/src/pages/GuidedClusteringPage.jsx
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Target,
  Plus,
  Trash2,
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  Search,
  ChevronDown,
  ChevronUp,
  History,
  Eye,
  AlertTriangle,
  Users,
  BarChart3,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  getGuidedFeatures,
  runGuidedClustering,
  listGuidedRuns,
  getGuidedRun,
  deleteGuidedRun,
} from '../services/api';

// ============================================================
// COLOR PALETTE FOR CLUSTERS
// ============================================================
const CLUSTER_COLORS = [
  '#4F46E5', '#059669', '#D97706', '#DC2626', '#7C3AED',
  '#0891B2', '#C026D3', '#EA580C', '#2563EB', '#65A30D',
  '#E11D48', '#0D9488', '#9333EA', '#CA8A04', '#6366F1',
  '#14B8A6', '#F43F5E', '#8B5CF6', '#F59E0B', '#10B981',
];

// ============================================================
// FEATURE SEARCH DROPDOWN COMPONENT
// ============================================================
function FeatureSearchDropdown({ features, selectedFeatures, onSelect }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  const filtered = useMemo(() => {
    if (!searchTerm) return features.slice(0, 50);
    const lower = searchTerm.toLowerCase();
    return features
      .filter(
        (f) => f.toLowerCase().includes(lower) && !selectedFeatures.includes(f)
      )
      .slice(0, 50);
  }, [searchTerm, features, selectedFeatures]);

  return (
    <div style={{ position: 'relative' }}>
      <div style={{ display: 'flex', gap: '8px' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search
            size={16}
            style={{
              position: 'absolute',
              left: '10px',
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-secondary)',
            }}
          />
          <input
            type="text"
            placeholder="Search features..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setIsOpen(true);
            }}
            onFocus={() => setIsOpen(true)}
            style={{
              width: '100%',
              padding: '8px 12px 8px 34px',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              fontSize: '0.875rem',
              backgroundColor: 'var(--bg-primary)',
              color: 'var(--text-primary)',
            }}
          />
        </div>
      </div>

      {isOpen && (
        <>
          {/* Invisible overlay to close dropdown on click outside */}
          <div
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 9,
            }}
            onClick={() => setIsOpen(false)}
          />
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              right: 0,
              maxHeight: '200px',
              overflowY: 'auto',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              backgroundColor: 'var(--bg-primary)',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              zIndex: 10,
              marginTop: '4px',
            }}
          >
            {filtered.length === 0 ? (
              <div
                style={{
                  padding: '12px',
                  color: 'var(--text-secondary)',
                  fontSize: '0.875rem',
                  textAlign: 'center',
                }}
              >
                No features found
              </div>
            ) : (
              filtered.map((feat) => (
                <div
                  key={feat}
                  onClick={() => {
                    onSelect(feat);
                    setSearchTerm('');
                    setIsOpen(false);
                  }}
                  style={{
                    padding: '8px 12px',
                    cursor: 'pointer',
                    fontSize: '0.875rem',
                    borderBottom: '1px solid var(--border-color)',
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={(e) =>
                    (e.target.style.backgroundColor = 'var(--bg-secondary)')
                  }
                  onMouseLeave={(e) =>
                    (e.target.style.backgroundColor = 'transparent')
                  }
                >
                  {feat}
                </div>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}





            style={{
              fontSize: '1.8rem',
              fontWeight: 700,
              color: 'var(--primary-color)',
            }}
          >
            {metrics.total_users.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Total Users
          </div>
        </div>
        <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
          <div
            style={{
              fontSize: '1.8rem',
              fontWeight: 700,
              color: 'var(--success-color)',
            }}
          >
            {metrics.total_assigned.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Assigned
          </div>
        </div>
        <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
          <div
            style={{
              fontSize: '1.8rem',
              fontWeight: 700,
              color:
                metrics.total_unassigned > 0
                  ? 'var(--warning-color)'
                  : 'var(--text-secondary)',
            }}
          >
            {metrics.total_unassigned.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Unassigned
          </div>
        </div>
        <div className="card" style={{ padding: '16px', textAlign: 'center' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color: '#7C3AED' }}>
            {metrics.global_mean_similarity}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Mean Similarity
          </div>
        </div>
      </div>

      {/* Distribution chart */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <h3>
            <BarChart3 size={18} /> Cluster Distribution
          </h3>
        </div>
        <div style={{ padding: '16px' }}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={distributionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(value, name) => {
                  if (name === 'users') return [value.toLocaleString(), 'Users'];
                  return [value, name];
                }}
              />
              <Bar dataKey="users" radius={[4, 4, 0, 0]}>
                {distributionData.map((_, i) => (
                  <Cell
                    key={i}
                    fill={
                      i < metrics.cluster_stats.length
                        ? CLUSTER_COLORS[i % CLUSTER_COLORS.length]
                        : '#9CA3AF'
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-cluster details */}
      <div className="card">
        <div className="card-header">
          <h3>
            <Users size={18} /> Cluster Details
          </h3>
        </div>
        <div style={{ padding: '16px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr
                style={{
                  borderBottom: '2px solid var(--border-color)',
                  textAlign: 'left',
                }}
              >
                <th style={{ padding: '8px 12px', fontSize: '0.85rem' }}>Cluster</th>
                <th style={{ padding: '8px 12px', fontSize: '0.85rem' }}>Size</th>
                <th style={{ padding: '8px 12px', fontSize: '0.85rem' }}>%</th>
                <th style={{ padding: '8px 12px', fontSize: '0.85rem' }}>
                  Mean Sim
                </th>
                <th style={{ padding: '8px 12px', fontSize: '0.85rem' }}>
                  Min Sim
                </th>
                <th style={{ padding: '8px 12px', fontSize: '0.85rem' }}>
                  Top Actual Features (by lift)
                </th>
              </tr>
            </thead>
            <tbody>
              {metrics.cluster_stats.map((cs) => {
                const profile = profiles[String(cs.cluster_id)] || {};
                const topFeats = (profile.top_features || []).slice(0, 5);

                return (
                  <tr
                    key={cs.cluster_id}
                    style={{ borderBottom: '1px solid var(--border-color)' }}
                  >
                    <td style={{ padding: '10px 12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div
                          style={{
                            width: '10px',
                            height: '10px',
                            borderRadius: '50%',
                            backgroundColor:
                              CLUSTER_COLORS[cs.cluster_id % CLUSTER_COLORS.length],
                          }}
                        />
                        Cluster {cs.cluster_id}
                      </div>
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>
                      {cs.size.toLocaleString()}
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>
                      {cs.percentage}%
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>
                      {cs.mean_similarity}
                    </td>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>
                      {cs.min_similarity}
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: '0.8rem' }}>
                      {topFeats.map((f) => (
                        <span
                          key={f.feature}
                          style={{
                            display: 'inline-block',
                            padding: '2px 8px',
                            margin: '2px 4px 2px 0',
                            backgroundColor: 'var(--bg-secondary)',
                            borderRadius: '4px',
                            fontFamily: 'monospace',
                            fontSize: '0.75rem',
                          }}
                        >
                          {f.feature}{' '}
                          <span style={{ color: 'var(--success-color)' }}>
                            {f.lift}x
                          </span>
                        </span>
                      ))}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}










                <div className="alert alert-error" style={{ marginBottom: '16px' }}>
                  <XCircle size={20} />
                  <div>
                    <strong>Error:</strong> {runError}
                  </div>
                </div>
              )}

              {/* Run success */}
              {result && result.success && (
                <>
                  <div
                    className="alert alert-success"
                    style={{ marginBottom: '16px' }}
                  >
                    <CheckCircle size={20} />
                    <div>{result.message}</div>
                  </div>
                  <ResultsView result={result} />
                </>
              )}
            </>
          )}
        </>
      )}

      {/* ---- HISTORY TAB ---- */}
      {activeTab === 'history' && (
        <>
          {historyLoading ? (
            <div className="loading">
              <Loader2 className="spinning" size={40} />
              <p>Loading run history...</p>
            </div>
          ) : (
            <HistoryView
              runs={runs}
              onViewRun={handleViewRun}
              onDeleteRun={handleDeleteRun}
              loadingRunId={loadingRunId}
            />
          )}
        </>
      )}
    </div>
  );
}

export default GuidedClusteringPage;
