import React, { useState } from 'react'
import { searchEvidence, checkCredibility, analysePropaganda, generateReport, extractKnowledgeGraph } from '../api/upload'

// ─── Evidence Search ───────────────────────────────────────────────────────────
function EvidenceTab() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    setError(''); setResult(null); setLoading(true)
    try {
      const res = await searchEvidence({ query, top_k: topK })
      setResult(res.data)
    } catch (err) { setError(err.response?.data?.detail || 'Search failed.') }
    finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={submit} className="flex gap-2">
        <input value={query} onChange={(e) => setQuery(e.target.value)} required
          placeholder="Enter a claim or article text..."
          className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 text-slate-700" />
        <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}
          className="bg-white border border-slate-200 rounded-lg px-3 text-sm text-slate-600">
          {[3,5,10].map(n => <option key={n} value={n}>Top {n}</option>)}
        </select>
        <button type="submit" disabled={loading || !query.trim()}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-4 py-2 rounded-lg text-sm text-white font-medium">
          {loading ? '...' : 'Search'}
        </button>
      </form>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      {result && (
        <div className="flex flex-col gap-3">
          {result.evidence.map((ev) => (
            <div key={ev.rank} className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm font-medium text-slate-700 flex-1">{ev.title}</p>
                <span className={`shrink-0 text-xs font-bold px-2 py-0.5 rounded border ${
                  ev.label === 'FAKE' ? 'bg-red-50 text-red-500 border-red-100' : 'bg-emerald-50 text-emerald-600 border-emerald-100'
                }`}>{ev.label}</span>
              </div>
              <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">{ev.snippet}</p>
              <div className="flex gap-3 mt-2 text-xs text-slate-400">
                <span>{Math.round(ev.similarity * 100)}% match</span>
                {ev.date && <span>{ev.date}</span>}
                {ev.subject && <span className="capitalize">{ev.subject}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Credibility Check ────────────────────────────────────────────────────────
function CredibilityTab() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    setError(''); setResult(null); setLoading(true)
    try {
      const res = await checkCredibility({ url })
      setResult(res.data)
    } catch (err) { setError(err.response?.data?.detail || 'Check failed.') }
    finally { setLoading(false) }
  }

  const labelColor = (l) => ({ HIGH: 'text-emerald-600', MEDIUM: 'text-amber-500', LOW: 'text-orange-500', VERY_LOW: 'text-red-500' }[l] || 'text-slate-500')

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={submit} className="flex gap-2">
        <input value={url} onChange={(e) => setUrl(e.target.value)} required
          placeholder="e.g. https://reuters.com/article or bbc.com"
          className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 text-slate-700" />
        <button type="submit" disabled={loading || !url.trim()}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-4 py-2 rounded-lg text-sm text-white font-medium">
          {loading ? '...' : 'Check'}
        </button>
      </form>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      {result && (
        <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <p className="font-semibold text-slate-700">{result.domain}</p>
            {result.found_in_database && (
              <span className={`text-sm font-bold ${labelColor(result.credibility_label)}`}>{result.credibility_label}</span>
            )}
          </div>
          <p className="text-sm text-slate-500">{result.verdict}</p>
          {result.found_in_database && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-1">
              {[
                { label: 'Trust Score', value: `${result.trust_score}/100` },
                { label: 'Reliability', value: `${result.reliability_score}/100` },
                { label: 'Bias', value: result.bias_rating },
                { label: 'Category', value: result.category },
              ].map(({ label, value }) => (
                <div key={label} className="bg-slate-50 rounded-lg p-3">
                  <p className="text-xs text-slate-400">{label}</p>
                  <p className="text-sm font-semibold text-slate-700 mt-0.5">{value}</p>
                </div>
              ))}
            </div>
          )}
          {result.notes && <p className="text-xs text-slate-400 italic">{result.notes}</p>}
        </div>
      )}
    </div>
  )
}

// ─── Propaganda Detection ─────────────────────────────────────────────────────
function PropagandaTab() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    setError(''); setResult(null); setLoading(true)
    try {
      const res = await analysePropaganda({ text })
      setResult(res.data)
    } catch (err) { setError(err.response?.data?.detail || 'Analysis failed.') }
    finally { setLoading(false) }
  }

  const scoreColor = (s) => s >= 0.7 ? 'text-red-500' : s >= 0.4 ? 'text-amber-500' : 'text-emerald-600'

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={submit} className="flex flex-col gap-3">
        <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} required
          placeholder="Paste article text to detect propaganda techniques..."
          className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 text-sm resize-none focus:outline-none focus:border-indigo-400 text-slate-700" />
        <button type="submit" disabled={loading || !text.trim()}
          className="self-end bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-5 py-2 rounded-lg text-sm text-white font-medium">
          {loading ? 'Analysing...' : 'Analyse'}
        </button>
      </form>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      {result && (
        <div className="flex flex-col gap-3">
          <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-semibold text-slate-600">Propaganda Score</p>
              <p className={`text-xl font-bold ${scoreColor(result.overall_score)}`}>
                {Math.round(result.overall_score * 100)}%
              </p>
            </div>
            <p className="text-sm text-slate-500">{result.summary}</p>
          </div>
          {result.techniques_found.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              <p className="px-5 py-3 border-b border-slate-100 text-sm font-semibold text-slate-600">Detected Techniques</p>
              <div className="divide-y divide-slate-50">
                {result.techniques_found.map((t, i) => (
                  <div key={i} className="px-5 py-4">
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-sm font-medium text-slate-700">{t.technique}</p>
                      <span className="text-xs font-semibold text-amber-500">{Math.round(t.confidence * 100)}%</span>
                    </div>
                    <p className="text-xs text-slate-400">{t.description}</p>
                    {t.matched_phrases.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {t.matched_phrases.map((p, j) => (
                          <span key={j} className="bg-amber-50 text-amber-600 text-xs px-2 py-0.5 rounded border border-amber-100">"{p}"</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Knowledge Graph ──────────────────────────────────────────────────────────
function KnowledgeGraphTab() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    setError(''); setResult(null); setLoading(true)
    try {
      const res = await extractKnowledgeGraph({ text })
      setResult(res.data)
    } catch (err) { setError(err.response?.data?.detail || 'Extraction failed.') }
    finally { setLoading(false) }
  }

  const typeColors = {
    PERSON: 'bg-indigo-50 text-indigo-600 border-indigo-100',
    ORG: 'bg-violet-50 text-violet-600 border-violet-100',
    GPE: 'bg-emerald-50 text-emerald-600 border-emerald-100',
    DATE: 'bg-amber-50 text-amber-600 border-amber-100',
    EVENT: 'bg-pink-50 text-pink-600 border-pink-100',
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={submit} className="flex flex-col gap-3">
        <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} required
          placeholder="Paste article text to extract entities and relationships..."
          className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 text-sm resize-none focus:outline-none focus:border-indigo-400 text-slate-700" />
        <button type="submit" disabled={loading || !text.trim()}
          className="self-end bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-5 py-2 rounded-lg text-sm text-white font-medium">
          {loading ? 'Extracting...' : 'Extract Graph'}
        </button>
      </form>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      {result && (
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Entities', value: result.total_nodes },
              { label: 'Relationships', value: result.total_edges },
              { label: 'Occurrences', value: result.total_entities },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white rounded-xl border border-slate-200 p-4 text-center shadow-sm">
                <p className="text-2xl font-bold text-slate-800">{value}</p>
                <p className="text-xs text-slate-400 mt-0.5">{label}</p>
              </div>
            ))}
          </div>

          {/* Entity counts by type */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(result.entity_counts).map(([type, count]) => (
              <span key={type} className={`text-xs font-medium px-2.5 py-1 rounded-full border ${typeColors[type] || 'bg-slate-50 text-slate-500 border-slate-200'}`}>
                {type}: {count}
              </span>
            ))}
          </div>

          {/* Nodes */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm">
            <p className="px-5 py-3 border-b border-slate-100 text-sm font-semibold text-slate-600">Entities</p>
            <div className="p-4 flex flex-wrap gap-2">
              {result.nodes.map((node) => (
                <div key={node.id} className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border ${typeColors[node.type] || 'bg-slate-50 text-slate-600 border-slate-200'}`}>
                  <span className="font-medium">{node.label}</span>
                  <span className="opacity-60">×{node.frequency}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Top edges */}
          {result.edges.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm">
              <p className="px-5 py-3 border-b border-slate-100 text-sm font-semibold text-slate-600">Top Relationships</p>
              <div className="divide-y divide-slate-50">
                {result.edges.slice(0, 8).map((edge, i) => (
                  <div key={i} className="px-5 py-3 flex items-center gap-2 text-sm text-slate-600">
                    <span className="font-medium">{edge.source}</span>
                    <span className="text-slate-300">↔</span>
                    <span className="font-medium">{edge.target}</span>
                    <span className="ml-auto text-xs text-slate-400">×{edge.weight}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Report Generator ─────────────────────────────────────────────────────────
function ReportTab() {
  const [text, setText] = useState('')
  const [model, setModel] = useState('logistic')
  const [sourceUrl, setSourceUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      const res = await generateReport({
        text, model,
        source_url: sourceUrl || null,
        include_claims: true, include_evidence: true,
        include_propaganda: true, include_credibility: !!sourceUrl,
      })
      const blob = new Blob([res.data], { type: 'application/pdf' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `verification_report_${Date.now()}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) { setError('Report generation failed. Please try again.') }
    finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={submit} className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm flex flex-col gap-4">
        <textarea rows={5} value={text} onChange={(e) => setText(e.target.value)} required
          placeholder="Paste article text for the full verification report..."
          className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 text-sm resize-none focus:outline-none focus:border-indigo-400 text-slate-700" />
        <input value={sourceUrl} onChange={(e) => setSourceUrl(e.target.value)}
          placeholder="Source URL (optional, for credibility section)"
          className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 text-slate-700" />
        <div className="flex gap-2">
          {['logistic','lstm','bert'].map((m) => (
            <button key={m} type="button" onClick={() => setModel(m)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition capitalize ${
                model === m ? 'bg-indigo-50 border-indigo-300 text-indigo-700' : 'bg-white border-slate-200 text-slate-500'
              }`}>{m}</button>
          ))}
        </div>
        {error && <p className="text-red-500 text-sm">{error}</p>}
        <button type="submit" disabled={loading || !text.trim()}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg py-3 text-sm font-semibold text-white">
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Generating PDF...
            </span>
          ) : '📄 Generate PDF Report'}
        </button>
      </form>
    </div>
  )
}

// ─── Main ToolsPage ───────────────────────────────────────────────────────────
const TABS = [
  { id: 'evidence', label: 'Evidence' },
  { id: 'credibility', label: 'Credibility' },
  { id: 'propaganda', label: 'Propaganda' },
  { id: 'graph', label: 'Knowledge Graph' },
  { id: 'report', label: 'PDF Report' },
]

export default function ToolsPage() {
  const [activeTab, setActiveTab] = useState('evidence')

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Verification Tools</h1>
        <p className="text-slate-400 text-sm mt-1">Advanced analysis tools for deep fact-checking.</p>
      </div>

      <div className="flex gap-1 bg-slate-100 rounded-xl p-1 mb-6 flex-wrap">
        {TABS.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2 px-3 rounded-lg text-xs font-medium transition whitespace-nowrap ${
              activeTab === tab.id ? 'bg-white shadow-sm text-indigo-700' : 'text-slate-500 hover:text-slate-700'
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'evidence' && <EvidenceTab />}
      {activeTab === 'credibility' && <CredibilityTab />}
      {activeTab === 'propaganda' && <PropagandaTab />}
      {activeTab === 'graph' && <KnowledgeGraphTab />}
      {activeTab === 'report' && <ReportTab />}
    </div>
  )
}
