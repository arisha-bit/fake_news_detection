import React, { useState } from 'react'
import { generateReport } from '../api/upload'

export default function ReportsPage() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleDownload = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setMessage('')
    try {
      const response = await generateReport({ text, model: 'bert', include_claims: true, include_evidence: true, include_propaganda: true, include_credibility: true })
      const blob = new Blob([response.data], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'verification-report.pdf'
      link.click()
      window.URL.revokeObjectURL(url)
      setMessage('Report downloaded successfully.')
    } catch (err) {
      setMessage(err.response?.data?.detail || 'Report generation failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Download Report</h1>
        <p className="text-slate-400 text-sm mt-1">Generate a downloadable verification report for a news article or multimodal analysis summary.</p>
      </div>
      <form onSubmit={handleDownload} className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm flex flex-col gap-4">
        <textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} className="w-full border border-slate-200 rounded-lg px-4 py-3 text-sm" placeholder="Paste article text or a multimodal report summary..." />
        <button type="submit" disabled={loading || !text.trim()} className="bg-indigo-600 text-white rounded-lg py-3 text-sm font-semibold disabled:opacity-50">
          {loading ? 'Generating PDF...' : 'Generate PDF Report'}
        </button>
        {message && <p className="text-sm text-slate-600">{message}</p>}
      </form>
    </div>
  )
}
