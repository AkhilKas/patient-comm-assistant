import React, { useState, useEffect } from 'react'
import { Upload, MessageCircle, FileText, BarChart3, Loader2, CheckCircle, AlertCircle, Trash2, Heart, Sparkles, ArrowRight, X } from 'lucide-react'

const API_URL = import.meta.env.PROD ? '' : '/api'

export default function App() {
  const [activeTab, setActiveTab] = useState('upload')
  const [stats, setStats] = useState({ total_chunks: 0, sections: [] })

  useEffect(() => { fetchStats() }, [])

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_URL}/stats`)
      const data = await res.json()
      setStats(data)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }

  const tabs = [
    { id: 'upload', label: 'Upload', icon: Upload, color: 'from-violet-500 to-purple-500' },
    { id: 'ask', label: 'Ask Question', icon: MessageCircle, color: 'from-blue-500 to-cyan-500' },
    { id: 'simplify', label: 'Simplify', icon: Sparkles, color: 'from-amber-500 to-orange-500' },
    { id: 'readability', label: 'Analyze', icon: BarChart3, color: 'from-emerald-500 to-teal-500' },
  ]

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Animated background */}
      <div className="fixed inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-slate-800/50 backdrop-blur-xl bg-slate-900/50">
          <div className="max-w-6xl mx-auto px-6 py-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg shadow-purple-500/25">
                  <Heart className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                    Patient Communication Assistant
                  </h1>
                  <p className="text-sm text-slate-500">AI-powered medical document simplification</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="px-4 py-2 bg-slate-800/50 rounded-full border border-slate-700/50">
                  <span className="text-slate-400 text-sm">{stats.total_chunks}</span>
                  <span className="text-slate-500 text-sm ml-1">chunks indexed</span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Navigation */}
        <div className="max-w-6xl mx-auto px-6 mt-8">
          <div className="flex gap-3 p-2 bg-slate-900/50 backdrop-blur rounded-2xl border border-slate-800/50">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center gap-3 px-6 py-4 rounded-xl font-medium transition-all duration-300 ${
                  activeTab === tab.id
                    ? `bg-gradient-to-r ${tab.color} text-white shadow-lg`
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <main className="max-w-6xl mx-auto px-6 py-8">
          {activeTab === 'upload' && <UploadTab onUpload={fetchStats} stats={stats} />}
          {activeTab === 'ask' && <AskTab stats={stats} />}
          {activeTab === 'simplify' && <SimplifyTab />}
          {activeTab === 'readability' && <ReadabilityTab />}
        </main>
      </div>
    </div>
  )
}

function Card({ children, className = '' }) {
  return (
    <div className={`bg-slate-900/50 backdrop-blur border border-slate-800/50 rounded-3xl p-8 ${className}`}>
      {children}
    </div>
  )
}

function UploadTab({ onUpload, stats }) {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  const handleUpload = async (file) => {
    if (!file) return
    setUploading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
      setResult(await res.json())
      onUpload()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    const file = e.dataTransfer.files[0]
    if (file?.type === 'application/pdf') handleUpload(file)
  }

  const handleClear = async () => {
    if (!confirm('Clear all indexed documents?')) return
    try {
      await fetch(`${API_URL}/clear`, { method: 'DELETE' })
      setResult(null)
      onUpload()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <Card>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-2">Upload Medical Document</h2>
        <p className="text-slate-400">Upload discharge summaries, medication guides, or doctor notes</p>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
          dragActive 
            ? 'border-purple-500 bg-purple-500/10' 
            : 'border-slate-700 hover:border-slate-600 hover:bg-slate-800/30'
        }`}
      >
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => handleUpload(e.target.files[0])}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={uploading}
        />
        
        {uploading ? (
          <div className="flex flex-col items-center">
            <div className="w-20 h-20 bg-gradient-to-br from-violet-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6">
              <Loader2 className="w-10 h-10 text-white animate-spin" />
            </div>
            <p className="text-xl font-medium text-white mb-2">Processing document...</p>
            <p className="text-slate-400">Extracting and indexing content</p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="w-20 h-20 bg-gradient-to-br from-slate-700 to-slate-800 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Upload className="w-10 h-10 text-slate-400" />
            </div>
            <p className="text-xl font-medium text-white mb-2">Drop your PDF here</p>
            <p className="text-slate-400 mb-4">or click to browse</p>
            <span className="px-4 py-2 bg-slate-800 rounded-full text-sm text-slate-400">PDF files only</span>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-6 p-5 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-start gap-4">
          <div className="w-10 h-10 bg-red-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-5 h-5 text-red-400" />
          </div>
          <div>
            <p className="font-medium text-red-400">Upload failed</p>
            <p className="text-red-400/70 text-sm mt-1">{error}</p>
          </div>
        </div>
      )}

      {result && (
        <div className="mt-6 p-5 bg-emerald-500/10 border border-emerald-500/20 rounded-2xl flex items-start gap-4">
          <div className="w-10 h-10 bg-emerald-500/20 rounded-xl flex items-center justify-center flex-shrink-0">
            <CheckCircle className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="flex-1">
            <p className="font-medium text-emerald-400">Upload successful!</p>
            <p className="text-emerald-400/70 text-sm mt-1">
              Added {result.chunks_added} chunks from <span className="font-mono">{result.filename}</span>
            </p>
            {result.sections_found.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {result.sections_found.map((section, i) => (
                  <span key={i} className="px-3 py-1 bg-emerald-500/20 rounded-full text-xs text-emerald-400">
                    {section}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {stats.total_chunks > 0 && (
        <div className="mt-8 pt-8 border-t border-slate-800 flex items-center justify-between">
          <div>
            <p className="text-slate-500 text-sm">Currently indexed</p>
            <p className="text-2xl font-bold">{stats.total_chunks} <span className="text-slate-500 text-lg font-normal">chunks</span></p>
          </div>
          <button
            onClick={handleClear}
            className="flex items-center gap-2 px-5 py-3 text-red-400 hover:bg-red-500/10 rounded-xl transition-colors border border-red-500/20"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </button>
        </div>
      )}
    </Card>
  )
}

function AskTab({ stats }) {
  const [question, setQuestion] = useState('')
  const [useSimplifier, setUseSimplifier] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, use_simplifier: useSimplifier, n_results: 3 })
      })
      if (!res.ok) throw new Error((await res.json()).detail || 'Request failed')
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const exampleQuestions = [
    "What medications should I take?",
    "When should I see my doctor?",
    "What are the warning signs?"
  ]

  return (
    <Card>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-2">Ask a Question</h2>
        <p className="text-slate-400">Get answers from your uploaded medical documents</p>
      </div>

      {stats.total_chunks === 0 && (
        <div className="mb-6 p-5 bg-amber-500/10 border border-amber-500/20 rounded-2xl flex items-center gap-4">
          <AlertCircle className="w-5 h-5 text-amber-400" />
          <p className="text-amber-400">Please upload a document first to ask questions.</p>
        </div>
      )}

      <form onSubmit={handleAsk}>
        <div className="relative">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type your question here..."
            className="w-full p-5 bg-slate-800/50 border border-slate-700 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none text-lg"
            rows={3}
            disabled={stats.total_chunks === 0}
          />
        </div>

        {stats.total_chunks > 0 && !question && (
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="text-slate-500 text-sm">Try:</span>
            {exampleQuestions.map((q, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setQuestion(q)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-full text-sm text-slate-300 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        <div className="mt-5 flex items-center justify-between">
          <label className="flex items-center gap-3 cursor-pointer group">
            <div className={`w-12 h-7 rounded-full transition-colors flex items-center px-1 ${useSimplifier ? 'bg-purple-600' : 'bg-slate-700'}`}>
              <div className={`w-5 h-5 bg-white rounded-full transition-transform ${useSimplifier ? 'translate-x-5' : ''}`}></div>
            </div>
            <input type="checkbox" checked={useSimplifier} onChange={(e) => setUseSimplifier(e.target.checked)} className="sr-only" />
            <span className="text-slate-400 group-hover:text-slate-300">AI simplification</span>
          </label>

          <button
            type="submit"
            disabled={loading || !question.trim() || stats.total_chunks === 0}
            className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-blue-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <ArrowRight className="w-5 h-5" />}
            {loading ? 'Searching...' : 'Ask'}
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-6 p-5 bg-red-500/10 border border-red-500/20 rounded-2xl">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {result && (
        <div className="mt-8 space-y-6">
          <div className="p-6 bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-blue-400">Answer</h3>
              <ReadabilityBadge readability={result.readability} />
            </div>
            <p className="text-slate-200 text-lg leading-relaxed whitespace-pre-wrap">{result.answer}</p>
          </div>

          <div>
            <h3 className="font-semibold text-slate-400 mb-4">Sources ({result.sources.length})</h3>
            <div className="space-y-3">
              {result.sources.map((source, i) => (
                <div key={i} className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-3 py-1 bg-slate-700 rounded-lg text-xs font-medium text-slate-300">{source.section}</span>
                    <span className="text-slate-500 text-sm">Score: {source.score}</span>
                  </div>
                  <p className="text-slate-400 text-sm">{source.content.slice(0, 200)}...</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}

function SimplifyTab() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSimplify = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_URL}/simplify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })
      if (!res.ok) throw new Error((await res.json()).detail || 'Request failed')
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const sampleText = "The patient should take omeprazole 20mg orally once daily, 30 minutes prior to the first meal. This proton pump inhibitor reduces gastric acid secretion and promotes healing of esophageal erosions associated with GERD."

  return (
    <Card>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-2">Simplify Medical Text</h2>
        <p className="text-slate-400">Transform complex medical language into patient-friendly content</p>
      </div>

      <div className="mb-4 p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl flex items-center gap-3">
        <Sparkles className="w-5 h-5 text-amber-400" />
        <p className="text-amber-400 text-sm">First request loads the AI model (~30-60 seconds)</p>
      </div>

      <form onSubmit={handleSimplify}>
        <div className="relative">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste complex medical text here..."
            className="w-full p-5 bg-slate-800/50 border border-slate-700 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none text-lg"
            rows={5}
          />
          {!text && (
            <button
              type="button"
              onClick={() => setText(sampleText)}
              className="absolute bottom-4 right-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm text-slate-300 transition-colors"
            >
              Try sample
            </button>
          )}
        </div>

        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="mt-5 w-full flex items-center justify-center gap-3 px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-orange-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
          {loading ? 'Simplifying...' : 'Simplify Text'}
        </button>
      </form>

      {error && (
        <div className="mt-6 p-5 bg-red-500/10 border border-red-500/20 rounded-2xl">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {result && (
        <div className="mt-8 space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-6 bg-slate-800/50 rounded-2xl border border-slate-700/50">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-slate-400">Original</h3>
                <ReadabilityBadge readability={result.readability_before} />
              </div>
              <p className="text-slate-300">{result.original}</p>
            </div>
            <div className="p-6 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 rounded-2xl border border-emerald-500/20">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-emerald-400">Simplified</h3>
                <ReadabilityBadge readability={result.readability_after} />
              </div>
              <p className="text-slate-200">{result.simplified}</p>
            </div>
          </div>

          <div className="p-6 bg-gradient-to-r from-violet-500/10 to-purple-500/10 rounded-2xl border border-purple-500/20">
            <h3 className="font-semibold text-purple-400 mb-4">Improvement Summary</h3>
            <div className="grid grid-cols-3 gap-6">
              <div className="text-center">
                <p className="text-3xl font-bold text-emerald-400">-{result.improvement.grade_level_reduction}</p>
                <p className="text-slate-400 text-sm mt-1">Grade Levels</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-bold text-emerald-400">+{result.improvement.flesch_ease_improvement}</p>
                <p className="text-slate-400 text-sm mt-1">Flesch Ease</p>
              </div>
              <div className="text-center">
                {result.improvement.met_target ? (
                  <div className="flex flex-col items-center">
                    <CheckCircle className="w-8 h-8 text-emerald-400" />
                    <p className="text-slate-400 text-sm mt-2">Target Met!</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <AlertCircle className="w-8 h-8 text-amber-400" />
                    <p className="text-slate-400 text-sm mt-2">Above Target</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}

function ReadabilityTab() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleCheck = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/readability`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })
      setResult(await res.json())
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-2">Analyze Readability</h2>
        <p className="text-slate-400">Check if your text is patient-friendly</p>
      </div>

      <form onSubmit={handleCheck}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste text to analyze..."
          className="w-full p-5 bg-slate-800/50 border border-slate-700 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent resize-none text-lg"
          rows={5}
        />
        <button
          type="submit"
          disabled={loading || !text.trim()}
          className="mt-5 w-full flex items-center justify-center gap-3 px-8 py-4 bg-gradient-to-r from-emerald-500 to-teal-500 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-teal-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <BarChart3 className="w-5 h-5" />}
          Analyze
        </button>
      </form>

      {result && (
        <div className="mt-8">
          <div className={`p-6 rounded-2xl border ${result.is_patient_friendly ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-amber-500/10 border-amber-500/20'}`}>
            <div className="flex items-center gap-4 mb-6">
              {result.is_patient_friendly ? (
                <div className="w-14 h-14 bg-emerald-500/20 rounded-2xl flex items-center justify-center">
                  <CheckCircle className="w-7 h-7 text-emerald-400" />
                </div>
              ) : (
                <div className="w-14 h-14 bg-amber-500/20 rounded-2xl flex items-center justify-center">
                  <AlertCircle className="w-7 h-7 text-amber-400" />
                </div>
              )}
              <div>
                <p className={`text-lg font-semibold ${result.is_patient_friendly ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {result.is_patient_friendly ? 'Patient-Friendly!' : 'Needs Simplification'}
                </p>
                <p className="text-slate-400 text-sm">{result.recommendation}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard label="Grade Level" value={result.readability.avg_grade_level.toFixed(1)} target="≤ 8" good={result.readability.avg_grade_level <= 8} />
              <MetricCard label="Flesch Ease" value={result.readability.flesch_reading_ease.toFixed(0)} target="60-70" good={result.readability.flesch_reading_ease >= 60} />
              <MetricCard label="Word Count" value={result.readability.word_count} />
              <MetricCard label="Words/Sentence" value={result.readability.avg_words_per_sentence.toFixed(1)} target="≤ 20" good={result.readability.avg_words_per_sentence <= 20} />
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}

function ReadabilityBadge({ readability }) {
  const isGood = readability.is_patient_friendly
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
      isGood ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
    }`}>
      {isGood ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
      Grade {readability.avg_grade_level.toFixed(1)}
    </div>
  )
}

function MetricCard({ label, value, target, good }) {
  return (
    <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
      <p className="text-slate-500 text-xs uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${good === true ? 'text-emerald-400' : good === false ? 'text-amber-400' : 'text-white'}`}>{value}</p>
      {target && <p className="text-slate-500 text-xs mt-1">Target: {target}</p>}
    </div>
  )
}