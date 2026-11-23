import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const API = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const initial = {
  age: 50, education: 2, sex: 'F', cigsPerDay: 0, BPMeds: 0,
  prevalentStroke: 0, prevalentHyp: 0, diabetes: 0, totChol: 220,
  sysBP: 120, diaBP: 80, BMI: 25, heartRate: 72, glucose: 85
}
const fields = [
  ['age','Age (years)','number'], ['education','Education level (1–4)','number'],
  ['sex','Sex','select'], ['cigsPerDay','Cigarettes per day','number'],
  ['BPMeds','Blood pressure medication','binary'],
  ['prevalentStroke','Previous stroke','binary'],
  ['prevalentHyp','Hypertension','binary'], ['diabetes','Diabetes','binary'],
  ['totChol','Total cholesterol (mg/dL)','number'],
  ['sysBP','Systolic BP (mmHg)','number'], ['diaBP','Diastolic BP (mmHg)','number'],
  ['BMI','BMI','number'], ['heartRate','Heart rate (bpm)','number'],
  ['glucose','Glucose (mg/dL)','number']
]
function App() {
  const [form,setForm] = useState(initial)
  const [result,setResult] = useState(null)
  const [error,setError] = useState('')
  const [loading,setLoading] = useState(false)
  const [apiStatus,setApiStatus] = useState('Checking API…')
  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(d => setApiStatus(d.model_loaded ? 'API connected · XGBoost loaded' : 'API online · Model unavailable'))
      .catch(() => setApiStatus('API not connected'))
  }, [])
  function change(key,value) { setForm(p => ({...p,[key]: key==='sex' ? value : Number(value)})) }
  async function submit(e) {
    e.preventDefault(); setLoading(true); setError(''); setResult(null)
    try {
      const response = await fetch(`${API}/predict`, {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(form)
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction request failed')
      setResult(data)
    } catch (err) { setError(err.message || 'Could not reach the prediction API') }
    finally { setLoading(false) }
  }
  return <main className="shell">
    <header className="top"><div className="brand"><span className="brand-mark">♥</span><span>VOIS <i>×</i> AICTE</span></div><span className="status"><b className={apiStatus.includes('connected')?'online':'offline'} />{apiStatus}</span></header>
    <section className="hero"><p className="eyebrow">MACHINE LEARNING · HEALTHCARE ANALYTICS</p><h1>Cardiovascular Risk<br/><em>Prediction</em></h1><p className="intro">An academic binary-classification demonstration using the Framingham cardiovascular risk dataset.</p></section>
    <section className="layout">
      <form className="panel" onSubmit={submit}>
        <div className="panel-heading"><div><p className="eyebrow">PARTICIPANT DATA</p><h2>Enter the input features</h2></div><span className="count">14 inputs</span></div>
        <div className="fields">{fields.map(([key,label,type]) => <label className="field" key={key}><span>{label}</span>
          {type==='select' ? <select value={form[key]} onChange={e=>change(key,e.target.value)}><option value="F">Female</option><option value="M">Male</option></select>
          : type==='binary' ? <select value={form[key]} onChange={e=>change(key,e.target.value)}><option value={0}>No</option><option value={1}>Yes</option></select>
          : <input type="number" min="0" step="any" value={form[key]} onChange={e=>change(key,e.target.value)} required />}
        </label>)}</div>
        <div className="actions"><button disabled={loading}>{loading?'Generating result…':'Run prediction'} <span>→</span></button><button className="reset" type="button" onClick={()=>{setForm(initial);setResult(null);setError('')}}>Reset</button></div>
        {error && <p className="error">{error}</p>}
      </form>
      <aside className="result-panel">
        <p className="eyebrow">MODEL OUTPUT</p><h2>Prediction result</h2>
        {!result && !error && <div className="empty"><div className="pulse">✳</div><p>Your model output will appear here.</p><small>Complete the form and run a prediction.</small></div>}
        {result && <div className="output"><div className={`result-icon ${result.prediction===1?'flag':'ok'}`}>{result.prediction===1?'!':'✓'}</div><h3>{result.prediction_label}</h3><p className="muted">Predicted class: <strong>{result.prediction}</strong></p><div className="prob"><div><span>Class 1 model score</span><strong>{(result.probability_class_1*100).toFixed(1)}%</strong></div><div className="bar"><i style={{width:`${result.probability_class_1*100}%`}} /></div></div><p className="muted small">Model: {result.model} · {result.model_version}</p><p className="disclaimer">{result.disclaimer}</p></div>}
      </aside>
    </section>
    <footer><strong>Important safety notice</strong><p>This is an educational prototype, not a medical device or clinical risk calculator. Model scores are not calibrated clinical probabilities. Do not use this output for diagnosis, treatment, or medication decisions.</p><span>Project by Farhan Abid · VOIS For Tech × AICTE</span></footer>
  </main>
}
createRoot(document.getElementById('root')).render(<App />)
