import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const API = (import.meta.env.VITE_API_BASE_URL || 'https://vois-nov-ai-ml.onrender.com').replace(/\/$/, '')
const initial = {
  age: 50, education: 2, sex: 'F', cigsPerDay: 0, BPMeds: 0,
  prevalentStroke: 0, prevalentHyp: 0, diabetes: 0, totChol: 220,
  sysBP: 120, diaBP: 80, BMI: 25, heartRate: 72, glucose: 85
}
const fields = [
  ['age','Age','number','Age in years'], ['education','Education level','number','Level 1 to 4'],
  ['sex','Sex','select','Choose the participant category'], ['cigsPerDay','Cigarettes per day','number','Enter 0 if not a smoker'],
  ['BPMeds','Blood pressure medication','binary','Is the participant taking medication?'],
  ['prevalentStroke','Previous stroke','binary','Has the participant had a stroke?'],
  ['prevalentHyp','Hypertension','binary','Has hypertension been diagnosed?'], ['diabetes','Diabetes','binary','Has diabetes been diagnosed?'],
  ['totChol','Total cholesterol','number','mg/dL'],
  ['sysBP','Systolic blood pressure','number','Upper BP value, in mmHg'], ['diaBP','Diastolic blood pressure','number','Lower BP value, in mmHg'],
  ['BMI','BMI','number','Body mass index'], ['heartRate','Heart rate','number','Beats per minute'],
  ['glucose','Glucose','number','mg/dL']
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
  const resultTitle = result?.prediction === 1 ? 'CHD risk class detected' : 'No CHD risk class detected'
  return <main className="shell">
    <header className="top"><div className="brand"><span className="brand-mark">♥</span><span>VOIS <i>×</i> AICTE</span></div><span className="status"><b className={apiStatus.includes('connected')?'online':'offline'} />{apiStatus}</span></header>
    <section className="hero"><p className="eyebrow">MACHINE LEARNING · HEALTHCARE ANALYTICS</p><h1>Cardiovascular Risk<br/><em>Prediction</em></h1><p className="intro">Enter the participant's basic health information to run the academic XGBoost model. This tool supports learning and demonstration only.</p></section>
    <section className="layout">
      <form className="panel" onSubmit={submit}>
        <div className="panel-heading"><div><p className="eyebrow">PARTICIPANT DATA</p><h2>Enter health information</h2><p className="panel-subtitle">Complete all 14 fields, then run the prediction.</p></div><span className="count">14 fields</span></div>
        <div className="form-group-title">Personal and lifestyle information</div>
        <div className="fields">{fields.slice(0, 9).map(([key,label,type,hint]) => <label className="field" key={key}><span>{label}</span>
          {type==='select' ? <select value={form[key]} onChange={e=>change(key,e.target.value)}><option value="F">Female</option><option value="M">Male</option></select>
          : type==='binary' ? <select value={form[key]} onChange={e=>change(key,e.target.value)}><option value={0}>No</option><option value={1}>Yes</option></select>
          : <input type="number" min="0" step="any" value={form[key]} onChange={e=>change(key,e.target.value)} required />}
          <small>{hint}</small>
        </label>)}</div>
        <div className="form-group-title clinical-title">Clinical measurements</div>
        <div className="fields">{fields.slice(9).map(([key,label,type,hint]) => <label className="field" key={key}><span>{label}</span>
          <input type="number" min="0" step="any" value={form[key]} onChange={e=>change(key,e.target.value)} required />
          <small>{hint}</small>
        </label>)}</div>
        <div className="actions"><button disabled={loading}>{loading?'Generating result…':'Run prediction'} <span>→</span></button><button className="reset" type="button" onClick={()=>{setForm(initial);setResult(null);setError('')}}>Reset</button></div>
        {error && <p className="error">{error}</p>}
      </form>
      <aside className="result-panel">
        <p className="eyebrow">MODEL OUTPUT</p><h2>How to read the result</h2>
        {!result && !error && <div className="empty"><div className="pulse">✳</div><p>Complete the form and select “Run prediction.”</p><small>The result will indicate a model class and score.</small></div>}
        {result && <div className="output"><div className={`result-icon ${result.prediction===1?'flag':'ok'}`}>{result.prediction===1?'!':'✓'}</div><p className="result-kicker">Predicted result</p><h3>{resultTitle}</h3><p className="muted">Model class: <strong>{result.prediction}</strong> · <strong>{result.prediction === 1 ? 'Risk class' : 'No-risk class'}</strong></p><div className="prob"><div><span>Risk-class model score</span><strong>{(result.probability_class_1*100).toFixed(1)}%</strong></div><div className="bar"><i style={{width:`${result.probability_class_1*100}%`}} /></div></div><p className="result-explanation">{result.prediction === 1 ? 'The model assigned this participant to the TenYearCHD risk class.' : 'The model assigned this participant to the no-risk class.'}</p><p className="muted small">Model: {result.model} · {result.model_version}</p><p className="disclaimer"><strong>Important:</strong> This is a model score, not a clinical diagnosis or calibrated probability of disease.</p></div>}
      </aside>
    </section>
    <footer><strong>Important safety notice</strong><p>This is an educational prototype, not a medical device or clinical risk calculator. Model scores are not calibrated clinical probabilities. Do not use this output for diagnosis, treatment, or medication decisions.</p><span>Project by Farhan Abid · VOIS For Tech × AICTE</span></footer>
  </main>
}
createRoot(document.getElementById('root')).render(<App />)
