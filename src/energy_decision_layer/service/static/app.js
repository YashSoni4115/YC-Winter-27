"use strict";
const $ = (id) => document.getElementById(id);
let token = "", busy = false, latest = null, permissions = {}, lastSuccess = 0;
let evaluationReport = null, evaluationSignature = "";
const pretty = (v) => JSON.stringify(v, null, 2);
const number = (v, n = 2) => typeof v === "number" ? v.toLocaleString(undefined, {maximumFractionDigits:n}) : "—";
async function api(path, options = {}) {
  const response = await fetch(path, {cache:"no-store", ...options, headers:{"Content-Type":"application/json", ...(token ? {Authorization:`Bearer ${token}`} : {}), ...(options.headers || {})}});
  const result = await response.json();
  if (!response.ok) throw Error(`${response.status}: ${typeof result.detail === "string" ? result.detail : pretty(result.detail)}`);
  return result;
}
function error(message) { $("error").hidden = !message; $("error").textContent = message || ""; }
function controls() { $("dispatch").disabled = busy || !permissions.can_dispatch; $("override").disabled = busy || !permissions.can_override; $("recommend").disabled = busy; }
async function operation(work) { if (busy) return; busy = true; controls(); error(""); try { await work(); } catch (e) { error(e.message); } finally { busy = false; controls(); } }
function text(tag, value) { const node = document.createElement(tag); node.textContent = value; return node; }
function overview(state, metrics) {
  $("timestamp").textContent = `${state.site_id} · ${state.timestamp_utc} · interval ${state.step_index} · ${state.timestep_seconds} s cadence`;
  const cost = Object.values(metrics.policies).reduce((sum, p) => sum + p.operating_cost_usd, 0);
  const fields = [["Demand", `${number(state.workload_demand_compute_units_per_h)} units/h`], ["Completed compute", `${number(state.completed_compute_units)} units`], ["Compute power", `${number(state.compute_power_mw)} MW`], ["Electricity price", `${number(state.electricity_price_usd_per_mwh)} USD/MWh`], ["Grid import / export", `${number(state.grid_power_mw)} MW`], ["Battery SOC", `${number(state.battery_state_of_charge * 100)} %`], ["Battery (+ discharge)", `${number(state.battery_power_mw)} MW`], ["Generator", `${state.generator_is_on ? "On" : "Off"} · ${number(state.generator_power_mw)} MW`], ["Cooling power", `${number(state.cooling_power_mw)} MW`], ["Site temperature", `${number(state.facility_temperature_c)} °C`], ["Ambient temperature", `${number(state.ambient_temperature_c)} °C`], ["Operating cost to date", `${number(cost)} USD`]];
  $("overview").replaceChildren(...fields.map(([label, value]) => { const el = text("div", ""); el.className="card"; el.append(text("span", label),text("strong", value)); return el; }));
}
function forecastCharts(bundle) {
  $("forecast-meta").textContent = `${bundle.model_name} ${bundle.model_version} · issued ${bundle.issued_at_utc} · ${bundle.provenance.is_synthetic ? "Synthetic/fixture model or fallback" : "Recorded data"} · shaded area = prediction interval`;
  const signals = [["workload_demand_compute_units_per_h","Compute demand · units/h"],["ambient_temperature_c","Ambient temperature · °C"],["electricity_price_usd_per_mwh","Electricity price · USD/MWh"]];
  $("forecasts").replaceChildren(...signals.map(([key, title]) => {
    const card = text("div", ""); card.className="chart"; card.append(text("h3", title));
    const points=bundle.points.map(p=>{const interval=p.prediction_intervals.find(i=>i.signal===key);return {v:p[key],lo:interval?.lower_value??p[key],hi:interval?.upper_value??p[key],coverage:interval?.coverage_probability};});
    const lo=Math.min(...points.map(p=>p.lo)), hi=Math.max(...points.map(p=>p.hi)), span=hi-lo||1;
    const x=i=>15+i*270/Math.max(1,points.length-1), y=v=>130-(v-lo)*110/span;
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox","0 0 300 150");svg.setAttribute("role","img");svg.setAttribute("aria-label",title);
    function line(type, coords, attrs){const el=document.createElementNS(svg.namespaceURI,type);el.setAttribute("points",coords);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);svg.append(el);}
    line("polygon", [...points.map((p,i)=>`${x(i)},${y(p.hi)}`),...points.map((p,i)=>`${x(i)},${y(p.lo)}`).reverse()].join(" "),{fill:"#5cddbb",opacity:".18"});
    line("polyline",points.map((p,i)=>`${x(i)},${y(p.v)}`).join(" "),{fill:"none",stroke:"#5cddbb","stroke-width":"2"});card.append(svg,text("p",`${number(lo)} – ${number(hi)} · coverage ${points[0].coverage ? number(points[0].coverage*100)+"%" : "unavailable"}`),text("small",`${bundle.valid_from_utc} → ${bundle.valid_until_utc}`));return card;
  }));
}
function history(items) {
  if (!items.length) { const tr=document.createElement("tr"), td=text("td","No dispatch decisions yet.");td.colSpan=6;tr.append(td);$("history").replaceChildren(tr);return; }
  $("history").replaceChildren(...items.map(item=>{const tr=document.createElement("tr");tr.tabIndex=0;[item.timestamp_utc,item.policy,item.response.status,item.status,item.response.reason_message,number(item.outcome?.operating_cost)].forEach(v=>tr.append(text("td",v??"—")));const inspect=()=>$("detail").textContent=pretty(item);tr.onclick=inspect;tr.onkeydown=e=>{if(e.key==="Enter")inspect();};return tr;}));
}
function metricsTable(metrics) {
  const entries=Object.entries(metrics.policies);if(!entries.length){$("metrics").textContent="No executed decisions yet.";return;}
  const table=document.createElement("table"), head=document.createElement("tr");["Policy","Steps","Cost USD","USD / compute","Missed compute","Battery MWh","Generator MWh","Thermal violations","Constraint events"].forEach(v=>head.append(text("th",v)));table.append(head);
  entries.forEach(([name,m])=>{const tr=document.createElement("tr");[name,m.executed_steps,number(m.operating_cost_usd),number(m.cost_per_completed_compute_usd,4),number(m.deadline_missed_compute_units),number(m.battery_throughput_mwh),number(m.generator_energy_mwh),m.thermal_violations,m.constraint_violations].forEach(v=>tr.append(text("td",v)));table.append(tr);});$("metrics").replaceChildren(table);
}
function comparisonTable(columns, rows) {
  const table=document.createElement("table"), head=document.createElement("thead"), heading=document.createElement("tr"), body=document.createElement("tbody");
  columns.forEach(value=>heading.append(text("th",value)));head.append(heading);table.append(head,body);
  rows.forEach(values=>{const row=document.createElement("tr");values.forEach(value=>row.append(text("td",value??"—")));body.append(row);});return table;
}
function renderPolicyComparison() {
  const records=evaluationReport?.["metrics.json"];
  if(!Array.isArray(records)||!records.length){$("evaluation-comparison").textContent="No policy comparison available. Run the documented evaluation command.";return;}
  const scenario=$("evaluation-scenario").value, selected=records.filter(item=>item.scenario===scenario);
  const rows=selected.map(item=>[item.policy,number(item.total_operating_cost),number(item.compute_throughput),number(item.deadline_violations),number(item.cost_per_compute,4),number(item.terminal_battery_energy_mwh),number(item.terminal_temperature_c),item.steps,item.failed?`Stopped: ${item.failure_reason||"see raw report"}`:"Completed"]);
  $("evaluation-comparison").replaceChildren(comparisonTable(["Policy","Cost USD","Completed compute","Missed compute","USD / compute","Final battery MWh","Final site °C","Steps","Outcome"],rows));
}
function renderEvaluation(report) {
  const signature=pretty(report);if(signature===evaluationSignature)return;evaluationSignature=signature;evaluationReport=report;$("evaluation").textContent=signature;
  const summary=report["ml-evaluation.json"], training=summary?.forecast_training, records=report["metrics.json"];
  const scenarios=Array.isArray(records)?[...new Set(records.map(item=>item.scenario))]:[], previous=$("evaluation-scenario").value;
  $("evaluation-scenario").replaceChildren(...scenarios.map(name=>{const option=text("option",name);option.value=name;return option;}));
  $("evaluation-scenario").disabled=!scenarios.length;$("evaluation-scenario").value=scenarios.includes(previous)?previous:scenarios.includes("normal")?"normal":scenarios[0]||"";
  $("evaluation-scope").textContent=summary?`${summary.synthetic?"Synthetic simulation evaluation":"Recorded evaluation"} · seed ${summary.seed??"unavailable"} · configured ${summary.steps??"unavailable"} steps per scenario. Policies share the saved scenario and site configuration.`:"No saved evaluation metadata available.";
  renderPolicyComparison();
  const labels={workload_demand_compute_units_per_h:"Compute demand · units/h",ambient_temperature_c:"Ambient temperature · °C",electricity_price_usd_per_mwh:"Electricity price · USD/MWh"};
  const signals=training?.signals?Object.entries(training.signals):[];
  if(!signals.length){$("forecast-accuracy").textContent="No held-out forecast metrics available. Train the forecasting models and run evaluation.";return;}
  $("forecast-accuracy-scope").textContent=`${training.data_kind||"Unspecified"} training data · untouched chronological test set · nominal interval coverage ${number(training.coverage_probability*100)}%. MAE and RMSE use each signal’s displayed units. ${training.interval_limitations||""}`;
  const rows=signals.filter(([,entry])=>entry.test).map(([signal,entry])=>{const m=entry.test;return [labels[signal]||signal,number(m.mae,3),number(m.rmse,3),typeof m.coverage==="number"?`${number(m.coverage*100)}%`:"—",number(m.persistence?.mae,3),number(m.seasonal_naive?.mae,3)];});
  $("forecast-accuracy").replaceChildren(comparisonTable(["Signal / units","Test MAE","Test RMSE","Interval coverage","Persistence MAE","Seasonal-naive MAE"],rows));
}
async function refresh() {
  const [state, forecast, decisions, metrics, policy] = await Promise.all([api("/api/state"),api("/api/forecasts"),api("/api/decisions?limit=30"),api("/api/metrics"),api("/api/policies")]);
  latest=state;permissions=policy;lastSuccess=Date.now();overview(state,metrics);forecastCharts(forecast);history(decisions.items);metricsTable(metrics);
  const selected=$("policy").value;$("policy").replaceChildren(...policy.available.map(name=>{const option=text("option",name);option.value=name;return option;}));$("policy").value=policy.available.includes(selected)?selected:policy.selected;
  $("permissions").textContent=`${policy.role} access · ${policy.can_dispatch?"Simulated dispatch enabled":"Execution controls disabled"}${policy.scenario_finished?" · Scenario finished":""}`;$("connection").textContent="Connected · simulation paused between steps";controls();
  api("/api/evaluation").then(renderEvaluation).catch(e=>{$("evaluation-scope").textContent=`Evaluation unavailable: ${e.message}`;});
  api("/api/forecast-status").then(status=>{if(status.grid_stress_probabilities?.some(v=>v!==null))$("stress").textContent="Grid-stress probabilities: "+status.grid_stress_probabilities.map(v=>v===null?"unavailable":number(v*100)+"%").join(", ");}).catch(()=>{});
}
function showDecision(result) { $("proposal").textContent=pretty(result.proposal);$("verdict").textContent=pretty({executed:result.executed,safety:result.safety,outcome:result.outcome,error:result.error,surrogate_advisory:result.surrogate});if(result.proposal)$("manual").value=pretty(result.proposal); }
$("authenticate").onclick=()=>operation(async()=>{token=$("token").value;$("token").value="";await refresh();});
$("recommend").onclick=()=>operation(async()=>{showDecision(await api(`/api/recommendations/${$("policy").value}`));});
$("dispatch").onclick=()=>operation(async()=>{const result=await api("/api/dispatch",{method:"POST",headers:{"Idempotency-Key":crypto.randomUUID()},body:pretty({policy:$("policy").value})});showDecision(result);await refresh();});
$("validate").onclick=()=>operation(async()=>{const result=await api("/api/safety/validate",{method:"POST",body:pretty(JSON.parse($("manual").value))});$("verdict").textContent=pretty({executed:false,safety:result});});
$("override").onclick=()=>operation(async()=>{const result=await api("/api/manual-overrides",{method:"POST",headers:{"Idempotency-Key":crypto.randomUUID()},body:pretty(JSON.parse($("manual").value))});showDecision(result);await refresh();});
$("policy").onchange=()=>operation(async()=>{if(permissions.can_dispatch)await api("/api/policies/select",{method:"POST",body:pretty({policy:$("policy").value})});});
$("evaluation-scenario").onchange=renderPolicyComparison;
operation(refresh);setInterval(()=>{if(!busy)operation(refresh);if(lastSuccess&&Date.now()-lastSuccess>30000)$("connection").textContent="Stale · backend has not responded for 30 s";},10000);
