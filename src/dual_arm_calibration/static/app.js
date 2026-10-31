"use strict";
const baseURL = new URL(".", location.href);
const apiURL = path => new URL(path.replace(/^\//, ""), baseURL).href;
const $ = selector => document.querySelector(selector);
let state = null;
let draftLoaded = false;
let submitting = false;
const labels = {left:"左侧", right:"右侧", top:"固定顶相机"};

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function route() {
  const page = ["workflow","integration","offline","results","hardware"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "workflow";
  document.querySelectorAll(".page").forEach(section => {section.hidden = section.id !== `page-${page}`;});
  document.querySelectorAll("nav a").forEach(a => {
    if (a.dataset.page === page) a.setAttribute("aria-current","page");
    else a.removeAttribute("aria-current");
  });
}
window.addEventListener("hashchange",route);
route();

function field(name, label) {
  const wrap = el("label",label,"field");
  const input = document.createElement("input");
  input.name = name; input.type = "text"; input.placeholder = "待接入"; input.maxLength = 2000;
  wrap.append(input); return wrap;
}
function formGroup(title, fields, note) {
  const group = el("div",undefined,"form-group"); group.append(el("h2",title));
  if (note) group.append(el("p",note,"subtle"));
  const grid = el("div",undefined,"form-grid");
  fields.forEach(([name,label]) => grid.append(field(name,label)));
  group.append(grid); $("#draft-fields").append(group);
}
for (const arm of ["left","right"]) formGroup(`${labels[arm]}机械臂`,[
  [`arms.${arm}.adapter`,"适配器"],[`arms.${arm}.model_id`,"模型 / 安装身份"],[`arms.${arm}.connection`,"连接配置"]]);
for (const role of ["left","right","top"]) formGroup(role === "top" ? labels.top : `${labels[role]}腕相机`,[
  [`cameras.${role}.adapter`,"相机适配器"],[`cameras.${role}.serial`,"相机序列号"],[`cameras.${role}.profile`,"图像配置 / 参数来源"]]);
formGroup("标定板",[
  ["board.squares_x","横向方格数"],["board.squares_y","纵向方格数"],["board.dictionary","标记字典"],
  ["board.square_length_mm","方格边长 · mm"],["board.marker_length_mm","标记边长 · mm"],["board.thickness_mm","图案到桌面的距离 · mm"]
],"规格保持留空；接入时按实际标定板填写并核验。");

async function request(path, body) {
  const response = await fetch(apiURL(path),body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}
function loadDraft(draft) {
  for (const input of $("#draft-form").elements) {
    if (!input.name) continue;
    let value = draft;
    for (const key of input.name.split(".")) value = value?.[key];
    input.value = value ?? "";
  }
  draftLoaded = true;
}
function readDraft() {
  const draft = {arms:{left:{},right:{}},cameras:{left:{},right:{},top:{}},board:{},notes:null};
  for (const input of $("#draft-form").elements) {
    if (!input.name) continue;
    const keys = input.name.split("."); let parent = draft;
    for (const key of keys.slice(0,-1)) parent = parent[key];
    parent[keys.at(-1)] = input.value.trim() || null;
  }
  return draft;
}
$("#draft-form").addEventListener("submit",async event => {
  event.preventDefault(); const button = event.submitter; button.disabled = true;
  try {await request("/api/draft",readDraft()); $("#draft-feedback").textContent = "草稿已保存 · 硬件仍待接入";}
  catch(error) {$("#draft-feedback").textContent = error.message;}
  finally {button.disabled = false;}
});

function renderJobs() {
  const container = $("#jobs"); container.replaceChildren();
  const jobs = state.jobs;
  if (!jobs.length) {
    const empty = el("div",undefined,"empty"); empty.append(el("h2","还没有标定结果"),el("p","先运行离线示例，或导入已有的设备配置与观测数据。"));
    const link = el("a","前往离线验证 →"); link.href="#offline"; empty.append(link); container.append(empty);
  }
  const statusNames = {queued:"等待求解",running:"正在求解",succeeded:"验证通过",failed:"求解失败"};
  for (const job of jobs) {
    const card = el("article",undefined,"job"); card.dataset.jobId = job.id;
    const head = el("div",undefined,"job-head"); head.append(el("h2",job.kind === "demo" ? "离线合成示例" : job.kind === "hardware" ? "Piper 实测标定" : "导入观测数据"),el("span",statusNames[job.status] || job.status,`pill ${job.status === "succeeded" ? "success" : job.status === "failed" ? "failed" : "pending"}`));
    card.append(head,el("p",new Date(job.created).toLocaleString(),"job-time"));
    if (job.status === "succeeded") {
      const s = job.summary; const metrics = el("div",undefined,"metrics");
      for (const [value,label] of [[s.world_frame === "table" ? "桌面" : "顶相机","公共坐标系"],[`${s.max_translation_mm.toFixed(3)} mm`,"最大平移偏差"],[`${s.max_rotation_deg.toFixed(3)}°`,"最大旋转偏差"]]) {
        const metric = el("div",undefined,"metric"); metric.append(el("strong",value),el("small",label)); metrics.append(metric);
      }
      card.append(metrics,el("p",s.table_calibrated ? "桌面标定已完成，桌面高度为 z = 0。" : "已跳过桌面标定，桌面高度未定义。","job-note"));
      if (job.kind === "demo") card.append(el("p","结果来自合成数据，用于检查软件流程，不代表硬件精度。","job-note"));
      const actions = el("div",undefined,"actions");
      for (const [name,label] of [["calibration.json","下载标定 JSON"],["bundle.zip","下载完整数据包"]]) {
        const link = el("a",label,"button secondary"); link.href=apiURL(`/api/jobs/${job.id}/${name}`); link.setAttribute("download",""); actions.append(link);
      }
      card.append(actions);
    } else if (job.error) card.append(el("p",job.error,"job-error"));
    else card.append(el("p","正在执行手眼求解、顶相机桥接与独立验证，完成后可下载结果。","job-note"));
    container.append(card);
  }
}
function busy() {return submitting || state?.jobs.some(j => ["queued","running"].includes(j.status));}
function controls() {$("#run-demo").disabled = !!busy(); $("#run-import").disabled = !!busy();}
async function refresh() {
  try {
    const next = await request("/api/state");
    const changed = !state || JSON.stringify(state.jobs) !== JSON.stringify(next.jobs);
    state = next;
    renderHardware();
    if (!draftLoaded) loadDraft(state.draft);
    if (changed) renderJobs();
    $("#connection-error").hidden = true;
  } catch(error) {
    $("#connection-error").textContent = `工作台连接失败：${error.message}。请确认服务仍在运行。`;
    $("#connection-error").hidden = false;
  }
  controls();
}
async function runJob(payload) {
  submitting = true; controls(); const feedback = $("#offline-feedback"); feedback.classList.remove("error");
  feedback.textContent = "正在提交离线求解…";
  try {await request("/api/jobs",payload); feedback.textContent = "任务已提交，可在标定结果中查看进度。"; await refresh(); location.hash="results";}
  catch(error) {feedback.textContent=error.message; feedback.classList.add("error");}
  finally {submitting=false; controls();}
}
$("#run-demo").addEventListener("click",() => runJob({kind:"demo",table:$("#demo-table").checked}));
$("#import-form").addEventListener("submit",async event => {
  event.preventDefault(); const feedback=$("#offline-feedback");
  try {
    const rigFile=$("#rig-file").files[0], dataFile=$("#data-file").files[0];
    if (rigFile.size+dataFile.size > 15*1024*1024) throw new Error("两份文件合计应小于 15 MiB。");
    const rig=JSON.parse(await rigFile.text()),data=JSON.parse(await dataFile.text());
    await runJob({kind:"import",rig,data});
  } catch(error) {feedback.textContent=`导入失败：${error.message}`; feedback.classList.add("error");}
});
refresh(); setInterval(refresh,1200);

let hardwareWorking=false;
let previewTimer=0;
function renderHardware() {
  const hw=state?.hardware;
  if (!hw?.configured) return;
  document.body.classList.add('hardware-on');
  $('#hardware-nav').hidden=false;
  const badge=$('header .pill'); badge.textContent=hw.connected?'设备在线':'设备连接异常';
  $('#side-hardware-label').textContent=hw.connected?'设备在线':'设备连接异常';
  document.querySelectorAll('.rig-strip small').forEach(node=>{node.textContent=hw.connected?'已接入':'等待设备';});
  badge.className=`pill ${hw.connected?'success':'pending'}`;
  $('#open-hardware').disabled=!hw.connected;
  $('#integration-status').textContent='已通过 RoboBench 接入双 Piper 与三台 D435i。请进入硬件标定，确认板规格并新建流程。';
  const error=$('#hardware-error'); error.hidden=!hw.error; error.textContent=hw.error||'';
  const b=hw.board;
  $('#hardware-board').textContent=`已确认板规格：${b.squares_x} × ${b.squares_y} 方格，方格 ${b.square_length_m*1000} mm，标记 ${b.marker_length_m*1000} mm，${b.dictionary}；板厚 ${hw.board_thickness_mm??'未配置'} mm。`;
  for (const role of ['left','right','top']) {
    const c=hw.devices?.cameras?.[role];
    const node=document.querySelector(`[data-identity="${role}"]`);
    node.textContent=c?`${c.serial} · ${c.intrinsics.width} × ${c.intrinsics.height}`:'等待相机';
  }
  const s=hw.session;
  for(const option of $('#hardware-batch').options) {
    option.disabled=!!s && (option.value==='table'&&!s.table_enabled || option.value.startsWith('intrinsics-')&&s.modes[option.value.split('-')[1]]!=='fit');
  }
  if($('#hardware-batch').selectedOptions[0]?.disabled) $('#hardware-batch').value=Array.from($('#hardware-batch').options).find(o=>!o.disabled).value;
  $('#hardware-session').textContent=s?`当前流程 ${s.id.slice(0,8)}${s.sealed?' · 已提交最终求解':''}`:'尚未新建流程';
  const batch=$('#hardware-batch').value;
  const count=s?.counts[batch]??0;
  let note=s?`已采集 ${count} 组${s.frozen.includes(batch)?' · 已通过并冻结':''}`:'请先新建流程';
  const role=batch.replace('intrinsics-','');
  const cov=s?.coverage?.[role];
  if(cov && batch.startsWith('intrinsics-')) note+=`。覆盖反馈：${cov.issues.length?cov.issues.join('；'):'已达标'}`;
  $('#batch-status').textContent=note;
  for(const name of ['capture','fit','finish','resume','reset']) $('#hardware-'+name).disabled=hardwareWorking||!s||!hw.connected;
  if(s){
    for(const name of ['capture','fit','reset']) $('#hardware-'+name).disabled ||= s.sealed||s.frozen.includes(batch);
    $('#hardware-finish').disabled ||= s.sealed||!s.frozen.includes('validation')||s.table_enabled&&!s.frozen.includes('table');
    $('#hardware-resume').disabled ||= !s.sealed||!state.jobs.some(j=>j.id===s.job_id&&j.status==='failed');
  }
  $('#hardware-new').disabled=hardwareWorking||!hw.connected;
  if (Date.now()-previewTimer>2200 && location.hash==='#hardware') {
    previewTimer=Date.now();
    for (const image of document.querySelectorAll('[data-camera]')) image.src=apiURL(`/api/hardware/preview/${image.dataset.camera}.jpg?t=${previewTimer}`);
  }
}
$('#open-hardware').addEventListener('click',()=>{location.hash='hardware'; renderHardware();});
$('#hardware-batch').addEventListener('change',renderHardware);
window.addEventListener('hashchange',()=>{if(state)renderHardware();});
async function hardwareAction(payload) {
  hardwareWorking=true; renderHardware();
  const feedback=$('#hardware-feedback'); feedback.classList.remove('error');
  feedback.textContent=payload.op==='capture'?'正在检查静止状态、采集原始图像…':'正在处理…';
  try {
    const result=await request('/api/hardware/action',payload);
    feedback.textContent=payload.op==='capture'?'已保存实测位姿与原始图像。':payload.op==='fit'?'该阶段已通过并冻结。':'操作完成。';
    await refresh();
    if(result.job_id)location.hash='results';
  } catch(error){feedback.textContent=error.message;feedback.classList.add('error');}
  finally{hardwareWorking=false;renderHardware();}
}
$('#hardware-new').addEventListener('click',()=>hardwareAction({op:'new',board_confirmed:$('#board-confirmed').checked,table:$('#hardware-table').checked,modes:Object.fromEntries(['left','right','top'].map(r=>[r,$('#mode-'+r).value]))}));
$('#hardware-capture').addEventListener('click',()=>hardwareAction({op:'capture',batch:$('#hardware-batch').value,stationary_confirmed:$('#stationary-confirmed').checked}));
$('#hardware-fit').addEventListener('click',()=>hardwareAction({op:'fit',batch:$('#hardware-batch').value}));
$('#hardware-reset').addEventListener('click',()=>hardwareAction({op:'reset',batch:$('#hardware-batch').value}));
$('#hardware-finish').addEventListener('click',()=>hardwareAction({op:'finish'}));
$('#hardware-resume').addEventListener('click',()=>hardwareAction({op:'resume'}));
