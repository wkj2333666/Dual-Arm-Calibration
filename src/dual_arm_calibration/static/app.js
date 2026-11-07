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
  const page = ["workflow","integration","offline","results","hardware","depth"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "workflow";
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
  event.preventDefault();
  if(state?.hardware?.configured)return;
  const button = event.submitter; button.disabled = true;
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
    if (!draftLoaded) loadDraft(state.draft);
    renderHardware();
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
let hardwareSetupSession=null;
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
  const step=$('#hardware-step-state');
  step.textContent=hw.connected?'已接入':'已配置 · 连接异常';
  step.className=`pill ${hw.connected?'success':'pending'}`;
  $('#integration-description').textContent='当前实例已加载 Piper 现场配置。以下为正在使用的设备映射和标定板规格，只读显示；修改接入配置需更新现场配置文件。';
  $('#save-draft').hidden=true;
  $('#draft-feedback').textContent=hw.connected?'现场配置已加载 · 设备在线':'现场配置已加载 · '+(hw.error||'等待设备');
  $('#integration-notice strong').textContent='Piper 硬件接入已完成';
  $('#integration-notice p').textContent='双臂反馈、三相机预览与原始图像采集均通过 RoboBench 设备服务接入。请在“硬件标定”中继续操作。通用未配置实例仍使用空白接入草稿。';
  $('#draft-fields .form-group:last-child .subtle').textContent='已加载现场确认的标定板规格。';
  const values={
    'board.squares_x':b.squares_x,'board.squares_y':b.squares_y,
    'board.square_length_mm':b.square_length_m*1000,'board.marker_length_mm':b.marker_length_m*1000,
    'board.dictionary':b.dictionary,'board.thickness_mm':hw.board_thickness_mm,
    notes:'当前显示实际加载的现场配置；空白接入草稿仅用于尚未配置硬件的实例。'
  };
  for(const arm of ['left','right']){
    const a=hw.devices?.arms?.[arm];
    values[`arms.${arm}.adapter`]='RoboBench / Piper';
    values[`arms.${arm}.model_id`]=a?`${a.serial} · ${hw.devices.model_sha256}`:'已配置，等待设备反馈';
    values[`arms.${arm}.connection`]=a?.channel||'由 RoboBench 设备服务提供';
  }
  for(const role of ['left','right','top']){
    const c=hw.devices?.cameras?.[role];
    values[`cameras.${role}.adapter`]='RoboBench / RealSense D435i';
    values[`cameras.${role}.serial`]=c?.serial||'已配置，等待设备反馈';
    values[`cameras.${role}.profile`]=c?`${c.intrinsics.width} × ${c.intrinsics.height} RGB · 厂家原始参数`:'已配置，等待设备反馈';
  }
  for(const input of $('#draft-form').elements){
    if(!input.name)continue;
    const value=String(values[input.name]??'未配置');
    if(input.value!==value)input.value=value;
    input.readOnly=true;
  }
  $('#hardware-board').textContent=`已确认板规格：${b.squares_x} × ${b.squares_y} 方格，方格 ${b.square_length_m*1000} mm，标记 ${b.marker_length_m*1000} mm，${b.dictionary}；板厚 ${hw.board_thickness_mm??'未配置'} mm。`;
  for (const role of ['left','right','top']) {
    const c=hw.devices?.cameras?.[role];
    const node=document.querySelector(`[data-identity="${role}"]`);
    node.textContent=c?`${c.serial} · ${c.intrinsics.width} × ${c.intrinsics.height}`:'等待相机';
  }
  const s=hw.session;
  if(s&&hardwareSetupSession!==s.id){
    hardwareSetupSession=s.id;
    for(const r of ['left','right','top'])$('#mode-'+r).value=s.modes[r];
    $('#hardware-table').checked=s.table_enabled;
  }
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
  renderAutomatic(hw);
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
  feedback.textContent=payload.op==='capture'?'正在检查静止状态、采集原始图像…':payload.op==='estimate'?'正在读取当前图像并估算板位姿…':'正在处理…';
  try {
    const result=await request('/api/hardware/action',payload);
    feedback.textContent=payload.op==='capture'?'已保存实测位姿与原始图像。':payload.op==='fit'?'该阶段已通过并冻结。':payload.op==='auto-plan'?'正在后台生成并检查轨迹。':payload.op==='auto-start'?'自动采集已启动，可随时停止并保持。':'操作完成。';
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

let previewId=null,previewData=null,viewYaw=-.7,viewPitch=.6,dragOrigin=null;
function renderAutomatic(hw){
 const a=hw.automatic||{state:'idle'},s=hw.session;
 const running=['planning','starting','running','stopping'].includes(a.state);
 const batch=$('#hardware-batch').value;
 const wristBatch=/^(intrinsics|wrists)-(left|right)$/.test(batch);
 const valid=hw.connected&&s&&!s.sealed&&wristBatch&&!s.frozen.includes(batch);
 const names={idle:'先估算当前板位姿，再生成轨迹。',estimated:'板位姿已估算，可生成采集轨迹。',planning:`正在生成并检查轨迹 · ${Math.round((a.progress||0)*100)}%`,preview:'轨迹已生成，请旋转预览并核验各点位。',starting:'正在复核起点与板位姿…',running:`自动采集 ${a.index||0}/${a.total||0} · 已保存 ${a.saved||0} · 跳过 ${a.skips?.length||0} · ${a.phase==='capturing'?'停稳采集中':a.phase==='awaiting_capture'?'等待取帧':'运动中'}`,completed:`自动采集结束 · 已保存 ${a.saved||0} · 跳过 ${a.skips?.length||0}，请计算并检查覆盖。`,cancelled:'自动采集已停止，已保存样本保留。',stopping:'已请求停止并保持当前位置。',failed:'自动流程失败：'+a.error,interrupted:a.error};
 $('#automatic-status').textContent=names[a.state]||a.state;
 $('#hardware-estimate').disabled=hardwareWorking||!valid||running;
 $('#hardware-plan').disabled=hardwareWorking||!valid||running||!a.estimate||a.batch!==batch;
 $('#hardware-auto-start').disabled=hardwareWorking||!valid||a.state!=='preview'||a.batch!==batch||!$('#trajectory-confirmed').checked;
 $('#hardware-stop').disabled=!running&&a.state!=='preview';
 if(running){for(const name of ['capture','fit','reset','finish','resume','new'])$('#hardware-'+name).disabled=true;}
 const e=a.estimate;
 $('#pose-estimate').textContent=e?`板在${e.arm==='left'?'左':'右'}基座中：XYZ ${e.board_position_m.map(v=>(v*1000).toFixed(1)).join(' / ')} mm；XYZ 角 ${e.board_rotation_deg.map(v=>v.toFixed(1)).join(' / ')}°；${e.corner_count} 个角点，重投影 ${e.reprojection_error_px.toFixed(3)} px。`:'';
 $('#planning-seed-note').textContent=e?.seed?.warning||'';
 if(e)$('#pose-estimate').textContent+=e.top_reference?' 固定板由顶相机独立跟踪。':' 固定板跟踪需已验证腕外参或顶相机可见。';
 const p=a.state==='interrupted'?null:a.preview;$('#trajectory-preview').hidden=!p;
 if(p&&previewId!==p.id){previewId=p.id;previewData=p;$('#trajectory-confirmed').checked=false;$('#trajectory-point').max=p.waypoints.length-1;$('#trajectory-point').value=0;drawTrajectory();}
 $('#trajectory-summary').textContent=p?`${p.waypoints.length} 个采集姿态 · 预计 ${Math.ceil(p.duration_s/60)} 分钟 · 关节速度上限 ${p.max_speed_deg_s}°/s · 路径模型最小间隙 ${p.min_clearance_mm.toFixed(1)} mm`:'';
}
function drawTrajectory(){
 const p=previewData;if(!p)return;
 const canvas=$('#trajectory-canvas'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
 const index=Number($('#trajectory-point').value),point=p.waypoints[index],B=p.estimate.base_from_board;
 const width=p.board.squares_x*p.board.square_length_m,height=p.board.squares_y*p.board.square_length_m;
 const board=[[0,0,0],[width,0,0],[width,height,0],[0,height,0]].map(x=>[0,1,2].map(i=>B[i][3]+x.reduce((s,v,j)=>s+B[i][j]*v,0)));
 const camera=q=>q.base_from_camera.slice(0,3).map(row=>row[3]);
 const raw=[...board,...point.links_base,...(p.other_links_base||[]),...p.waypoints.map(camera)];
 const rotate=v=>{const x=Math.cos(viewYaw)*v[0]-Math.sin(viewYaw)*v[1],y=Math.sin(viewYaw)*v[0]+Math.cos(viewYaw)*v[1];return[x,Math.sin(viewPitch)*y-Math.cos(viewPitch)*v[2]];};
 const values=raw.map(rotate),lo=[0,1].map(i=>Math.min(...values.map(x=>x[i]))),hi=[0,1].map(i=>Math.max(...values.map(x=>x[i])));
 const scale=Math.min((w-100)/Math.max(.1,hi[0]-lo[0]),(h-80)/Math.max(.1,hi[1]-lo[1]));
 const project=v=>{const r=rotate(v);return[w/2+(r[0]-(lo[0]+hi[0])/2)*scale,h/2+(r[1]-(lo[1]+hi[1])/2)*scale];};
 ctx.clearRect(0,0,w,h);ctx.font='13px sans-serif';
 const line=(pts,color,size=3,close=false)=>{if(!pts.length)return;ctx.beginPath();pts.map(project).forEach((x,i)=>i?ctx.lineTo(...x):ctx.moveTo(...x));if(close)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=size;ctx.stroke();};
 line(p.other_links_base||[],'#acb5b1',11);line(point.links_base,'#176b55',9);line([...p.waypoints.map(camera)],'#719c88',2);line(board,'#be922b',4,true);
 p.waypoints.forEach((q,i)=>{const xy=project(camera(q));ctx.beginPath();ctx.arc(...xy,i===index?7:3,0,Math.PI*2);ctx.fillStyle=i===index?'#b45b24':'#176b55';ctx.fill();});
 const origin=project([0,0,0]);for(const [v,color,label] of [[[.1,0,0],'#b44','X'],[[0,.1,0],'#497','Y'],[[0,0,.1],'#57b','Z']]){const xy=project(v);line([[0,0,0],v],color,2);ctx.fillStyle=color;ctx.fillText(label,xy[0]+4,xy[1]);}ctx.fillStyle='#68756d';ctx.fillText('拖动旋转视角 · 坐标单位 m',18,24);
 $('#trajectory-joints').textContent=`点位 ${index+1}/${p.waypoints.length} · ${point.visible_corners} 个预计可见角点 · 关节角 ${point.joints_deg.map(v=>v.toFixed(1)+'°').join(' / ')}`;
}
$('#trajectory-canvas').addEventListener('pointerdown',e=>{dragOrigin=[e.clientX,e.clientY,viewYaw,viewPitch];e.target.setPointerCapture(e.pointerId);});
$('#trajectory-canvas').addEventListener('pointermove',e=>{if(!dragOrigin)return;viewYaw=dragOrigin[2]+(e.clientX-dragOrigin[0])*.01;viewPitch=Math.max(-1.4,Math.min(1.4,dragOrigin[3]+(e.clientY-dragOrigin[1])*.01));drawTrajectory();});
$('#trajectory-canvas').addEventListener('pointerup',()=>{dragOrigin=null;});
$('#trajectory-point').addEventListener('input',drawTrajectory);
$('#trajectory-confirmed').addEventListener('change',()=>{if(state)renderHardware();});
$('#hardware-estimate').addEventListener('click',()=>hardwareAction({op:'estimate',batch:$('#hardware-batch').value}));
$('#hardware-plan').addEventListener('click',()=>hardwareAction({op:'auto-plan',batch:$('#hardware-batch').value}));
$('#hardware-auto-start').addEventListener('click',()=>hardwareAction({op:'auto-start',batch:$('#hardware-batch').value,confirmed:$('#trajectory-confirmed').checked}));
$('#hardware-stop').addEventListener('click',async()=>{try{await request('/api/hardware/action',{op:'auto-stop'});await refresh();}catch(e){$('#hardware-feedback').textContent=e.message;}});
