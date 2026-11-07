/* Optional depth calibration. No device connection until an explicit start. */
(()=>{
 const get=id=>document.getElementById('depth-'+id);let audio=null,lastId=null,lastCount=null,pending=false;
 function soundState(){
  const enabled=audio?.state==='running';
  get('sound').textContent=enabled?'再次试音':'开启提示音并试音';
  get('sound-status').textContent=enabled?'声音已启用：每保存一组响两声。若试音听不到，请检查此网站是否静音及电脑的音量和输出设备。':'提示音未启用或已暂停，请点击试音。刷新页面后需要重新开启。';
 }
 async function enableSound(){
  try{
   if(!audio||audio.state==='closed'){audio=new (window.AudioContext||window.webkitAudioContext)();audio.onstatechange=soundState}
   await audio.resume();soundState();beep();
  }catch(e){get('sound-status').textContent=`无法启用声音：${e.message}。可继续根据保存计数采集。`}
 }
 get('sound').onclick=enableSound;
 function beep(){
  if(audio?.state!=='running'){soundState();return}
  for(const delay of [0,.32]){
   const o=audio.createOscillator(),g=audio.createGain(),t=audio.currentTime+delay;
   o.connect(g);g.connect(audio.destination);o.frequency.value=880;
   g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(.25,t+.02);g.gain.setValueAtTime(.25,t+.18);g.gain.linearRampToValueAtTime(0,t+.23);
   o.onended=()=>{o.disconnect();g.disconnect()};o.start(t);o.stop(t+.24);
  }
 }
 for(const op of ['start','resume','stop','solve'])get(op).onclick=async()=>{try{if(op==='start'||op==='resume')await enableSound();await request('/api/depth/action',{op});await update()}catch(e){get('message').textContent=e.message}};
 function comparison(c){
  const box=get('comparison');box.replaceChildren();if(!c){box.textContent='暂无对比记录';return}
  box.append(el('p',`板面参考高度：${c.expected_board_height_mm.toFixed(1)} mm。下表为各帧区域中位数的均值。`));
  const table=el('table');const header=el('tr');['投射器','方法','板高 mm','偏差 mm','帧间 σ mm'].forEach(x=>header.append(el('th',x)));table.append(header);
  const names={native_raw:'相机原生',native_filtered:'相机 + SDK 滤波',fast_foundation_stereo:'FFS · 原参数',ffs_factory_intrinsics_refit_extrinsics:'FFS · 重标双目外参',ffs_fit_intrinsics_radial2:'FFS · 重标内外参',ffs_factory_intrinsics_refit_extrinsics_refit_color:'FFS · 双目外参 + RGB 外参',ffs_fit_intrinsics_radial2_refit_color:'FFS · 双目内外参 + RGB 外参'};
  names.ffs_fixed_K_radial2='FFS · 固定焦距主点，仅拟合畸变及双目外参';
  names.ffs_fixed_K_radial2_refit_color='FFS · 固定焦距主点 + RGB 外参';
  for(const r of c.summary){const tr=el('tr');[r.mode,names[r.method]||r.method,r.board_height_mm.mean.toFixed(2),(r.board_height_mm.mean-c.expected_board_height_mm).toFixed(2),r.board_height_mm.std.toFixed(2)].forEach(x=>tr.append(el('td',x)));table.append(tr)}box.append(table);
  if(c.independent_validation)box.append(el('p','新标定结果使用采集前保存的独立图像验证，未拟合高度偏移；仍沿用既有桌面坐标和相机外参。'));
  for(const mode of ['emitter_on','emitter_off']){
   const rows=c.summary.filter(r=>r.mode===mode);if(!rows.length)continue;
   box.append(el('h3',`衣服区域 · ${mode} · 高度 mm`));const t=el('table'),h=el('tr');h.append(el('th','区域'));rows.forEach(r=>h.append(el('th',names[r.method]||r.method)));t.append(h);
   for(const name of Object.keys(rows[0].cloth_height_mm)){const tr=el('tr');tr.append(el('td',name));rows.forEach(r=>tr.append(el('td',r.cloth_height_mm[name]?.mean?.toFixed(1)??'无数据')));t.append(tr)}box.append(t);
  }
 }
 async function update(){
  if(pending)return;pending=true;
  try{const s=(await request('/api/state')).depth;
   if(!s)return;
   const c=s.capture||{};get('hint').textContent=s.solve?.state==='solving'?'正在拟合候选':c.hint||'准备好后开始双目采集';get('count').textContent=`已保存 ${c.count||0} / ${c.target||30} 组 · 左右共同角点 ${c.corners||0}`+(c.motion_rms_px!=null?` · 板位移 ${c.motion_rms_px.toFixed(2)} / ${c.max_motion_rms_px} px · 停稳 ${c.stable_s} / ${c.required_stable_s} 秒`:'');
   if(s.active&&c.remaining_s!=null)get('count').textContent+=` · 本轮剩余 ${c.remaining_s} 秒`;
   if(c.board_distance_m!=null)get('count').textContent+=` · 板距相机约 ${(c.board_distance_m*100).toFixed(0)} cm`;
   get('resume').disabled=s.active||!(c.count>0)||c.count>=(c.target||30);
   get('start').disabled=s.active;get('stop').disabled=!s.active;get('solve').disabled=s.active||(c.count||0)<25;
   if(s.id===lastId&&lastCount!==null&&c.count>lastCount)beep();lastId=s.id;lastCount=c.count||0;
   if(s.id){get('preview').hidden=false;if(s.active)get('preview').src=apiURL('/api/depth/preview.jpg')+'?t='+Date.now()}
   const result=s.result;
   get('result-summary').textContent=result?`${result.training_ids.length} 组拟合、${result.heldout_ids.length} 组留出验证；候选 ${result.selected}，右图重投影 RMS ${result.candidate.heldout_right_rms_px.toFixed(3)} px。${result.height_validation?.hint||'仍需独立高度验证。'}${result.color_extrinsics_fit?.hint||''}`:s.solve?.hint||'尚未拟合';
   if(result?.fixed_K_comparison)get('result-summary').textContent+=result.fixed_K_comparison.hint;
   get('result').textContent=result?JSON.stringify(result,null,2):'尚未拟合';
   if(s.interrupted)get('message').textContent='上次采集意外中断，请核对相机恢复状态；原始样本已保留。';
   comparison(s.comparison);
  }catch(e){get('message').textContent=e.message}finally{pending=false}
 }
 setInterval(()=>{if(location.hash==='#depth')update()},1000);window.addEventListener('hashchange',update);update();
})();
