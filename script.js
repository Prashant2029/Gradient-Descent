let N=40, noise=0.3, trueM=1.8, trueB=0.5;
let pts=[], m=0, b=0, lr=0.01, stepsPerEpoch=10;
let lossHistory=[], path3d=[], running=false, raf=null, epoch=0;

// ─── data & math ─────────────────────────────────────────────────────────────
function genData(){
  pts=[];
  for(let i=0;i<N;i++){
    const x=-1.4+2.8*Math.random();
    const y=trueM*x+trueB+(Math.random()-0.5)*2*noise;
    pts.push([x,y]);
  }
}
function mse(mm,bb){
  let s=0; for(const[x,y] of pts){const d=mm*x+bb-y;s+=d*d;} return s/pts.length;
}
function grad(mm,bb){
  let gm=0,gb=0;
  for(const[x,y] of pts){const d=mm*x+bb-y;gm+=2*d*x;gb+=2*d;}
  return[gm/pts.length,gb/pts.length];
}
function doStep(){
  const[gm,gb]=grad(m,b);
  m-=lr*gm; b-=lr*gb;
}

// ─── one full epoch = stepsPerEpoch gradient steps ───────────────────────────
function runEpoch(){
  for(let i=0;i<stepsPerEpoch;i++) doStep();
  epoch++;
  const loss=mse(m,b);
  lossHistory.push(loss);
  path3d.push([m,b,loss]);
  if(path3d.length>400) path3d.shift();
  updateStats();
  logEpoch(epoch, loss);
  drawScatter();
  updateBall();
  render3D();
  drawLossChart();
}

// ─── UI helpers ──────────────────────────────────────────────────────────────
function updateStats(){
  document.getElementById('sv').textContent=m.toFixed(4);
  document.getElementById('bv').textContent=b.toFixed(4);
  document.getElementById('lv').textContent=mse(m,b).toFixed(5);
  document.getElementById('iv').textContent=epoch;
}
function logEpoch(ep, loss){
  const log=document.getElementById('epoch-log');
  const prev=lossHistory[lossHistory.length-2];
  const delta=prev!=null?(loss-prev).toFixed(5):'—';
  const converged=prev!=null&&Math.abs(loss-prev)<1e-6;
  const entry=document.createElement('div');
  entry.className='entry';
  entry.innerHTML=`<span class="ep">ep ${String(ep).padStart(4,'0')}</span>  loss=<span class="val">${loss.toFixed(5)}</span>  Δ=${delta}` + (converged?`  <span class="converge">✓ converged</span>`:'');
  log.appendChild(entry);
  log.scrollTop=log.scrollHeight;
}

// ─── scatter canvas ───────────────────────────────────────────────────────────
const sc=document.getElementById('scatter');
const sctx=sc.getContext('2d');
function drawScatter(){
  const dpr=devicePixelRatio;
  const W=sc.clientWidth*dpr, H=sc.clientHeight*dpr;
  sc.width=W; sc.height=H;
  const pad=36*dpr;
  const toX=v=>(v+1.6)/3.2*(W-2*pad)+pad;
  const toY=v=>H-pad-(v+2)/4.5*(H-2*pad);

  sctx.fillStyle='#0f0f0f'; sctx.fillRect(0,0,W,H);

  // grid
  sctx.strokeStyle='#2a2a2a'; sctx.lineWidth=0.5*dpr;
  for(let v=-1.5;v<=1.5;v+=0.5){
    sctx.beginPath();sctx.moveTo(toX(-1.6),toY(v));sctx.lineTo(toX(1.6),toY(v));sctx.stroke();
    sctx.beginPath();sctx.moveTo(toX(v),toY(-2));sctx.lineTo(toX(v),toY(2.5));sctx.stroke();
  }
  // axes
  sctx.strokeStyle='#505050'; sctx.lineWidth=1*dpr;
  sctx.beginPath();sctx.moveTo(toX(-1.6),toY(0));sctx.lineTo(toX(1.6),toY(0));sctx.stroke();
  sctx.beginPath();sctx.moveTo(toX(0),toY(-2));sctx.lineTo(toX(0),toY(2.5));sctx.stroke();

  // true line (dim)
  sctx.strokeStyle='rgba(180,180,180,0.32)'; sctx.lineWidth=1.5*dpr;
  sctx.setLineDash([4*dpr,4*dpr]);
  sctx.beginPath();sctx.moveTo(toX(-1.6),toY(trueM*(-1.6)+trueB));sctx.lineTo(toX(1.6),toY(trueM*1.6+trueB));sctx.stroke();
  sctx.setLineDash([]);

  // data points
  for(const[x,y] of pts){
    // residual line
    const ypred=m*x+b;
    sctx.strokeStyle='rgba(210,210,210,0.2)'; sctx.lineWidth=0.8*dpr;
    sctx.beginPath();sctx.moveTo(toX(x),toY(y));sctx.lineTo(toX(x),toY(ypred));sctx.stroke();
    // dot
    sctx.beginPath();sctx.arc(toX(x),toY(y),3.5*dpr,0,2*Math.PI);
    sctx.fillStyle='#d7d7d7'; sctx.fill();
  }

  // fitted line
  sctx.strokeStyle='#ffffff'; sctx.lineWidth=2.2*dpr;
  sctx.beginPath();sctx.moveTo(toX(-1.6),toY(m*(-1.6)+b));sctx.lineTo(toX(1.6),toY(m*1.6+b));sctx.stroke();

  // labels
  sctx.font=`${9*dpr}px JetBrains Mono,monospace`;
  sctx.fillStyle='#9a9a9a'; sctx.textAlign='right';
  sctx.fillText('true',toX(1.55),toY(trueM*1.55+trueB)-4*dpr);
  sctx.fillStyle='#f0f0f0'; sctx.fillText('fit',toX(1.55),toY(m*1.55+b)-4*dpr);
}

// ─── loss history chart ───────────────────────────────────────────────────────
const lc=document.getElementById('loss-chart');
const lctx=lc.getContext('2d');
function drawLossChart(){
  const dpr=devicePixelRatio;
  const W=lc.clientWidth*dpr, H=lc.clientHeight*dpr;
  lc.width=W; lc.height=H;
  lctx.fillStyle='#131313'; lctx.fillRect(0,0,W,H);
  if(lossHistory.length<2) return;
  const pad=20*dpr;
  const mn=Math.min(...lossHistory), mx=Math.max(...lossHistory);
  const range=mx-mn||1;
  const toX=i=>(i/(lossHistory.length-1))*(W-2*pad)+pad;
  const toY=v=>H-pad-(v-mn)/range*(H-2*pad);

  lctx.strokeStyle='rgba(255,255,255,0.12)';lctx.lineWidth=0.5*dpr;
  for(let i=0;i<=4;i++){
    const y=pad+i*(H-2*pad)/4;
    lctx.beginPath();lctx.moveTo(pad,y);lctx.lineTo(W-pad,y);lctx.stroke();
  }

  lctx.strokeStyle='#f1f1f1'; lctx.lineWidth=1.5*dpr;
  lctx.beginPath();
  lossHistory.forEach((v,i)=>{ i===0?lctx.moveTo(toX(i),toY(v)):lctx.lineTo(toX(i),toY(v)); });
  lctx.stroke();

  // last value
  lctx.font=`${9*dpr}px JetBrains Mono,monospace`;
  lctx.fillStyle='#f1f1f1'; lctx.textAlign='right';
  lctx.fillText(lossHistory[lossHistory.length-1].toFixed(4),W-pad,pad+10*dpr);
}

// ─── THREE.js loss surface ────────────────────────────────────────────────────
let renderer,scene,camera,ballMesh,lineGeo,linePts3d,lineObj;
function init3D(){
  const c=document.getElementById('loss3d');
  const w=c.clientWidth, h=c.clientHeight;
  renderer=new THREE.WebGLRenderer({canvas:c,antialias:true,alpha:false});
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(w,h);
  renderer.setClearColor(0x0a0a0a,1);
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(45,w/h,0.01,100);
  camera.position.set(6,4.5,6); camera.lookAt(0,0,0);

  // surface geometry
  const GRID=55, RANGE=3.2;
  const geo=new THREE.PlaneGeometry(RANGE*2,RANGE*2,GRID,GRID);
  const pos=geo.attributes.position;
  for(let i=0;i<pos.count;i++){
    pos.setZ(i, Math.min(mse(pos.getX(i),pos.getY(i)),7)*0.45);
  }
  geo.computeVertexNormals();
  const mat=new THREE.MeshLambertMaterial({color:0x252525,side:THREE.DoubleSide,transparent:true,opacity:0.9});
  const mesh=new THREE.Mesh(geo,mat); mesh.rotation.x=-Math.PI/2; scene.add(mesh);

  // wireframe
  const wgeo=new THREE.PlaneGeometry(RANGE*2,RANGE*2,GRID,GRID);
  const wpos=wgeo.attributes.position;
  for(let i=0;i<wpos.count;i++) wpos.setZ(i,Math.min(mse(wpos.getX(i),wpos.getY(i)),7)*0.45+0.005);
  wgeo.computeVertexNormals();
  const wmesh=new THREE.Mesh(wgeo,new THREE.MeshBasicMaterial({color:0x5c5c5c,wireframe:true,opacity:0.32,transparent:true}));
  wmesh.rotation.x=-Math.PI/2; scene.add(wmesh);

  // lights
  scene.add(new THREE.AmbientLight(0xffffff,0.6));
  const dl=new THREE.DirectionalLight(0xffffff,0.9); dl.position.set(5,8,5); scene.add(dl);

  // ball
  const bGeo=new THREE.SphereGeometry(0.1,16,16);
  ballMesh=new THREE.Mesh(bGeo,new THREE.MeshLambertMaterial({color:0xffffff,emissive:0x444444,emissiveIntensity:0.36}));
  scene.add(ballMesh);

  // path line
  linePts3d=[new THREE.Vector3(m, Math.min(mse(m,b),7)*0.45+0.12, b)];
  lineGeo=new THREE.BufferGeometry().setFromPoints(linePts3d);
  lineMat=new THREE.LineBasicMaterial({color:0xe8e8e8,linewidth:2});
  lineObj=new THREE.Line(lineGeo,lineMat); scene.add(lineObj);

  updateBall();
  setupOrbit();
}
function updateBall(){
  const loss=Math.min(mse(m,b),7)*0.45+0.12;
  ballMesh.position.set(m,loss,b);
  linePts3d.push(new THREE.Vector3(m,loss,b));
  if(linePts3d.length>500) linePts3d.shift();
  lineGeo.setFromPoints(linePts3d);
}
function render3D(){ renderer.render(scene,camera); }

function resize3D(){
  if(!renderer||!camera) return;
  const c=document.getElementById('loss3d');
  const w=c.clientWidth, h=c.clientHeight;
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(w,h,false);
  camera.aspect=w/h;
  camera.updateProjectionMatrix();
  render3D();
}

function setupOrbit(){
  const c=document.getElementById('loss3d');
  let active=false,last={x:0,y:0},theta=-0.8,phi=1.0,radius=9;
  c.addEventListener('mousedown',e=>{active=true;last={x:e.clientX,y:e.clientY}});
  window.addEventListener('mousemove',e=>{
    if(!active)return;
    const dx=e.clientX-last.x, dy=e.clientY-last.y;
    theta-=dx*0.012; phi=Math.max(0.15,Math.min(1.5,phi-dy*0.012));
    last={x:e.clientX,y:e.clientY};
    camera.position.set(radius*Math.sin(phi)*Math.sin(theta),radius*Math.cos(phi),radius*Math.sin(phi)*Math.cos(theta));
    camera.lookAt(0,0,0); render3D();
  });
  window.addEventListener('mouseup',()=>active=false);
  c.addEventListener('wheel',e=>{
    radius=Math.max(3,Math.min(20,radius+e.deltaY*0.01));
    camera.position.normalize().multiplyScalar(radius);
    camera.lookAt(0,0,0); render3D();
  },{passive:true});
  // touch support
  let t0=null;
  c.addEventListener('touchstart',e=>{if(e.touches.length===1){active=true;last={x:e.touches[0].clientX,y:e.touches[0].clientY};}},{passive:true});
  c.addEventListener('touchmove',e=>{
    if(!active||e.touches.length!==1)return;
    const dx=e.touches[0].clientX-last.x, dy=e.touches[0].clientY-last.y;
    theta-=dx*0.015; phi=Math.max(0.15,Math.min(1.5,phi-dy*0.015));
    last={x:e.touches[0].clientX,y:e.touches[0].clientY};
    camera.position.set(radius*Math.sin(phi)*Math.sin(theta),radius*Math.cos(phi),radius*Math.sin(phi)*Math.cos(theta));
    camera.lookAt(0,0,0); render3D();
  },{passive:true});
  c.addEventListener('touchend',()=>active=false);
}

// ─── reset helpers ────────────────────────────────────────────────────────────
function resetParams(){
  m=(-1+2*Math.random())*2.5;
  b=(-1+2*Math.random())*2.5;
  epoch=0; lossHistory=[]; path3d=[];
  linePts3d=[new THREE.Vector3(m,Math.min(mse(m,b),7)*0.45+0.12,b)];
  lineGeo.setFromPoints(linePts3d);
  document.getElementById('epoch-log').innerHTML='';
  logEpoch(0, mse(m,b)); epoch=0; // reset counter after log
  document.getElementById('iv').textContent='0';
  updateStats(); updateBall(); drawScatter(); render3D(); drawLossChart();
}

// ─── auto-run loop ────────────────────────────────────────────────────────────
function loop(){
  if(!running)return;
  runEpoch();
  raf=requestAnimationFrame(loop);
}

function setRunButtonState(isRunning){
  const runBtn=document.getElementById('run-btn');
  const label=runBtn.querySelector('.btn-label');
  const playIcon=runBtn.querySelector('.icon-play');
  const pauseIcon=runBtn.querySelector('.icon-pause');
  runBtn.classList.toggle('is-running', isRunning);
  runBtn.setAttribute('aria-pressed', String(isRunning));
  label.textContent=isRunning?'pause':'run';
  playIcon.classList.toggle('is-hidden', isRunning);
  pauseIcon.classList.toggle('is-hidden', !isRunning);
}

// ─── wire up controls ─────────────────────────────────────────────────────────
document.getElementById('run-btn').addEventListener('click',()=>{
  running=!running;
  setRunButtonState(running);
  if(running) loop();
  else if(raf) cancelAnimationFrame(raf);
});
document.getElementById('step-btn').addEventListener('click',()=>{
  if(running){ running=false; setRunButtonState(false); if(raf)cancelAnimationFrame(raf); }
  runEpoch();
});
document.getElementById('reset-btn').addEventListener('click',()=>{
  running=false; setRunButtonState(false);
  if(raf)cancelAnimationFrame(raf);
  resetParams();
});
document.getElementById('regen-btn').addEventListener('click',()=>{
  running=false; setRunButtonState(false);
  if(raf)cancelAnimationFrame(raf);
  genData(); resetParams();
});
document.getElementById('lr').addEventListener('input',e=>{
  lr=Math.pow(10,parseFloat(e.target.value));
  document.getElementById('lr-out').textContent=lr.toFixed(4);
});
document.getElementById('sp').addEventListener('input',e=>{
  stepsPerEpoch=parseInt(e.target.value);
  document.getElementById('sp-out').textContent=stepsPerEpoch;
});
document.getElementById('ns').addEventListener('input',e=>{
  noise=parseFloat(e.target.value);
  document.getElementById('ns-out').textContent=noise.toFixed(2);
});

// ─── init ──────────────────────────────────────────────────────────────────────
genData();
requestAnimationFrame(()=>{
  init3D();
  drawScatter();
  drawLossChart();
  updateStats();
  setRunButtonState(false);
  render3D();
});
window.addEventListener('resize',()=>{ drawScatter(); drawLossChart(); resize3D(); });