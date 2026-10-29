"""
可视化生成器：把算法输出绘制为单文件交互式 HTML，风格与论文配图（4.2 轨迹图）一致。

风格取自论文绘图脚本：
    路网        #79706E
    跑道        #4D4A49（加粗）
    轨迹点      #2CA02C（绿点，每 3 点取 1，与论文一致）
    识别路径    #FF7F0E（橙）
    潜在R       #1F77B4（蓝）
    候选路径    #BBBBBB（浅灰）
    参照路线    #00BFFF（天蓝虚线）
    端点        白色填充 + 黑色描边
    机坪/航站楼  灰阶 #D8D8D8...#A8A8A8 / 纯黑
    字体        Times New Roman + 宋体

交互：
    画布旋转（默认 30°，0-360° 可调）、滚轮缩放、拖拽平移、双击复位、
    图层开关（右侧图例同步只显示已勾选图层）、悬停查看路径名、导出 PNG。

用法：
    python visualize.py
    python visualize.py --out output/viewer.html --max-candidates 60
"""

import os
import json
import argparse

import config


# ==============================
# 论文配色（与 4.2 绘图脚本一致）
# ==============================
APRON_COLORS = ["#D8D8D8", "#CFCFCF", "#C6C6C6", "#BCBCBC", "#B2B2B2", "#A8A8A8"]
APRON_GEOJSON = os.path.join(config.DATA_DIR, "XiAn_Apron_Terminal_Segmentation.geojson")

# 跑道路径对（取自论文脚本 RUNWAYS）。
# 注意：数据里跑道被切成十几段首尾相接的路径（中间段如 N-244/N-242/... 并不在本表中），
# 若只按本表绘制，跑道上就只剩首尾两截有粗线。因此这里仅用它们定位"跑道的 z 值"，
# 实际绘制由 build_runways() 把同 z 的共线段合并成整条。
RUNWAY_SEEDS = [("N-5", "N-7"), ("N-9", "N-11"), ("N-4", "N-3")]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================
# 路网
# ==============================
def build_network():
    """pathname 列表 + 折线数组 + 长度（坐标保留 0.1，z 取反使北向上）"""
    data = load_json(config.INIT_MATRIX_PATH)

    names, lines, lengths = [], [], []

    for p in data["paths"]:
        names.append(p["pathname"])
        lengths.append(p.get("pathLength", 0))
        flat = []
        for pt in p["points"]:
            flat.append(round(pt["x"], 1))
            flat.append(round(-pt["z"], 1))
        lines.append(flat)

    return {
        "names": names,
        "lines": lines,
        "lengths": lengths,
        "runways": build_runways(names, lines)
    }


def build_runways(names, lines, z_tol=3.0, dz_tol=2.0, min_seg=30, gap=50):
    """
    把跑道补全为整条：以 RUNWAY_SEEDS 的 z 值为种子，收集同一 z 上的水平共线段，
    合并 x 区间后取最长连续段，输出 [x0, z, x1, z]。

    跑道中段在数据里被切成很多小段（N-244、N-242、N-240 ...），只画种子路径会只剩首尾两截。
    """
    idx = {n: i for i, n in enumerate(names)}

    seeds = []
    for pair in RUNWAY_SEEDS:
        for nm in pair:
            i = idx.get(nm)
            if i is None:
                continue
            zs = lines[i][1::2]
            if zs:
                seeds.append(sum(zs) / len(zs))

    runways, used = [], []

    for z0 in seeds:
        if any(abs(z0 - u) <= z_tol for u in used):
            continue

        spans = []
        for flat in lines:
            if len(flat) < 4:
                continue
            xs, zs = flat[0::2], flat[1::2]
            if max(zs) - min(zs) > dz_tol:      # 只取水平段
                continue
            zc = sum(zs) / len(zs)
            if abs(zc - z0) > z_tol:
                continue
            if max(xs) - min(xs) < min_seg:
                continue
            spans.append((min(xs), max(xs), zc))

        if not spans:
            continue

        spans.sort()
        merged = []
        for x0, x1, _zc in spans:
            if merged and x0 <= merged[-1][1] + gap:
                merged[-1][1] = max(merged[-1][1], x1)
            else:
                merged.append([x0, x1])

        a, b = max(merged, key=lambda m: m[1] - m[0])
        z = sum(s[2] for s in spans) / len(spans)
        runways.append([round(a, 1), round(z, 1), round(b, 1), round(z, 1)])
        used.append(z0)

    runways.sort(key=lambda r: r[1])            # 由南到北
    return runways


# ==============================
# 机坪 / 航站楼底图
# ==============================
def build_background():
    if not os.path.exists(APRON_GEOJSON):
        return []

    geo = load_json(APRON_GEOJSON)

    a, b, c, d = config.AFFINE_A, config.AFFINE_B, config.AFFINE_C, config.AFFINE_D
    tx, tz = config.AFFINE_TX, config.AFFINE_TZ

    polys = []

    for feat in geo.get("features", []):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue

        if props.get("region_type") == "terminal":
            color = "#000000"
        else:
            try:
                cluster = int(float(props.get("cluster") or 0))
            except (TypeError, ValueError):
                cluster = 0
            color = APRON_COLORS[cluster % len(APRON_COLORS)]

        pts = []
        for lon, lat in geom["coordinates"][0]:
            x = a * lon + b * lat + tx
            z = c * lon + d * lat + tz
            pts.append(round(x, 1))
            pts.append(round(-z, 1))

        polys.append({"color": color, "pts": pts})

    return polys


# ==============================
# 航班结果
# ==============================
def build_flights(output_dir, max_candidates=60):
    conv_dir = os.path.join(output_dir, "flight_data_converted")
    r_dir = os.path.join(output_dir, "1-潜在R识别")
    best_dir = os.path.join(output_dir, "4-最优路径")

    if not os.path.isdir(best_dir):
        return []

    flights = []

    for file_name in sorted(os.listdir(best_dir)):
        if not (file_name.startswith("best-") and file_name.endswith(".json")):
            continue

        converted_name = file_name.replace("best-", "")
        base = os.path.splitext(converted_name)[0]
        flight_id = base.replace("_converted", "")

        conv_path = os.path.join(conv_dir, converted_name)
        r_path = os.path.join(r_dir, f"{base}-潜在R识别.json")
        best_path = os.path.join(best_dir, file_name)

        if not os.path.exists(conv_path):
            continue

        track = []
        for item in load_json(conv_path):
            props = item.get("properties", {})
            x = props.get("unity_x")
            z = props.get("unity_z")
            if x is not None and z is not None:
                track.append([round(x, 1), round(-z, 1)])

        potential = list(load_json(r_path).keys()) if os.path.exists(r_path) else []

        best = load_json(best_path)
        best_route = best.get("best_path") or []
        all_results = best.get("all_results") or []
        total_candidates = best.get("total_candidates", len(all_results))

        top = sorted(all_results, key=lambda r: r["score"])[:max_candidates]

        # 可选：参照路线（reference-<xxx>.json，诊断/对比用，不属于算法输出）
        ref_file = os.path.join(best_dir, f"reference-{converted_name}")
        reference = None
        if os.path.exists(ref_file):
            ref = load_json(ref_file)
            reference = {
                "route": ref.get("route", []),
                "score": ref.get("score"),
                "meanDev": ref.get("mean_dev")
            }

        flights.append({
            "id": flight_id,
            "track": track,
            "potential": potential,
            "best": best_route,
            "bestScore": best.get("best_score"),
            "totalCandidates": total_candidates,
            "candidates": [c["path"] for c in top],
            "reference": reference
        })

    return flights


# ==============================
# HTML 模板（论文风格）
# ==============================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Taxi Path Identification · 结果查看器</title>
<style>
  :root{
    --net:#79706E;
    --runway:#4D4A49;
    --track:#2CA02C;
    --best:#FF7F0E;
    --pot:#1F77B4;
    --cand:#BBBBBB;
    --ref:#00BFFF;
    --ink:#1A1A1A;
    --muted:#5A5A5A;
    --line:#111111;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:#fff;color:var(--ink);
    font-family:"Times New Roman","SimSun","Songti SC",serif;}
  #app{display:flex;height:100%;}

  #side{width:266px;flex:0 0 266px;background:#fff;border-right:1px solid var(--line);
    padding:18px 18px 14px;overflow-y:auto;}
  h1{font-size:15px;margin:0 0 2px;font-weight:700;letter-spacing:.2px;}
  .sub{font-size:11px;color:var(--muted);margin-bottom:16px;font-style:italic;}

  .cap{font-size:10px;letter-spacing:1.2px;color:var(--muted);text-transform:uppercase;
    margin:14px 0 6px;border-bottom:1px solid #DDD;padding-bottom:4px;}

  select,button{width:100%;padding:6px 8px;font-size:13px;border:1px solid var(--line);
    border-radius:2px;background:#fff;color:var(--ink);font-family:inherit;}
  button{cursor:pointer;}
  button:hover{background:#F4F4F4;}

  .layer{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:12.5px;cursor:pointer;}
  .layer input{accent-color:#1A1A1A;margin:0;}
  .swatch{width:16px;height:3px;border-radius:0;display:inline-block;flex:0 0 16px;}
  .swatch.dot{width:8px;height:8px;border-radius:50%;flex-basis:8px;}

  .stats{margin-top:6px;font-size:12px;line-height:1.85;color:var(--muted);}
  .stats b{color:var(--ink);font-weight:700;}
  .stats .ref{border-top:1px dotted #CCC;margin-top:4px;padding-top:4px;}

  .rot{display:flex;align-items:center;gap:8px;margin:2px 0 10px;}
  .rot input[type=range]{flex:1 1 auto;width:auto;padding:0;border:0;background:none;
    accent-color:#1A1A1A;height:14px;cursor:pointer;}
  .rot span{font-size:11.5px;color:var(--muted);min-width:36px;text-align:right;
    font-variant-numeric:tabular-nums;}

  .hint{margin-top:14px;font-size:10.5px;color:#8A8A8A;line-height:1.6;font-style:italic;}

  #wrap{position:relative;flex:1;min-width:0;padding:14px;background:#fff;}
  #frame{position:absolute;inset:14px;border:1px solid var(--line);pointer-events:none;}
  canvas{display:block;width:100%;height:100%;cursor:grab;}
  canvas.drag{cursor:grabbing;}

  #tooltip{position:absolute;pointer-events:none;padding:2px 6px;font-size:11.5px;
    background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:2px;
    transform:translate(-50%,-135%);display:none;white-space:nowrap;
    font-family:"Times New Roman","SimSun",serif;}

  #legend{position:absolute;right:26px;top:26px;background:#fff;border:1px solid var(--line);
    padding:8px 12px;font-size:11.5px;line-height:1.75;color:var(--ink);display:none;}
  #legend .t{font-size:10px;letter-spacing:1px;color:var(--muted);margin-bottom:3px;
    text-transform:uppercase;}
  #legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:7px;}
  #legend i.d{width:8px;height:8px;border-radius:50%;}
</style>
</head>
<body>
<div id="app">
  <div id="side">
    <h1>Taxi Path Identification</h1>
    <div class="sub">Xi'an Xianyang · GCCPG results</div>

    <div class="cap">Flight</div>
    <select id="flight"></select>

    <div class="cap">Layers</div>
    <label class="layer"><input type="checkbox" id="ly-bg" checked><span class="swatch" style="background:#C6C6C6"></span>Apron / Terminal</label>
    <label class="layer"><input type="checkbox" id="ly-net" checked><span class="swatch" style="background:var(--net)"></span>Taxiway network</label>
    <label class="layer"><input type="checkbox" id="ly-rwy" checked><span class="swatch" style="background:var(--runway)"></span>Runways</label>
    <label class="layer"><input type="checkbox" id="ly-node" checked><span class="swatch dot" style="background:#fff;border:1px solid #111"></span>Path endpoints</label>
    <label class="layer"><input type="checkbox" id="ly-pot" checked><span class="swatch" style="background:var(--pot)"></span>Potential R</label>
    <label class="layer"><input type="checkbox" id="ly-cand"><span class="swatch" style="background:var(--cand)"></span>Candidate paths</label>
    <label class="layer"><input type="checkbox" id="ly-track" checked><span class="swatch dot" style="background:var(--track)"></span>Trajectory points</label>
    <label class="layer"><input type="checkbox" id="ly-line"><span class="swatch" style="background:var(--track);opacity:.55"></span>Trajectory line</label>
    <label class="layer"><input type="checkbox" id="ly-best" checked><span class="swatch" style="background:var(--best)"></span>Identified path</label>
    <label class="layer"><input type="checkbox" id="ly-ref"><span class="swatch" style="background:var(--ref)"></span>Reference route</label>

    <div class="cap">Statistics</div>
    <div class="stats" id="stats"></div>

    <div class="cap">View</div>
    <div class="rot">
      <input type="range" id="rot" min="0" max="360" step="1" value="30">
      <span id="rotVal">30&deg;</span>
    </div>
    <button id="reset">Reset view</button>
    <div style="height:6px"></div>
    <button id="png">Export PNG</button>

    <div class="hint">Scroll to zoom &middot; drag to pan &middot; double-click to reset<br>Hover network to see path name</div>
  </div>

  <div id="wrap">
    <canvas id="cv"></canvas>
    <div id="frame"></div>
    <div id="tooltip"></div>
    <div id="legend"></div>
  </div>
</div>

<script>
const NET = __NET__;
const BG  = __BG__;
const FLIGHTS = __FLIGHTS__;

const NAME_IDX = new Map(NET.names.map((n,i)=>[n,i]));

const cv  = document.getElementById("cv");
const ctx = cv.getContext("2d");
const wrap= document.getElementById("wrap");
const tip = document.getElementById("tooltip");

let dpr = window.devicePixelRatio || 1;

// 视图：世界原点 -> 屏幕中心的旋转/缩放/平移
// X = ctr.x + ((x-ox)*c - (z-oz)*s) * scale
// Y = ctr.y + ((x-ox)*s + (z-oz)*c) * scale
const ctr = {x:0, y:0};
let view = { scale:1, ox:0, oz:0, rot:30*Math.PI/180, c:Math.cos(30*Math.PI/180), s:Math.sin(30*Math.PI/180) };

let current = null, bbox = null;

function syncRot(){
  view.c = Math.cos(view.rot);
  view.s = Math.sin(view.rot);
}

function w2s(x,z){
  const dx=(x-view.ox)*view.scale, dz=(z-view.oz)*view.scale;
  return [ctr.x + dx*view.c - dz*view.s, ctr.y + dx*view.s + dz*view.c];
}

function s2w(X,Y){
  const u=(X-ctr.x)/view.scale, v=(Y-ctr.y)/view.scale;
  return [view.ox + u*view.c + v*view.s, view.oz - u*view.s + v*view.c];
}

// 预构建路网与跑道 Path2D（世界坐标，绘制时 transform 缩放）
const netPath = new Path2D();
for (const l of NET.lines){
  if(l.length<4) continue;
  netPath.moveTo(l[0], l[1]);
  for(let i=2;i<l.length;i+=2) netPath.lineTo(l[i], l[i+1]);
}
// 跑道：整条（同 z 共线段已在 Python 端合并，不再只有首尾两截）
const rwyPath = new Path2D();
for (const r of (NET.runways || [])){
  rwyPath.moveTo(r[0], r[1]);
  rwyPath.lineTo(r[2], r[3]);
}

// ---------- 图例（跟随图层勾选动态生成） ----------
const LEGEND_ITEMS = [
  {id:"ly-bg",    label:"Apron / Terminal",  icon:'<i style="background:#C6C6C6"></i>'},
  {id:"ly-net",   label:"Taxiway",           icon:'<i style="background:#79706E"></i>'},
  {id:"ly-rwy",   label:"Runway",            icon:'<i style="background:#4D4A49;height:5px"></i>'},
  {id:"ly-node",  label:"Path endpoint",     icon:'<i class="d" style="background:#fff;border:1px solid #111"></i>'},
  {id:"ly-cand",  label:"Candidate paths",   icon:'<i style="background:#BBBBBB"></i>'},
  {id:"ly-pot",   label:"Potential R",       icon:'<i style="background:#1F77B4"></i>'},
  {id:"ly-line",  label:"Trajectory line",   icon:'<i style="background:#2CA02C;opacity:.55"></i>'},
  {id:"ly-best",  label:"Identified path",   icon:'<i style="background:#FF7F0E"></i>'},
  {id:"ly-ref",   label:"Reference route",   icon:'<i style="background:repeating-linear-gradient(90deg,#00BFFF 0 4px,transparent 4px 7px)"></i>'},
  {id:"ly-track", label:"Trajectory",        icon:'<i class="d" style="background:#2CA02C"></i>'}
];

function updateLegend(){
  const box = document.getElementById("legend");
  let rows = "";
  for(const it of LEGEND_ITEMS){
    const cb = document.getElementById(it.id);
    if(!cb || !cb.checked) continue;
    if(it.id === "ly-ref" && !(current && current.reference)) continue;
    rows += "<div>" + it.icon + it.label + "</div>";
  }
  if(!rows){ box.style.display = "none"; return; }
  box.innerHTML = '<div class="t">Legend</div>' + rows;
  box.style.display = "block";
}

// 视野
function computeBBox(){
  let minX=Infinity,maxX=-Infinity,minZ=Infinity,maxZ=-Infinity;
  const acc=(x,z)=>{ if(x<minX)minX=x; if(x>maxX)maxX=x; if(z<minZ)minZ=z; if(z>maxZ)maxZ=z; };
  let used=false;
  if(current && current.best.length){
    for(const name of current.best){
      const i=NAME_IDX.get(name); if(i===undefined) continue;
      const l=NET.lines[i];
      for(let k=0;k<l.length;k+=2) acc(l[k],l[k+1]);
      used=true;
    }
  }
  if(!used && current && current.track.length){ current.track.forEach(p=>acc(p[0],p[1])); used=true; }
  if(!used) NET.lines.forEach(l=>{for(let k=0;k<l.length;k+=2) acc(l[k],l[k+1]);});
  return {minX,maxX,minZ,maxZ};
}

function resetView(){
  bbox = computeBBox();
  syncRot();
  // 旋转后包围盒（用于适配缩放）——矩形绕其中心旋转，中心仍在旋转中心
  let mnX=Infinity, mxX=-Infinity, mnY=Infinity, mxY=-Infinity;
  const corners=[[bbox.minX,bbox.minZ],[bbox.minX,bbox.maxZ],[bbox.maxX,bbox.minZ],[bbox.maxX,bbox.maxZ]];
  for(const c0 of corners){
    const rx=c0[0]*view.c - c0[1]*view.s;
    const ry=c0[0]*view.s + c0[1]*view.c;
    if(rx<mnX)mnX=rx; if(rx>mxX)mxX=rx;
    if(ry<mnY)mnY=ry; if(ry>mxY)mxY=ry;
  }
  view.ox = (bbox.minX + bbox.maxX)/2;
  view.oz = (bbox.minZ + bbox.maxZ)/2;
  const w = Math.max(cv.clientWidth - 70, 40);
  const h = Math.max(cv.clientHeight - 70, 40);
  view.scale = Math.min(w/Math.max(mxX-mnX,1), h/Math.max(mxY-mnY,1));
  draw();
}

function resize(){
  dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth - 28, H = wrap.clientHeight - 28;
  cv.width = Math.max(1, Math.floor(W*dpr));
  cv.height= Math.max(1, Math.floor(H*dpr));
  cv.style.width = W+"px"; cv.style.height = H+"px";
  ctr.x = cv.clientWidth/2;
  ctr.y = cv.clientHeight/2;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}

function polyline(flat,color,widthPx,alpha){
  if(flat.length<4) return;
  ctx.globalAlpha = alpha===undefined?1:alpha;
  ctx.beginPath();
  let p = w2s(flat[0], flat[1]);
  ctx.moveTo(p[0], p[1]);
  for(let i=2;i<flat.length;i+=2){
    p = w2s(flat[i], flat[i+1]);
    ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle=color; ctx.lineWidth=widthPx;
  ctx.lineCap="round"; ctx.lineJoin="round"; ctx.stroke();
  ctx.globalAlpha=1;
}

function drawNames(names,color,widthPx,alpha){
  for(const nm of names){
    const i=NAME_IDX.get(nm); if(i===undefined) continue;
    polyline(NET.lines[i], color, widthPx, alpha);
  }
}

function draw(){
  syncRot();

  const onBG   = document.getElementById("ly-bg").checked;
  const onNet  = document.getElementById("ly-net").checked;
  const onRwy  = document.getElementById("ly-rwy").checked;
  const onNode = document.getElementById("ly-node").checked;
  const onPot  = document.getElementById("ly-pot").checked;
  const onCand = document.getElementById("ly-cand").checked;
  const onTrk  = document.getElementById("ly-track").checked;
  const onLine = document.getElementById("ly-line").checked;
  const onBest = document.getElementById("ly-best").checked;
  const onRef  = document.getElementById("ly-ref").checked;

  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle="#FFFFFF";
  ctx.fillRect(0,0,cv.clientWidth,cv.clientHeight);

  // 1 机坪 / 航站楼
  if(onBG){
    for(const poly of BG){
      const p=poly.pts;
      ctx.beginPath();
      let q = w2s(p[0], p[1]);
      ctx.moveTo(q[0], q[1]);
      for(let i=2;i<p.length;i+=2){
        q = w2s(p[i], p[i+1]);
        ctx.lineTo(q[0], q[1]);
      }
      ctx.closePath();
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = poly.color; ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#8A8A8A";
      ctx.lineWidth = 0.8; ctx.stroke();
    }
  }

  // 世界坐标 -> 屏幕（旋转 + 缩放 + 平移），路网/跑道一次 stroke
  const applyWorld = () => {
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.translate(ctr.x, ctr.y);
    ctx.rotate(view.rot);
    ctx.scale(view.scale, view.scale);
    ctx.translate(-view.ox, -view.oz);
  };

  // 2 路网
  if(onNet){
    ctx.save();
    applyWorld();
    ctx.strokeStyle="#79706E";
    ctx.lineWidth = 1.6/view.scale;
    ctx.lineCap="round"; ctx.lineJoin="round";
    ctx.stroke(netPath);
    ctx.restore();
  }

  // 3 跑道（加粗深灰）
  if(onRwy){
    ctx.save();
    applyWorld();
    ctx.strokeStyle="#4D4A49";
    ctx.lineWidth = 6/view.scale;
    ctx.lineCap="butt";
    ctx.stroke(rwyPath);
    ctx.restore();
  }

  // 4 路网端点（放大后显示，白填充黑边）
  if(onNode && view.scale > 0.45){
    ctx.fillStyle="#FFFFFF";
    ctx.strokeStyle="#111111";
    ctx.lineWidth=1.1;
    for(const l of NET.lines){
      const ends=[[l[0],l[1]],[l[l.length-2],l[l.length-1]]];
      for(const e of ends){
        const q=w2s(e[0], e[1]);
        if(q[0]<-10||q[1]<-10||q[0]>cv.clientWidth+10||q[1]>cv.clientHeight+10) continue;
        ctx.beginPath(); ctx.arc(q[0],q[1],2.2,0,Math.PI*2); ctx.fill(); ctx.stroke();
      }
    }
  }

  // 5 候选路径
  if(onCand && current){
    for(const cand of current.candidates) drawNames(cand,"#BBBBBB",1.2,0.5);
  }

  // 6 潜在R
  if(onPot && current){
    drawNames(current.potential,"#1F77B4",3.2,0.9);
  }

  // 7 轨迹线（可选，浅绿细线）
  if(onLine && current){
    const flat=[]; current.track.forEach(p=>flat.push(p[0],p[1]));
    polyline(flat,"#2CA02C",1.4,0.45);
  }

  // 8 识别路径（橙）
  if(onBest && current){
    drawNames(current.best,"#FF7F0E",4.0,0.95);
  }

  // 9 参照路线（天蓝虚线，画在轨迹点之下以免遮挡）
  if(onRef && current && current.reference){
    ctx.setLineDash([7,5]);
    drawNames(current.reference.route,"#00BFFF",3.0,1);
    ctx.setLineDash([]);
  }

  // 10 轨迹点（每 3 点取 1；与论文一致，画在识别路径之上）
  if(onTrk && current){
    ctx.fillStyle="#2CA02C";
    const t=current.track;
    for(let i=0;i<t.length;i+=3){
      const q=w2s(t[i][0], t[i][1]);
      if(q[0]<-6||q[1]<-6||q[0]>cv.clientWidth+6||q[1]>cv.clientHeight+6) continue;
      ctx.beginPath(); ctx.arc(q[0],q[1],3.4,0,Math.PI*2); ctx.fill();
    }
  }

  updateLegend();
}

// 交互
let drag=null;
cv.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,ox:view.ox,oz:view.oz};cv.classList.add("drag");});
window.addEventListener("mouseup",()=>{drag=null;cv.classList.remove("drag");});
window.addEventListener("mousemove",e=>{
  if(!drag) return;
  syncRot();
  const du=(e.clientX-drag.x)/view.scale, dv=(e.clientY-drag.y)/view.scale;
  view.ox = drag.ox - (du*view.c + dv*view.s);
  view.oz = drag.oz - (-du*view.s + dv*view.c);
  draw();
});
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  syncRot();
  const r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  const w=s2w(mx,my);
  const k=Math.exp(-e.deltaY*0.0015);
  view.scale=Math.min(Math.max(view.scale*k,0.02),60);
  const u=(mx-ctr.x)/view.scale, v=(my-ctr.y)/view.scale;
  view.ox = w[0] - u*view.c - v*view.s;
  view.oz = w[1] + u*view.s - v*view.c;
  draw();
},{passive:false});
cv.addEventListener("dblclick",resetView);

// 悬停显示路径名
cv.addEventListener("mousemove",e=>{
  if(drag) return;
  const r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  let best=null, bestD=5;
  for(let i=0;i<NET.lines.length;i++){
    const l=NET.lines[i];
    for(let j=0;j<l.length-2;j+=2){
      const a=w2s(l[j],l[j+1]), b=w2s(l[j+2],l[j+3]);
      const dx=b[0]-a[0], dy=b[1]-a[1];
      const L2=dx*dx+dy*dy||1e-9;
      let t=((mx-a[0])*dx+(my-a[1])*dy)/L2;
      t=Math.max(0,Math.min(1,t));
      const d=Math.hypot(mx-(a[0]+t*dx), my-(a[1]+t*dy));
      if(d<bestD){ bestD=d; best=i; }
    }
  }
  if(best!==null){
    tip.textContent = NET.names[best] + (NET.lengths[best]? " · "+Math.round(NET.lengths[best])+" m" : "");
    tip.style.left=mx+"px"; tip.style.top=my+"px"; tip.style.display="block";
  }else{
    tip.style.display="none";
  }
});
cv.addEventListener("mouseleave",()=>{tip.style.display="none";});

// 航班切换
const sel=document.getElementById("flight");
FLIGHTS.forEach((f,i)=>{
  const o=document.createElement("option"); o.value=i; o.textContent=f.id; sel.appendChild(o);
});

function setFlight(i){
  current=FLIGHTS[i];
  const f=current;
  document.getElementById("ly-ref").checked = !!f.reference;
  const meanDev = (f.bestScore!=null && f.track.length) ? (Number(f.bestScore)/f.track.length).toFixed(1) : "-";
  document.getElementById("stats").innerHTML =
    "<div>Track points: <b>"+f.track.length+"</b></div>"+
    "<div>Potential R: <b>"+f.potential.length+"</b></div>"+
    "<div>Candidate paths: <b>"+f.totalCandidates+"</b></div>"+
    "<div>Best score: <b>"+(f.bestScore!=null?Number(f.bestScore).toFixed(1):"-")+"</b></div>"+
    "<div>Mean deviation: <b>"+meanDev+"</b> m/pt</div>"+
    "<div>Path nodes: <b>"+f.best.length+"</b></div>"+
    (f.reference
      ? "<div class='ref'>Reference: score <b>"+Number(f.reference.score).toFixed(1)+
        "</b> · dev <b>"+Number(f.reference.meanDev).toFixed(1)+"</b> m/pt</div>"
      : "");
  resetView();
}
sel.addEventListener("change",()=>setFlight(+sel.value));
document.querySelectorAll(".layer input[type=checkbox]").forEach(cb=>cb.addEventListener("change",draw));

// 画布旋转（默认 30°）
const rotEl=document.getElementById("rot"), rotVal=document.getElementById("rotVal");
rotEl.addEventListener("input",()=>{
  view.rot = (+rotEl.value) * Math.PI/180;
  rotVal.textContent = rotEl.value + "\u00B0";
  syncRot();
  draw();
});

document.getElementById("reset").addEventListener("click",resetView);
document.getElementById("png").addEventListener("click",()=>{
  const a=document.createElement("a");
  a.download=(current?current.id:"viewer")+".png";
  a.href=cv.toDataURL("image/png");
  a.click();
});
window.addEventListener("resize",resize);

syncRot();
resize();
if(FLIGHTS.length){ sel.value=0; setFlight(0); } else { draw(); }
</script>
</body>
</html>
"""


# ==============================
# 主程序
# ==============================
def main():
    parser = argparse.ArgumentParser(description="算法输出可视化（论文风格单文件 HTML）")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--out", default=None, help="HTML 输出路径")
    parser.add_argument("--max-candidates", type=int, default=60,
                        help="每个航班绘制的候选路径条数（按得分取前 N）")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    out_html = args.out or os.path.join(output_dir, "viewer.html")

    print("[可视化] 加载路网 ...")
    net = build_network()
    bg = build_background()
    print(f"[可视化] 路网路径 {len(net['names'])} 条，机坪/航站楼 {len(bg)} 面")
    for r in net["runways"]:
        print(f"[可视化] 跑道 z={r[1]:.0f}  x {r[0]:.0f} .. {r[2]:.0f}  长 {r[2]-r[0]:.0f} m")

    print("[可视化] 加载航班结果 ...")
    flights = build_flights(output_dir, args.max_candidates)
    print(f"[可视化] 航班 {len(flights)} 个")

    def js(obj):
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    html = (HTML_TEMPLATE
            .replace("__NET__", js(net))
            .replace("__BG__", js(bg))
            .replace("__FLIGHTS__", js(flights)))

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[可视化] 已生成: {out_html} ({os.path.getsize(out_html)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
