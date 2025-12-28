function qs(sel) { return document.querySelector(sel); }
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function toast(title, sub = "") {
  const host = qs("#toastHost");
  if (!host) return;
  const div = document.createElement("div");
  div.className = "toast";
  div.innerHTML = `<div class="t-title">${escapeHtml(title)}</div><div class="t-sub">${escapeHtml(sub)}</div>`;
  host.appendChild(div);
  setTimeout(() => { div.style.opacity = "0"; div.style.transform = "translateY(6px)"; }, 2600);
  setTimeout(() => { div.remove(); }, 3000);
}

function modalConfirm(opts) {
  const host = qs("#modalHost");
  if (!host) return Promise.resolve(false);
  const title = opts?.title || "确认操作";
  const body = opts?.body || "";
  const okText = opts?.okText || "确认";
  const cancelText = opts?.cancelText || "取消";

  return new Promise((resolve) => {
    host.classList.add("active");
    host.innerHTML = `
      <div class="modal-backdrop"></div>
      <div class="modal" role="dialog" aria-modal="true">
        <div class="modal-title">${escapeHtml(title)}</div>
        <div class="modal-body">${escapeHtml(body)}</div>
        <div class="modal-actions">
          <button class="btn secondary small" data-act="cancel">${escapeHtml(cancelText)}</button>
          <button class="btn small" data-act="ok">${escapeHtml(okText)}</button>
        </div>
      </div>
    `;
    function close(result) {
      host.classList.remove("active");
      host.innerHTML = "";
      resolve(result);
    }
    host.querySelector("[data-act='cancel']").addEventListener("click", () => close(false));
    host.querySelector("[data-act='ok']").addEventListener("click", () => close(true));
    host.querySelector(".modal-backdrop").addEventListener("click", () => close(false));
  });
}

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function apiPostJson(url, obj) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj || {})
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function refreshBadge() {
  try {
    const d = await apiGet("/api/stats");
    const b = qs("#badgePending");
    if (b) b.textContent = String(d.pending ?? 0);
  } catch { }
}

function formatBytes(bytes) {
  const num = Number(bytes || 0);
  if (!num) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let idx = 0;
  let val = num;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx += 1;
  }
  return `${val.toFixed(val >= 10 || idx === 0 ? 0 : 2)} ${units[idx]}`;
}

function formatEta(seconds) {
  const sec = Math.max(0, Number(seconds || 0));
  if (!sec) return "-";
  const mins = Math.floor(sec / 60);
  const hrs = Math.floor(mins / 60);
  const remMin = mins % 60;
  const remSec = Math.floor(sec % 60);
  if (hrs > 0) return `${hrs}h${remMin}m`;
  if (mins > 0) return `${mins}m${remSec}s`;
  return `${remSec}s`;
}

function buildDownloadNote(dl) {
  const doneFiles = Number(dl.done_files || 0);
  const totalFiles = Number(dl.total_files || 0);
  const fileText = totalFiles ? `文件 ${doneFiles}/${totalFiles}` : "";
  const currentFile = dl.current_file || "";
  const curDone = Number(dl.current_file_bytes || 0);
  const curTotal = Number(dl.current_file_total || 0);
  let curText = "";
  if (currentFile) {
    const pct = curTotal ? Math.min(100, (curDone / curTotal) * 100) : 0;
    const sizeText = curTotal ? `${formatBytes(curDone)}/${formatBytes(curTotal)}` : formatBytes(curDone);
    curText = `${currentFile} ${pct.toFixed(1)}% (${sizeText})`;
  }
  return [fileText, curText].filter(Boolean).join(" · ");
}

function buildDownloadSpeed(dl) {
  const speed = Number(dl.speed_mbps || 0);
  const bytesDone = Number(dl.bytes_done || 0);
  const bytesTotal = Number(dl.bytes_total || 0);
  const eta = formatEta(dl.eta_seconds || 0);
  const parts = [];
  if (bytesTotal) {
    parts.push(`${formatBytes(bytesDone)}/${formatBytes(bytesTotal)}`);
  }
  if (speed) {
    parts.push(`速度 ${speed.toFixed(2)} MB/s`);
  }
  if (speed && bytesTotal) {
    parts.push(`预计 ${eta}`);
  }
  return parts.join(" · ");
}

async function loadSeriesOptions(selectEl) {
  if (!selectEl) return;
  const d = await apiGet("/api/series");
  for (const it of (d.items || [])) {
    const opt = document.createElement("option");
    opt.value = it.name;
    opt.textContent = it.name;
    selectEl.appendChild(opt);
  }
}

function renderResults(container, items) {
  container.innerHTML = "";
  for (const it of items) {
    const match = Number(it.match_rate ?? it.fit_rate ?? 0);
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <img src="${it.url}" />
      <div class="meta"><span>${escapeHtml(it.series)}</span><span>匹配度 ${match.toFixed(2)}%</span></div>
      <div class="meta"><span>raw ${Number(it.raw || 0).toFixed(3)}</span><span>#${it.id}</span></div>
      <div class="meta"><span class="mono">${escapeHtml((it.tags || []).slice(0, 8).join(" "))}</span><span></span></div>
    `;
    container.appendChild(div);
  }
}

function onTryPage() {
  const btn = qs("#btnTry");
  if (!btn) return;
  const tags = qs("#tags");
  const k = qs("#k");
  const series = qs("#series");
  const order = qs("#order");
  const result = qs("#result");
  const resultMeta = qs("#resultMeta");

  loadSeriesOptions(series).catch(() => { });

  async function run() {
    const text = tags.value || "";
    const kk = Number(k.value || 6);
    const sr = series.value || "";
    const ord = order ? (order.value || "desc") : "desc";
    btn.disabled = true;
    resultMeta.textContent = "匹配中…";
    try {
      const d = await apiPostJson("/api/select", {
        tags: text,
        k: kk,
        series: sr,
        order: ord
      });
      renderResults(result, d.items || []);
      resultMeta.textContent = `返回 ${d.items?.length ?? 0} 条 · k=${kk} · series=${sr || "全部"} · 排序=${ord}`;
      toast("匹配完成", `返回 ${d.items?.length ?? 0} 条`);
    } catch (e) {
      alert(String(e));
      resultMeta.textContent = "失败";
    } finally {
      btn.disabled = false;
    }
  }

  btn.addEventListener("click", run);
  tags.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      run();
    }
  });
}

function onLabPage() {
  const out = qs("#labOut");
  if (!out) return;

  const btnClear = qs("#btnLabClear");
  const tags = qs("#labTags");
  const k = qs("#labK");
  const series = qs("#labSeries");
  const order = qs("#labOrder");
  const btnSel = qs("#btnLabSelect");
  const codeSel = qs("#labSelectCode");
  const seriesName = qs("#labSeriesName");
  const btnSeriesExists = qs("#btnLabSeriesExists");
  const codeSeriesExists = qs("#labSeriesExistsCode");
  const seriesCountId = qs("#labSeriesCountId");
  const seriesCountName = qs("#labSeriesCountName");
  const btnSeriesCount = qs("#btnLabSeriesCount");
  const codeSeriesCount = qs("#labSeriesCountCode");

  function writeOut(obj) {
    out.textContent = JSON.stringify(obj, null, 2);
  }

  function buildCode() {
    const payload = {
      tags: tags.value || "",
      k: Number(k.value || 6),
      series: series.value || "",
      order: order ? (order.value || "desc") : "desc"
    };
    const payloadText = JSON.stringify(payload, null, 2);
    codeSel.textContent =
      `curl -X POST "http://127.0.0.1:8000/api/select" \\
  -H "Content-Type: application/json" \\
  -d '${payloadText}'

fetch("/api/select", {
  method: "POST",
  headers: {"Content-Type":"application/json"},
  body: JSON.stringify(${payloadText})
}).then(r=>r.json()).then(console.log);`;
  }

  function buildSeriesExistsCode() {
    if (!codeSeriesExists) return;
    const payload = { name: seriesName ? (seriesName.value || "") : "" };
    const payloadText = JSON.stringify(payload, null, 2);
    codeSeriesExists.textContent =
      `curl -X POST "http://127.0.0.1:8000/api/series/exists" \\
  -H "Content-Type: application/json" \\
  -d '${payloadText}'

fetch("/api/series/exists", {
  method: "POST",
  headers: {"Content-Type":"application/json"},
  body: JSON.stringify(${payloadText})
}).then(r=>r.json()).then(console.log);`;
  }

  function buildSeriesCountCode() {
    if (!codeSeriesCount) return;
    const payload = {};
    const idVal = seriesCountId ? String(seriesCountId.value || "").trim() : "";
    const nameVal = seriesCountName ? (seriesCountName.value || "").trim() : "";
    if (idVal) {
      payload.series_id = Number(idVal);
    } else if (nameVal) {
      payload.name = nameVal;
    }
    const payloadText = JSON.stringify(payload, null, 2);
    codeSeriesCount.textContent =
      `curl -X POST "http://127.0.0.1:8000/api/series/count" \\
  -H "Content-Type: application/json" \\
  -d '${payloadText}'

fetch("/api/series/count", {
  method: "POST",
  headers: {"Content-Type":"application/json"},
  body: JSON.stringify(${payloadText})
}).then(r=>r.json()).then(console.log);`;
  }

  buildCode();
  [tags, k, series, order].filter(Boolean).forEach(el => {
    el.addEventListener("input", buildCode);
  });
  if (seriesName) seriesName.addEventListener("input", buildSeriesExistsCode);
  if (seriesCountId) seriesCountId.addEventListener("input", buildSeriesCountCode);
  if (seriesCountName) seriesCountName.addEventListener("input", buildSeriesCountCode);
  buildSeriesExistsCode();
  buildSeriesCountCode();

  btnSel.addEventListener("click", async () => {
    try {
      const d = await apiPostJson("/api/select", {
        tags: tags.value || "",
        k: Number(k.value || 6),
        series: series.value || "",
        order: order ? (order.value || "desc") : "desc"
      });
      writeOut(d);
      toast("调用成功", "/api/select");
    } catch (e) {
      toast("调用失败", String(e));
    }
  });

  if (btnSeriesExists) {
    btnSeriesExists.addEventListener("click", async () => {
      try {
        const d = await apiPostJson("/api/series/exists", {
          name: seriesName ? (seriesName.value || "") : ""
        });
        writeOut(d);
        toast("调用成功", "/api/series/exists");
      } catch (e) {
        toast("调用失败", String(e));
      }
    });
  }

  if (btnSeriesCount) {
    btnSeriesCount.addEventListener("click", async () => {
      try {
        const payload = {};
        const idVal = seriesCountId ? String(seriesCountId.value || "").trim() : "";
        const nameVal = seriesCountName ? (seriesCountName.value || "").trim() : "";
        if (idVal) {
          payload.series_id = Number(idVal);
        } else if (nameVal) {
          payload.name = nameVal;
        }
        const d = await apiPostJson("/api/series/count", payload);
        writeOut(d);
        toast("调用成功", "/api/series/count");
      } catch (e) {
        toast("调用失败", String(e));
      }
    });
  }

  btnClear.addEventListener("click", () => {
    out.textContent = "";
  });
}

function onBenchmarkPage() {
  const modelList = qs("#modelList");
  if (!modelList) return;
  const downloadWrap = qs("#downloadWrap");
  const downloadBar = qs("#downloadBar");
  const downloadText = qs("#downloadText");
  const downloadSpeed = qs("#downloadSpeed");
  const downloadNote = qs("#downloadNote");
  const btnCancelDownload = qs("#btnCancelDownload");
  const restartWrap = qs("#restartWrap");
  const hfEndpointInput = qs("#hfEndpointInput");
  const recallTopkInput = qs("#recallTopkInput");
  const btnSaveSettings = qs("#btnSaveSettings");
  const btnRefreshModels = qs("#btnRefreshModels");
  const btnRefreshDevice = qs("#btnRefreshDevice");
  const embeddingWrap = qs("#embeddingWrap");
  const embeddingBar = qs("#embeddingBar");
  const embeddingText = qs("#embeddingText");
  const embeddingReady = qs("#embeddingReady");

  const torchVersion = qs("#torchVersion");
  const cudaBuild = qs("#cudaBuild");
  const cudaAvail = qs("#cudaAvail");
  const cudaDevice = qs("#cudaDevice");
  const cpuCores = qs("#cpuCores");
  const cpuMaxMhz = qs("#cpuMaxMhz");
  const ramTotal = qs("#ramTotal");
  const ramAvail = qs("#ramAvail");
  const cudaVram = qs("#cudaVram");
  const cudaClock = qs("#cudaClock");

  const recommendLevel = qs("#recommendLevel");
  const recommendModel = qs("#recommendModel");
  const recommendNote = qs("#recommendNote");
  const recommendMeta = qs("#recommendMeta");
  const btnApplyRecommend = qs("#btnApplyRecommend");

  const MIXED_KEY = "bge-large-zh-v1.5-rerank-fp32";
  const DEFAULT_RECALL_TOPK = 20;
  let restartTriggered = false;
  let waitLoopRunning = false;
  let recommendKey = "";
  let recommendRecallTopk = 0;
  let currentKey = "";
  let recallDirty = false;
  let recallValue = DEFAULT_RECALL_TOPK;
  let endpointDirty = false;
  let recallDirtyTop = false;
  let refreshTimer = null;

  function isMixed(key) {
    return key === MIXED_KEY;
  }

  function setProgress(wrap, bar, textEl, pct, note) {
    if (!wrap || !bar || !textEl) return;
    wrap.style.display = "block";
    const val = Math.max(0, Math.min(100, Number(pct || 0)));
    bar.style.width = `${val.toFixed(1)}%`;
    textEl.textContent = note ? `${val.toFixed(1)}% · ${note}` : `${val.toFixed(1)}%`;
  }

  function fmtGb(val) {
    const num = Number(val || 0);
    if (!num) return "-";
    return `${num.toFixed(2)} GB`;
  }

  function fmtMhz(val) {
    const num = Number(val || 0);
    if (!num) return "-";
    return `${Math.round(num)} MHz`;
  }

  async function waitForServer() {
    if (waitLoopRunning) return;
    waitLoopRunning = true;
    while (true) {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (res.ok) {
          location.href = "/benchmark";
          return;
        }
      } catch { }
      await new Promise(r => setTimeout(r, 1000));
    }
  }

  function renderModelList(models, currentKey, downloadState) {
    if (!Array.isArray(models)) return;
    const busy = downloadState.status === "downloading" || downloadState.status === "ready";
    const downloadingKey = downloadState.model_key || "";
    modelList.innerHTML = "";
    models.forEach((m) => {
      const row = document.createElement("div");
      row.className = "model-row";
      const isCurrent = m.key === currentKey;
      if (isCurrent) row.classList.add("current");
      if (busy && m.key === downloadingKey) row.classList.add("downloading");

      const info = document.createElement("div");
      const title = document.createElement("div");
      title.className = "model-title";
      title.textContent = m.label;
      const sub = document.createElement("div");
      sub.className = "model-sub";
      const poolingText = m.pooling ? `pooling ${m.pooling}` : "";
      const precisionText = m.precision ? ` · ${m.precision}` : "";
      const modeText = m.mode === "rerank" ? " · rerank" : "";
      const missingText = (!m.downloaded && m.missing?.length) ? ` · 缺少: ${m.missing.join(", ")}` : "";
      sub.textContent = `key=${m.key}${precisionText}${modeText}${poolingText ? " · " + poolingText : ""}${missingText}`;
      // 在此之前先添加一个br
      const br = document.createElement("br");


      const statusPill = document.createElement("span");
      statusPill.className = "pill";
      if (busy && m.key === downloadingKey) {
        statusPill.classList.add("warn");
        statusPill.textContent = "下载中";
      } else if (m.downloaded) {
        statusPill.classList.add("ok");
        statusPill.textContent = "已下载";
      } else {
        statusPill.classList.add("bad");
        statusPill.textContent = "未下载";
      }

      info.appendChild(title);
      info.appendChild(sub);
      info.appendChild(br);
      info.appendChild(statusPill);

      if (isMixed(m.key)) {
        // recall topk moved to top settings
      }

      const actions = document.createElement("div");
      actions.className = "model-actions";

      if (!m.downloaded) {
        const btnDownload = document.createElement("button");
        btnDownload.className = "btn small";
        btnDownload.textContent = (busy && m.key === downloadingKey) ? "下载中" : "下载";
        btnDownload.disabled = busy;
        btnDownload.addEventListener("click", async () => {
          const ok = await modalConfirm({
            title: "下载模型",
            body: "下载过程中会强制返回本页，直到完成或出错。确定开始下载吗？",
            okText: "开始下载",
            cancelText: "取消",
          });
          if (!ok) return;
          try {
            await apiPostJson("/api/model/download", { model_key: m.key });
            location.href = "/progress";
          } catch (e) {
            toast("下载失败", String(e));
          }
        });
        actions.appendChild(btnDownload);
      } else {
        const btnUse = document.createElement("button");
        btnUse.className = "btn small";
        if (isCurrent) {
          btnUse.textContent = "使用中";
        } else {
          btnUse.textContent = "使用";
        }
        btnUse.disabled = busy || (!isMixed(m.key) && isCurrent);
        btnUse.addEventListener("click", async () => {
          const ok = await modalConfirm({
            title: "切换模型",
            body: "切换后会先准备模型，完成后会提示是否手动重启（低配置推荐重启），也可以直接加载或稍后重启。",
            okText: isCurrent ? "保存" : "切换",
            cancelText: "取消",
          });
          if (!ok) return;
          const payload = { model_key: m.key };
          if (isMixed(m.key)) {
            payload.recall_topk = Number(recallValue || DEFAULT_RECALL_TOPK);
          }
          recallDirty = false;
          try {
            await apiPostJson("/api/model/switch", payload);
            location.href = "/progress";
          } catch (e) {
            toast("切换失败", String(e));
          }
        });
        actions.appendChild(btnUse);

        const btnDelete = document.createElement("button");
        btnDelete.className = "btn danger small";
        btnDelete.textContent = "删除";
        btnDelete.disabled = busy || isCurrent;
        btnDelete.addEventListener("click", async () => {
          const ok = await modalConfirm({
            title: "删除模型缓存",
            body: "删除会清空本地模型文件，需要重新下载才能使用。",
            okText: "删除",
            cancelText: "取消",
          });
          if (!ok) return;
          try {
            await apiPostJson("/api/model/delete", { model_key: m.key });
          } catch (e) {
            toast("删除失败", String(e));
          }
        });
        actions.appendChild(btnDelete);
      }

      row.appendChild(info);
      row.appendChild(actions);
      modelList.appendChild(row);
    });
  }

  async function refreshStatus() {
    try {
      const d = await apiGet("/api/model/status");
      if (torchVersion) torchVersion.textContent = d.device?.torch_version || "-";
      if (cudaBuild) cudaBuild.textContent = d.device?.cuda_build || "-";
      if (cudaAvail) cudaAvail.textContent = d.device?.cuda_available ? "是" : "否";
      if (cudaDevice) cudaDevice.textContent = d.device?.cuda_device || "-";
      if (cpuCores) cpuCores.textContent = String(d.device?.cpu_cores ?? "-");
      if (cpuMaxMhz) cpuMaxMhz.textContent = fmtMhz(d.device?.cpu_max_mhz);
      if (ramTotal) ramTotal.textContent = fmtGb(d.device?.ram_total_gb);
      if (ramAvail) ramAvail.textContent = fmtGb(d.device?.ram_available_gb);
      if (cudaVram) cudaVram.textContent = fmtGb(d.device?.cuda_vram_gb);
      if (cudaClock) {
        const core = d.device?.cuda_clock_mhz || 0;
        const mem = d.device?.cuda_mem_clock_mhz || 0;
        if (core || mem) {
          cudaClock.textContent = mem ? `${fmtMhz(core)} / 显存 ${fmtMhz(mem)}` : fmtMhz(core);
        } else {
          cudaClock.textContent = "-";
        }
      }

      const rec = d.recommendation || {};

      const recommendKey = rec.model_key || "";
      const recommendRecallTopk = Number(rec.recall_topk || 0);
      const level = rec.level || "-";
      const label = rec.label || "-";
      const reason = rec.reason || "-";
      const confidence = rec.confidence || "medium";

      if (recommendLevel) recommendLevel.textContent = level;
      if (recommendModel) recommendModel.textContent = label;
      if (recommendNote) recommendNote.textContent = reason;

      if (recommendMeta) {
        let badge = "";
        if (confidence === "high") badge = "⭐ 默认推荐";
        else if (confidence === "medium") badge = "⚖️ 偏向高精度";
        else badge = "⚠️ 专家模式";

        if (recommendKey) {
          const extra =
            recommendKey.includes("rerank")
              ? ` · 召回 topk=${recommendRecallTopk || DEFAULT_RECALL_TOPK}`
              : "";

          recommendMeta.textContent = `${badge} · ${recommendKey}${extra}`;
        } else {
          recommendMeta.textContent = "";
        }
      }

      if (btnApplyRecommend) {
        if (!recommendKey) {
          btnApplyRecommend.disabled = true;
          btnApplyRecommend.textContent = "暂无推荐";
        } else if (recommendKey === d.current?.model_key) {
          btnApplyRecommend.disabled = true;
          btnApplyRecommend.textContent = "已是推荐";
        } else {
          btnApplyRecommend.disabled = false;
          btnApplyRecommend.textContent =
            confidence === "low"
              ? "我了解风险，仍然使用"
              : "一键选择";
        }
      }


      if (hfEndpointInput) {
        if (!endpointDirty) {
          hfEndpointInput.value = d.hf_endpoint || "https://huggingface.co";
        }
      }
      if (recallTopkInput) {
        const isEditingRecall = document.activeElement === recallTopkInput;
        if (!recallDirtyTop && !isEditingRecall) {
          recallValue = Number(d.current?.recall_topk || DEFAULT_RECALL_TOPK);
          recallTopkInput.value = String(recallValue || DEFAULT_RECALL_TOPK);
        }
      }

      const serverKey = d.current?.model_key || "";
      if (serverKey && serverKey !== currentKey) {
        currentKey = serverKey;
        recallDirty = false;
      }
      const dl = d.download || {};
      const emb = d.embedding || {};
      const pending = !!d.current?.pending_rebuild;
      if (
        dl.status === "downloading"
        || dl.status === "ready"
        || pending
        || emb.status === "running"
      ) {
        location.href = "/progress";
        return;
      }

      if (dl.status === "error") {
        setProgress(downloadWrap, downloadBar, downloadText, 0, "error");
        if (downloadSpeed) downloadSpeed.textContent = "";
        if (downloadNote) downloadNote.textContent = "下载失败，可重试";
      } else if (dl.status === "done") {
        setProgress(downloadWrap, downloadBar, downloadText, 100, "downloaded");
        if (downloadSpeed) downloadSpeed.textContent = "";
        if (downloadNote) downloadNote.textContent = "下载完成，可使用或删除";
      } else if (dl.status === "downloading") {
        setProgress(downloadWrap, downloadBar, downloadText, dl.progress || 0, buildDownloadNote(dl));
        if (downloadSpeed) downloadSpeed.textContent = buildDownloadSpeed(dl);
        if (downloadNote) downloadNote.textContent = "下载中将强制返回本页";
      } else if (downloadWrap) {
        downloadWrap.style.display = "none";
      }
      if (dl.status === "downloading") {
        if (!refreshTimer) {
          refreshTimer = setInterval(refreshStatus, 1500);
        }
      } else if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
      }
      if (btnCancelDownload) {
        btnCancelDownload.style.display = (dl.status === "downloading") ? "inline-flex" : "none";
      }

      if (pending || emb.status === "running") {
        setProgress(embeddingWrap, embeddingBar, embeddingText, emb.progress || 0, emb.message || "");
        if (embeddingReady) embeddingReady.style.display = "none";
      } else {
        if (embeddingWrap) embeddingWrap.style.display = "none";
        if (embeddingReady) embeddingReady.style.display = "inline-flex";
      }

      renderModelList(d.models || [], d.current?.model_key || "", dl);
    } catch (e) {
      // ignore
    }
  }

  async function applyRecommend() {
    if (!recommendKey) return;
    const ok = await modalConfirm({
      title: "应用推荐模型",
      body: "切换后会先准备模型，完成后会提示是否手动重启（低配置推荐重启），也可以直接加载或稍后重启。",
      okText: "应用",
      cancelText: "取消",
    });
    if (!ok) return;
    const payload = { model_key: recommendKey };
    if (isMixed(recommendKey)) {
      payload.recall_topk = Number(recallValue || recommendRecallTopk || DEFAULT_RECALL_TOPK);
    }
    recallDirty = false;
    try {
      await apiPostJson("/api/model/switch", payload);
      location.href = "/progress";
    } catch (e) {
      toast("应用失败", String(e));
    }
  }

  async function cancelDownload() {
    const ok = await modalConfirm({
      title: "取消下载",
      body: "将停止下载并清空已下载的部分缓存。",
      okText: "取消并清空",
      cancelText: "继续下载",
    });
    if (!ok) return;
    try {
      await apiPostJson("/api/model/download/cancel", { clear: true });
    } catch (e) {
      toast("取消失败", String(e));
    }
  }

  if (btnApplyRecommend) btnApplyRecommend.addEventListener("click", applyRecommend);
  if (btnCancelDownload) btnCancelDownload.addEventListener("click", cancelDownload);
  if (hfEndpointInput) {
    hfEndpointInput.addEventListener("input", () => {
      endpointDirty = true;
    });
  }
  if (recallTopkInput) {
    const markRecallDirty = () => {
      recallDirtyTop = true;
      recallValue = Number(recallTopkInput.value || DEFAULT_RECALL_TOPK);
    };
    recallTopkInput.addEventListener("input", markRecallDirty);
    recallTopkInput.addEventListener("change", markRecallDirty);
  }
  if (btnSaveSettings) {
    btnSaveSettings.addEventListener("click", async () => {
      const ops = [];
      if (!hfEndpointInput) return;
      const endpointValue = String(hfEndpointInput.value || "").trim();
      if (endpointValue) {
        ops.push(apiPostJson("/api/model/hf-endpoint", { endpoint: endpointValue }));
      }
      if (recallTopkInput && currentKey) {
        const recallVal = Number(recallTopkInput.value || DEFAULT_RECALL_TOPK);
        ops.push(apiPostJson("/api/model/switch", { model_key: currentKey, recall_topk: recallVal }));
      }
      if (!ops.length) {
        toast("没有可保存的内容", "");
        return;
      }
      try {
        await Promise.all(ops);
        endpointDirty = false;
        recallDirtyTop = false;
        toast("保存成功", "设置已更新");
        await refreshStatus();
      } catch (e) {
        toast("保存失败", String(e));
      }
    });
  }
  if (btnRefreshModels) {
    btnRefreshModels.addEventListener("click", refreshStatus);
  }
  if (btnRefreshDevice) {
    btnRefreshDevice.addEventListener("click", refreshStatus);
  }

  refreshStatus();
}

// =============== PROGRESS PAGE ===============
function onProgressPage() {
  const downloadWrap = qs("#downloadWrap");
  if (!downloadWrap) return;
  const downloadBar = qs("#downloadBar");
  const downloadText = qs("#downloadText");
  const downloadSpeed = qs("#downloadSpeed");
  const downloadNote = qs("#downloadNote");
  const downloadIdle = qs("#downloadIdle");
  const btnCancelDownload = qs("#btnCancelDownload");
  const switchWrap = qs("#switchWrap");
  const btnRestartNow = qs("#btnRestartNow");
  const btnLoadNoRestart = qs("#btnLoadNoRestart");
  const btnRestartLater = qs("#btnRestartLater");
  const restartWrap = qs("#restartWrap");
  const embeddingWrap = qs("#embeddingWrap");
  const embeddingBar = qs("#embeddingBar");
  const embeddingText = qs("#embeddingText");
  const embeddingReady = qs("#embeddingReady");
  const btnGoBack = qs("#btnGoBack");
  const statusHint = qs("#statusHint");

  let restartTriggered = false;
  let waitLoopRunning = false;
  let backScheduled = false;

  function setProgress(wrap, bar, textEl, pct, note) {
    if (!wrap || !bar || !textEl) return;
    wrap.style.display = "block";
    const val = Math.max(0, Math.min(100, Number(pct || 0)));
    bar.style.width = `${val.toFixed(1)}%`;
    textEl.textContent = note ? `${val.toFixed(1)}% · ${note}` : `${val.toFixed(1)}%`;
  }

  async function waitForServer() {
    if (waitLoopRunning) return;
    waitLoopRunning = true;
    while (true) {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (res.ok) {
          location.reload();
          return;
        }
      } catch { }
      await new Promise(r => setTimeout(r, 1000));
    }
  }

  async function restartNow() {
    if (restartTriggered) return;
    restartTriggered = true;
    if (restartWrap) restartWrap.style.display = "block";
    if (switchWrap) switchWrap.style.display = "none";
    try {
      await apiPostJson("/api/restart", {});
    } catch (e) {
      restartTriggered = false;
      if (restartWrap) restartWrap.style.display = "none";
      if (switchWrap) switchWrap.style.display = "block";
      toast("重启失败", String(e));
      return;
    }
    waitForServer();
  }

  async function applySwitchNow() {
    try {
      const res = await apiPostJson("/api/model/switch/apply", {});
      if (!res.ok) {
        toast("切换失败", res.message || res.status || "");
        return;
      }
      if (switchWrap) switchWrap.style.display = "none";
    } catch (e) {
      toast("切换失败", String(e));
    }
  }

  async function deferSwitch() {
    try {
      const res = await apiPostJson("/api/model/switch/defer", {});
      if (!res.ok) {
        toast("操作失败", res.message || res.status || "");
        return;
      }
      if (switchWrap) switchWrap.style.display = "none";
    } catch (e) {
      toast("操作失败", String(e));
    }
  }

  async function cancelDownload() {
    const ok = await modalConfirm({
      title: "取消下载",
      body: "将停止下载并清空已下载的部分缓存。",
      okText: "取消并清空",
      cancelText: "继续下载",
    });
    if (!ok) return;
    try {
      await apiPostJson("/api/model/download/cancel", { clear: true });
    } catch (e) {
      toast("取消失败", String(e));
    }
  }

  async function refreshStatus() {
    try {
      const res = await fetch("/api/progress/status", { cache: "no-store" });
      if (!res.ok) throw new Error(await res.text());
      const d = await res.json();
      const dl = d.download || {};
      const emb = d.embedding || {};
      const pending = !!d.pending_rebuild;

      const downloading = dl.status === "downloading";
      const ready = dl.status === "ready";
      const done = dl.status === "done";
      const error = dl.status === "error";
      const activeDownload = downloading || ready;
      const activeEmbed = pending || emb.status === "running";
      const active = activeDownload || activeEmbed;
      const canAutoBack = !active && !error;

      if (downloading) {
        setProgress(downloadWrap, downloadBar, downloadText, dl.progress || 0, buildDownloadNote(dl));
        if (downloadSpeed) downloadSpeed.textContent = buildDownloadSpeed(dl);
        if (downloadNote) downloadNote.textContent = "下载中将强制停留在本页";
        if (downloadIdle) downloadIdle.style.display = "none";
        if (switchWrap) switchWrap.style.display = "none";
        if (restartWrap) restartWrap.style.display = restartTriggered ? "block" : "none";
      } else if (ready) {
        setProgress(downloadWrap, downloadBar, downloadText, 100, "ready");
        if (downloadSpeed) downloadSpeed.textContent = "";
        if (downloadNote) downloadNote.textContent = "下载完成，已准备切换";
        if (downloadIdle) downloadIdle.style.display = "none";
        if (switchWrap) switchWrap.style.display = restartTriggered ? "none" : "block";
        if (restartWrap) restartWrap.style.display = restartTriggered ? "block" : "none";
      } else if (done) {
        setProgress(downloadWrap, downloadBar, downloadText, 100, "downloaded");
        if (downloadSpeed) downloadSpeed.textContent = "";
        if (downloadNote) downloadNote.textContent = "下载完成，可返回";
        if (downloadIdle) downloadIdle.style.display = "none";
        if (switchWrap) switchWrap.style.display = "none";
        if (restartWrap) restartWrap.style.display = restartTriggered ? "block" : "none";
      } else if (error) {
        setProgress(downloadWrap, downloadBar, downloadText, 0, "error");
        if (downloadSpeed) downloadSpeed.textContent = "";
        if (downloadNote) downloadNote.textContent = "下载失败，可重试";
        if (downloadIdle) downloadIdle.style.display = "none";
        if (switchWrap) switchWrap.style.display = "none";
        if (restartWrap) restartWrap.style.display = restartTriggered ? "block" : "none";
      } else {
        if (downloadWrap) downloadWrap.style.display = "none";
        if (downloadIdle) downloadIdle.style.display = "block";
        if (restartWrap) restartWrap.style.display = "none";
        if (switchWrap) switchWrap.style.display = "none";
      }

      if (btnCancelDownload) {
        btnCancelDownload.style.display = downloading ? "inline-flex" : "none";
      }
      if (btnRestartNow) btnRestartNow.disabled = restartTriggered;
      if (btnLoadNoRestart) btnLoadNoRestart.disabled = restartTriggered;
      if (btnRestartLater) btnRestartLater.disabled = restartTriggered;

      if (activeEmbed) {
        setProgress(embeddingWrap, embeddingBar, embeddingText, emb.progress || 0, emb.message || "");
        if (embeddingReady) embeddingReady.style.display = "none";
      } else {
        if (embeddingWrap) embeddingWrap.style.display = "none";
        if (embeddingReady) embeddingReady.style.display = "inline-flex";
      }

      if (statusHint) {
        if (ready) {
          statusHint.textContent = "模型已准备好切换，可选择重启或直接加载。";
        } else if (active) {
          statusHint.textContent = "处理中，请勿关闭页面";
        } else {
          statusHint.textContent = error ? "出错了，可返回或重试。" : "已完成，可返回。";
        }
      }
      if (btnGoBack) {
        btnGoBack.style.display = active ? "none" : "inline-flex";
      }
      if (canAutoBack && !backScheduled) {
        backScheduled = true;
        setTimeout(() => { location.href = "/benchmark"; }, 800);
      }
    } catch (e) {
      if (statusHint) statusHint.textContent = "读取进度失败，请稍后重试。";
    }
  }

  if (btnCancelDownload) btnCancelDownload.addEventListener("click", cancelDownload);
  if (btnRestartNow) btnRestartNow.addEventListener("click", restartNow);
  if (btnLoadNoRestart) btnLoadNoRestart.addEventListener("click", applySwitchNow);
  if (btnRestartLater) btnRestartLater.addEventListener("click", deferSwitch);
  if (btnGoBack) btnGoBack.addEventListener("click", () => { location.href = "/benchmark"; });

  refreshStatus();
  setInterval(refreshStatus, 1200);
}

// =============== boot ===============
window.addEventListener("DOMContentLoaded", () => {
  refreshBadge();
  if (window.__PAGE__ === "try") onTryPage();
  if (window.__PAGE__ === "lab") onLabPage();
  if (window.__PAGE__ === "benchmark") onBenchmarkPage();
  if (window.__PAGE__ === "init") onBenchmarkPage();
  if (window.__PAGE__ === "progress") onProgressPage();
});
