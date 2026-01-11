function updateQueryParams() {
  const url = new URL(window.location.toString());
  const params = url.searchParams;

  const featureTypeEl = document.querySelector("[name='feat_type']");
  const featureValueEl = document.querySelector("[name='feat_value']");

  const featureType = featureTypeEl ? featureTypeEl.value : "";
  const featureValue = featureValueEl ? featureValueEl.value : "";

  if (featureType) {
    params.set("type", featureType)
  } else {
    params.delete("type")
  }

  if (featureValue) {
    params.set("value", featureValue)
  } else {
    params.delete("value")
  }

  // Canonical graph params:
  // - graph-mode=(include|exclude), omitted when include
  // - repeated graph=... values for checked graphs
  const mode = getGraphMode();
  if (mode && mode !== "include") {
    params.set("graph-mode", mode);
  } else {
    params.delete("graph-mode");
  }

  params.delete("graph");
  document.querySelectorAll(".graphs-list input[type='checkbox']").forEach((el) => {
    if (el instanceof HTMLInputElement && el.checked) {
      params.append("graph", el.value);
    }
  });

  window.history.replaceState({}, '', url)
}

function showCopiedState(btn) {
  const prev = btn.textContent;
  btn.classList.add("copied");
  btn.textContent = "Copied";
  window.setTimeout(() => {
    btn.classList.remove("copied");
    btn.textContent = prev;
  }, 900);
}

async function copyToClipboard(text, btn) {
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    if (btn) showCopiedState(btn);
  } catch (err) {
    // Fallback for older browsers / permissions
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.top = "-9999px";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    if (btn) showCopiedState(btn);
  }
}

function submitQueryForm() {
  const form = document.querySelector(".query-form");
  if (!(form instanceof HTMLFormElement)) return;

  // Let htmx intercept the submit event on the form.
  form.requestSubmit();
}

function getGraphMode() {
  const checked = document.querySelector("input[name='graph_mode']:checked");
  if (checked instanceof HTMLInputElement) return checked.value;
  return "include";
}

function setGraphMode(mode) {
  const el = document.querySelector(`input[name='graph_mode'][value="${CSS.escape(mode)}"]`);
  if (el instanceof HTMLInputElement) el.checked = true;
  syncGraphCheckboxNames();
  updateGraphsHelpText();
}

function syncGraphCheckboxNames() {
  const mode = getGraphMode();
  document.querySelectorAll(".graphs-list input[type='checkbox']").forEach((el) => {
    if (el instanceof HTMLInputElement) {
      el.name = mode === "exclude" ? "exclude_graphs" : "include_graphs";
    }
  });
}

function updateGraphsHelpText() {
  const help = document.getElementById("graphs-help");
  if (!(help instanceof HTMLElement)) return;

  help.textContent = "Select none = search all graphs";
}

document.addEventListener("change", (e) => {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;

  if (target.matches("input[name='graph_mode']")) {
    syncGraphCheckboxNames();
    updateGraphsHelpText();
    updateQueryParams();
    return;
  }

  if (target.matches(".graphs-list input[type='checkbox']")) {
    updateQueryParams();
    return;
  }
});

document.addEventListener("click", (e) => {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;

  if (target.closest("#graphs-reset")) {
    document.querySelectorAll(".graphs-list input[type='checkbox']").forEach((el) => {
      if (el instanceof HTMLInputElement) el.checked = false;
    });
    updateQueryParams();
    submitQueryForm();
  }
});

// initial sync
syncGraphCheckboxNames();
updateGraphsHelpText();

let activeRowContext = null;

function parseTemplateJson(tmpl) {
  if (!(tmpl instanceof HTMLTemplateElement)) return "";

  // We render with Jinja `tojson`, so innerHTML is valid JSON (quoted string).
  try {
    return JSON.parse(tmpl.innerHTML || "\"\"");
  } catch (e) {
    return tmpl.textContent || "";
  }
}

function hideActionsMenu() {
  const menu = document.getElementById("actions-menu");
  if (!(menu instanceof HTMLElement)) return;
  menu.hidden = true;
  menu.removeAttribute("style");
  activeRowContext = null;
}

function showActionsMenuAt(triggerEl) {
  const menu = document.getElementById("actions-menu");
  if (!(menu instanceof HTMLElement)) return;

  const rect = triggerEl.getBoundingClientRect();

  // Position using fixed so it can escape scroll containers.
  menu.style.position = "fixed";

  // Prefer aligning right edge with trigger, but clamp to viewport.
  const menuWidth = 260;
  const desiredLeft = rect.right - menuWidth;
  const left = Math.max(8, Math.min(desiredLeft, window.innerWidth - menuWidth - 8));

  // Prefer opening below, but flip above if near bottom.
  const desiredTop = rect.bottom + 6;
  const approxHeight = 170;
  const top = (desiredTop + approxHeight > window.innerHeight - 8)
    ? Math.max(8, rect.top - 6 - approxHeight)
    : desiredTop;

  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
  menu.hidden = false;
}

document.addEventListener("scroll", () => hideActionsMenu(), true);
window.addEventListener("resize", () => hideActionsMenu());
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hideActionsMenu();
});

document.addEventListener("click", (e) => {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;

  // Copy URI
  const copyBtn = target.closest(".copy-btn");
  if (copyBtn) {
    const text = copyBtn.getAttribute("data-copy") || "";
    copyToClipboard(text, copyBtn);
    return;
  }

  // Open per-row actions (portal menu)
  const triggerBtn = target.closest(".row-actions-trigger-btn");
  if (triggerBtn instanceof HTMLButtonElement) {
    if (!triggerBtn.getAttribute("data-uri")) return;

    const uri = triggerBtn.getAttribute("data-uri") || "";
    const graph = triggerBtn.getAttribute("data-graph") || "";
    const external = triggerBtn.getAttribute("data-external") || "";

    // Extract repr from adjacent template.
    let repr = "";
    const cell = triggerBtn.closest("td");
    if (cell instanceof HTMLTableCellElement) {
      const tmpl = cell.querySelector(".repr-template");
      repr = parseTemplateJson(tmpl);
    }

    activeRowContext = { uri, graph, external, repr };

    const externalLink = document.getElementById("actions-external");
    if (externalLink instanceof HTMLAnchorElement) {
      externalLink.href = external || "#";
    }

    showActionsMenuAt(triggerBtn);
    return;
  }

  // Actions in the portal menu
  const menuBtn = target.closest(".actions-menu-btn");
  if (menuBtn instanceof HTMLButtonElement) {
    const action = menuBtn.getAttribute("data-action") || "";
    const ctx = activeRowContext;
    hideActionsMenu();

    if (!ctx) return;

    if (action === "similar") {
      const featureTypeEl = document.querySelector("[name='feat_type']");
      const featureValueEl = document.querySelector("[name='feat_value']");

      if (featureTypeEl instanceof HTMLSelectElement) featureTypeEl.value = "node";
      if (featureValueEl instanceof HTMLInputElement) featureValueEl.value = ctx.uri;

      submitQueryForm();
      return;
    }

    if (action === "restrict-graph") {
      setGraphMode("include");

      const graphInputs = document.querySelectorAll(".graphs-list input[type='checkbox']");
      graphInputs.forEach((el) => {
        if (el instanceof HTMLInputElement) el.checked = false;
      });

      if (ctx.graph) {
        const toCheck = document.querySelector(`.graphs-list input[type='checkbox'][value="${CSS.escape(ctx.graph)}"]`);
        if (toCheck instanceof HTMLInputElement) toCheck.checked = true;
      }

      submitQueryForm();
      return;
    }

    if (action === "exclude-graph") {
      setGraphMode("exclude");

      if (ctx.graph) {
        const toCheck = document.querySelector(`.graphs-list input[type='checkbox'][value="${CSS.escape(ctx.graph)}"]`);
        if (toCheck instanceof HTMLInputElement) toCheck.checked = true;
      }

      submitQueryForm();
      return;
    }

    if (action === "show-repr") {
      const dialog = document.getElementById("repr-dialog");
      const uriEl = document.getElementById("repr-dialog-uri");
      const bodyEl = document.getElementById("repr-dialog-body");

      if (uriEl) uriEl.textContent = ctx.uri;
      if (bodyEl) bodyEl.textContent = ctx.repr;

      if (dialog instanceof HTMLDialogElement) {
        dialog.showModal();
      }
      return;
    }

    return;
  }

  // Click outside closes the portal menu.
  const menu = document.getElementById("actions-menu");
  if (menu instanceof HTMLElement && !menu.hidden) {
    if (!target.closest("#actions-menu")) {
      hideActionsMenu();
    }
  }
});

window.Frink = {
  updateQueryParams,
}
