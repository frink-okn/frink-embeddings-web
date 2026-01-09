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

document.addEventListener("click", (e) => {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;

  const btn = target.closest(".copy-btn");
  if (!btn) return;

  const text = btn.getAttribute("data-copy") || "";
  copyToClipboard(text, btn);
});

window.Frink = {
  updateQueryParams,
}
