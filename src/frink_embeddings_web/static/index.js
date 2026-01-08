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

window.Frink = {
  updateQueryParams,
}
