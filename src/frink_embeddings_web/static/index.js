function updateQueryParams() {
  const url = new URL(window.location.toString());
  const params = url.searchParams;
  const featureType = document.querySelector("#features [name='feat_type']").value
  const featureValue = document.querySelector("#features [name='feat_value']").value

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
