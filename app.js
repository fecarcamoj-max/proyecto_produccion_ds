// URL relativa: funciona en localhost, Docker y detrás de un proxy HTTPS.
const API_URL = "/predict";

document.querySelector("#consultar").addEventListener("click", async () => {
  const form = document.querySelector("#form-cliente");
  const result = document.querySelector("#resultado");
  if (!form.reportValidity()) return;

  const raw = Object.fromEntries(new FormData(form).entries());
  raw.tenure = Number(raw.tenure);
  raw.MonthlyCharges = Number(raw.MonthlyCharges);
  raw.TotalCharges = Number(raw.TotalCharges);
  raw.SeniorCitizen = Number(raw.SeniorCitizen);
  result.textContent = "Consultando…";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(raw),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Error de consulta");
    result.textContent = `Predicción: ${data.prediction === "Yes" ? "Fuga" : "No fuga"} | ` +
      `Probabilidad de fuga: ${(data.churn_probability * 100).toFixed(1)}% | Riesgo ${data.risk}`;
  } catch (error) {
    result.textContent = `No se pudo consultar la API: ${error.message}`;
  }
});
