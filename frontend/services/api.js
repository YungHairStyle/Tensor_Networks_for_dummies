export async function postJSON(path, payload) {
  const t0 = performance.now();
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const t1 = performance.now();

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }

  const data = await res.json();
  return { data, ms: (t1 - t0) };
}