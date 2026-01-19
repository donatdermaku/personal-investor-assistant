(function () {
  function renderLineChart(canvas, series, options) {
    if (!canvas || !series || !series.length) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 30;
    const values = series.map(p => p.value).filter(v => Number.isFinite(v));
    if (!values.length) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = options?.stroke || '#0b5d4a';
    ctx.lineWidth = 2;

    const points = series.map((p, i) => {
      const x = padding + (i / (series.length - 1 || 1)) * (width - padding * 2);
      const y = height - padding - ((p.value - min) / range) * (height - padding * 2);
      return { x, y };
    });

    ctx.beginPath();
    points.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.stroke();

    ctx.fillStyle = '#5c5c5c';
    ctx.font = '11px IBM Plex Sans';
    ctx.fillText(min.toFixed(2), 6, height - padding);
    ctx.fillText(max.toFixed(2), 6, padding + 4);
  }

  function renderMultiSeries(canvas, seriesList) {
    if (!canvas || !seriesList || !seriesList.length) return;
    const colors = ['#0b5d4a', '#1f7a8c', '#e07a5f', '#6d597a'];
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 30;

    const allValues = seriesList.flatMap(s => s.values.map(p => p.value).filter(v => Number.isFinite(v)));
    if (!allValues.length) return;
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    ctx.clearRect(0, 0, width, height);

    seriesList.forEach((series, idx) => {
      const values = series.values;
      if (!values.length) return;
      ctx.strokeStyle = colors[idx % colors.length];
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((p, i) => {
        const x = padding + (i / (values.length - 1 || 1)) * (width - padding * 2);
        const y = height - padding - ((p.value - min) / range) * (height - padding * 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  async function loadTickerData(path) {
    const resp = await fetch(path);
    if (!resp.ok) return null;
    return resp.json();
  }

  document.addEventListener('DOMContentLoaded', async () => {
    const container = document.querySelector('[data-ticker-chart]');
    if (!container) return;
    const dataPath = container.dataset.tickerChart;
    const payload = await loadTickerData(dataPath);
    if (!payload) return;

    const priceCanvas = document.getElementById('priceChart');
    renderLineChart(priceCanvas, payload.price || [], { stroke: '#0b5d4a' });

    const factorCanvas = document.getElementById('factorChart');
    const factors = (payload.factors || []).map(p => ({
      date: p.date,
      composite: p.value,
      value: p.value_score,
      quality: p.quality_score,
      momentum: p.momentum_score,
    }));
    const compositeSeries = factors.map(p => ({ date: p.date, value: p.composite }));
    const valueSeries = factors.map(p => ({ date: p.date, value: p.value }));
    const qualitySeries = factors.map(p => ({ date: p.date, value: p.quality }));
    const momentumSeries = factors.map(p => ({ date: p.date, value: p.momentum }));
    renderMultiSeries(factorCanvas, [
      { name: 'Composite', values: compositeSeries },
      { name: 'Value', values: valueSeries },
      { name: 'Quality', values: qualitySeries },
      { name: 'Momentum', values: momentumSeries },
    ]);
  });
})();
