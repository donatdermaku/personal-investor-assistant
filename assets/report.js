(function () {
  const table = document.querySelector('[data-sortable]');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  const headers = table.querySelectorAll('th[data-key]');
  let sortKey = null;
  let sortDir = 1;

  function parseValue(value) {
    const num = parseFloat(value.replace(/[%,$]/g, ''));
    if (!Number.isNaN(num)) return num;
    return value.toLowerCase();
  }

  function sortRows(key) {
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const aVal = parseValue(a.dataset[key] || '');
      const bVal = parseValue(b.dataset[key] || '');
      if (aVal < bVal) return -1 * sortDir;
      if (aVal > bVal) return 1 * sortDir;
      return 0;
    });
    rows.forEach(row => tbody.appendChild(row));
  }

  headers.forEach(header => {
    header.addEventListener('click', () => {
      const key = header.dataset.key;
      if (sortKey === key) {
        sortDir *= -1;
      } else {
        sortKey = key;
        sortDir = -1;
      }
      sortRows(key);
    });
  });

  const filterInput = document.querySelector('[data-filter]');
  if (filterInput) {
    filterInput.addEventListener('input', () => {
      const query = filterInput.value.trim().toLowerCase();
      Array.from(tbody.querySelectorAll('tr')).forEach(row => {
        const ticker = (row.dataset.ticker || '').toLowerCase();
        const note = (row.dataset.note || '').toLowerCase();
        const hit = ticker.includes(query) || note.includes(query);
        row.style.display = hit ? '' : 'none';
      });
    });
  }
})();
