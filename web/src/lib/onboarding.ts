const REQUIRED_COLUMNS = ["date", "ticker", "action", "price"];

export function parseCsvHeaders(text: string): string[] {
  const [headerLine] = text.split(/\r?\n/);
  if (!headerLine) return [];
  return headerLine
    .split(",")
    .map((value) => value.replace(/^"|"$/g, "").trim().toLowerCase())
    .filter(Boolean);
}

export function validateOnboardingCsvHeaders(headers: string[]): string[] {
  const missing = REQUIRED_COLUMNS.filter((col) => !headers.includes(col));
  const hasQuantity = headers.includes("quantity") || headers.includes("shares");
  if (!hasQuantity) missing.push("quantity (or shares)");
  return missing;
}
