export function downsampleSeries<T>(series: T[], maxPoints: number): T[] {
    if (!Array.isArray(series) || maxPoints <= 0 || series.length <= maxPoints) {
        return series;
    }
    const step = Math.ceil(series.length / maxPoints);
    return series.filter((_, index) => index % step === 0 || index === series.length - 1);
}
