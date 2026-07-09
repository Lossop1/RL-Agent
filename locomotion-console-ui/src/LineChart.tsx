// Zero-dependency SVG line chart for the live metric stream.
// Supports multiple aligned series so monitoring can show raw + smoothed
// reward and auxiliary signals without adding chart dependencies.

interface Series {
  values: Array<number | null>;
  color: string;
  label?: string;
  width?: number;
  dashed?: boolean;
}

interface Props {
  values?: Array<number | null>;
  series?: Series[];
  width?: number;
  height?: number;
  color?: string;
  label?: string;
  emptyLabel?: string;
}

export default function LineChart({
  values = [],
  series,
  width = 720,
  height = 160,
  color = "#4ea1ff",
  label,
  emptyLabel = "暂无数据",
}: Props) {
  const pad = 8;
  const allSeries = series?.length ? series : [{ values, color, label }];
  const flat = allSeries.flatMap((item) => item.values).filter(isFiniteNumber);
  const maxLength = Math.max(0, ...allSeries.map((item) => item.values.length));
  let min = 0;
  let max = 1;
  if (flat.length > 0) {
    min = Math.min(...flat);
    max = Math.max(...flat);
  }
  const span = max - min || 1;
  function pathFor(source: Array<number | null>) {
    let segmentOpen = false;
    return source
      .map((v, i) => {
        if (!isFiniteNumber(v)) {
          segmentOpen = false;
          return "";
        }
        const x = pad + (i / Math.max(1, source.length - 1)) * (width - 2 * pad);
        const y = height - pad - ((v - min) / span) * (height - 2 * pad);
        const command = segmentOpen ? "L" : "M";
        segmentOpen = true;
        return `${command}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");
  }
  function axisLabel(value: number) {
    const maxAbs = Math.max(Math.abs(max), Math.abs(min), Math.abs(span));
    if (maxAbs < 1) return value.toFixed(3);
    if (maxAbs < 10) return value.toFixed(2);
    if (maxAbs < 100) return value.toFixed(1);
    return value.toFixed(0);
  }
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label || "折线图"}
      className="line-chart"
      preserveAspectRatio="none"
    >
      {label && (
        <text x={pad + 2} y={16} fill="#7a8699" fontSize={11} fontFamily="monospace">
          {label}
        </text>
      )}
      {flat.length > 0 && (
        <>
          <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} stroke="#2b3129" strokeWidth={1} />
          <line x1={pad} x2={width - pad} y1={pad} y2={pad} stroke="#2b3129" strokeWidth={1} />
          <line x1={pad} x2={width - pad} y1={height / 2} y2={height / 2} stroke="#222720" strokeWidth={1} />
        </>
      )}
      {allSeries.map((item, index) => (
        item.values.filter(isFiniteNumber).length > 1
          ? (
              <path
                key={`${item.label ?? "series"}-${index}`}
                d={pathFor(item.values)}
                fill="none"
                stroke={item.color}
                strokeWidth={item.width ?? 1.8}
                strokeDasharray={item.dashed ? "5 4" : undefined}
                opacity={item.dashed ? 0.75 : 1}
              />
            )
          : null
      ))}
      {maxLength === 0 && (
        <text x={width / 2} y={height / 2} fill="#7a8699" fontSize={12}
              fontFamily="sans-serif" textAnchor="middle">
          {emptyLabel}
        </text>
      )}
      {flat.length > 0 && (
        <>
          <text x={width - pad} y={16} fill="#7a8699" fontSize={11}
                fontFamily="monospace" textAnchor="end">
            {axisLabel(max)}
          </text>
          <text x={width - pad} y={height - pad} fill="#7a8699" fontSize={11}
                fontFamily="monospace" textAnchor="end">
            {axisLabel(min)}
          </text>
        </>
      )}
    </svg>
  );
}

function isFiniteNumber(value: number | null): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
