interface MetricProps {
  label: string;
  value: string | number;
  wide?: boolean;
  accent?: boolean;
}

export default function Metric({ label, value, wide = false, accent = false }: MetricProps) {
  return (
    <div className={`metric-card${wide ? ' wide' : ''}${accent ? ' accent' : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
