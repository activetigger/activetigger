import { FC } from 'react';
import { VictoryAxis, VictoryBar, VictoryChart, VictoryTheme, VictoryTooltip } from 'victory';

export type UserActivityPoint = {
  hour: string;
  annotations: number;
};

/**
 * Hourly annotation counts for a single user (same layout as the
 * instance-wide chart on the monitor page)
 */
export const UserActivityChart: FC<{ points: UserActivityPoint[] }> = ({ points }) => {
  if (!points || points.length === 0) {
    return <div className="alert alert-info m-3">No activity in the selected period.</div>;
  }

  const data = points.map((p, i) => ({
    i,
    date: new Date(p.hour),
    annotations: p.annotations,
  }));

  // One tick per day (every 24 hours), labelled with the day boundary
  const dayTickValues = data.filter((d) => d.date.getUTCHours() === 0).map((d) => d.i);
  const dayTickFormat = (i: number) => {
    const d = data[i]?.date;
    return d ? `${d.getUTCMonth() + 1}/${d.getUTCDate()}` : '';
  };

  return (
    <VictoryChart
      theme={VictoryTheme.material}
      domainPadding={{ x: 5 }}
      width={1000}
      height={170}
      padding={{ top: 10, bottom: 35, left: 55, right: 15 }}
    >
      <VictoryAxis
        tickValues={dayTickValues}
        tickFormat={dayTickFormat}
        style={{ tickLabels: { fontSize: 10 } }}
      />
      <VictoryAxis
        dependentAxis
        label="Annotations"
        style={{ axisLabel: { padding: 40, fontSize: 12 }, tickLabels: { fontSize: 10 } }}
      />
      <VictoryBar
        data={data}
        x="i"
        y="annotations"
        style={{ data: { fill: '#0072B2' } }}
        labels={({ datum }) =>
          `${datum.date.toISOString().slice(0, 13)}:00\nAnnotations: ${datum.annotations}`
        }
        labelComponent={<VictoryTooltip />}
      />
    </VictoryChart>
  );
};
