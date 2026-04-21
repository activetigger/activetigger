import React from 'react';
import {
  VictoryAxis,
  VictoryChart,
  VictoryLegend,
  VictoryLine,
  VictoryScatter,
  VictoryTheme,
} from 'victory';

interface LossChartProps {
  loss: LossData | null;
  xmax?: number;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

export const LossChart: React.FC<LossChartProps> = ({ loss, xmax }) => {
  const val_epochs = loss ? (Object.values(loss.epoch) as unknown as number[]) : [];
  const val_loss = loss ? (Object.values(loss.val_loss) as unknown as number[]) : [];
  const val_eval_loss = loss ? (Object.values(loss.val_eval_loss) as unknown as number[]) : [];

  const valLossData = val_epochs.map((epoch, i) => ({
    x: epoch as number,
    y: val_loss[i] as number,
  }));
  const valEvalLossData = val_epochs.map((epoch, i) => ({
    x: epoch as number,
    y: val_eval_loss[i] as number,
  }));

  const allYValues = [...valLossData.map((d) => d.y), ...valEvalLossData.map((d) => d.y)];

  const maxY = Math.max(...allYValues);

  const initial = { x: 0, y: Infinity };
  const minValLossPoint = valEvalLossData.reduce(
    (min, curr) => (curr.y < min.y ? curr : min),
    initial,
  );
  if (valEvalLossData.length < 1)
    return (
      <div className="alert alert-info m-3">
        Loss chart will be displayed when enough data is available
      </div>
    );

  return (
    <>
      <VictoryChart
        theme={VictoryTheme.material}
        minDomain={{ y: 0 }}
        maxDomain={{ x: xmax, y: maxY * 1.1 }}
        width={1000}
        height={500}
      >
        <VictoryAxis
          label="Epoch"
          style={{
            axisLabel: { padding: 30 },
          }}
        />
        <VictoryAxis
          dependentAxis
          label="Loss"
          style={{
            axisLabel: { padding: 40 },
          }}
        />
        <VictoryLine
          data={valLossData}
          style={{
            data: { stroke: '#0072B2', strokeWidth: 2 },
          }}
        />
        <VictoryScatter
          data={valLossData}
          size={5}
          symbol="circle"
          style={{
            data: { fill: '#0072B2' },
          }}
        />
        <VictoryLine
          data={valEvalLossData}
          style={{
            data: { stroke: '#D55E00', strokeWidth: 2, strokeDasharray: '8,4' },
          }}
        />
        <VictoryScatter
          data={valEvalLossData}
          size={6}
          symbol="triangleUp"
          style={{
            data: { fill: '#D55E00' },
          }}
        />
        <VictoryLine
          data={[
            { x: minValLossPoint.x, y: 0 },
            { x: minValLossPoint.x, y: maxY },
          ]}
          style={{
            data: { stroke: '#009E73', strokeWidth: 2, strokeDasharray: '5,5' },
          }}
        />
        <VictoryLegend
          x={100}
          y={0}
          centerTitle
          orientation="horizontal"
          gutter={20}
          style={{ border: { stroke: 'black' }, title: { fontSize: 10 } }}
          data={[
            { name: 'Train Loss', symbol: { fill: '#0072B2', type: 'circle' } },
            { name: 'Eval Loss', symbol: { fill: '#D55E00', type: 'triangleUp' } },
            {
              name: 'Best model',
              symbol: {
                fill: '#009E73',
                type: 'square',
                size: 3,
              },
            },
          ]}
          standalone={true}
        />
      </VictoryChart>
    </>
  );
};
