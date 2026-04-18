export interface SimParams {
  particleCount: number;
  h: number;
  restDensity: number;
  dt: number;
  solverIters: number;
}

export function defaultParams(): SimParams {
  return {
    particleCount: 1 << 14,
    h: 0.1,
    restDensity: 1000,
    dt: 1 / 60,
    solverIters: 3,
  };
}
