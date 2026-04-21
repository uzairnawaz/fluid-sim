export interface SimParams {
    particleCount: number;
    h: number;
    restDensity: number;
    dt: number;
    solverIters: number;
}

export function defaultParams(): SimParams {
    return {
        particleCount: 1 << 13,
        h: 0.1,
        restDensity: 50000,
        dt: 1 / 120,
        solverIters: 3,
    };
}
