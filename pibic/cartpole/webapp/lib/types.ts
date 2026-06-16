export interface TrajectoryFrame {
  x:         number;
  x_dot:     number;
  theta:     number;
  theta_dot: number;
  action:    number;
  q0:        number;
  q1:        number;
  q2?:       number;
  q3?:       number;
  q4?:       number;
  failed?:   boolean;  // true nos frames pós-colapso do contraexemplo
}

export interface Episode {
  seed:            number;
  score:           number;
  type:            'controlled' | 'random' | 'counterexample';
  trajectory:      TrajectoryFrame[];
  critical_frame?: number;
  esbmc_note?:     string;
  esbmc_property?: string;
}

export interface ClosedLoopProperty {
  result:         string;
  counterexample: string;
}

export interface ClosedLoopVerification {
  property_a_right: ClosedLoopProperty;
  property_a_left:  ClosedLoopProperty;
  property_b_safety: ClosedLoopProperty;
}

export interface NeuronInfo {
  id:       number;
  bias_q88: number;
  status:   string;
}

export interface TrainingPoint {
  episode: number;
  score:   number;
  avg100:  number;
  epsilon: number;
}

export interface SimulationData {
  model_info: {
    architecture:       string;
    training_episodes:  number;
    final_avg_score:    number;
  };
  training_history?: TrainingPoint[];
  episodes: Episode[];
  verification: {
    dead_neurons: {
      total:   number;
      dead:    number[];
      neurons: NeuronInfo[];
    };
    saturation: {
      saturated_neurons: number[];
      output_status:     string;
    };
  };
  closed_loop_verification?: ClosedLoopVerification;
}
