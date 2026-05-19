export interface TrajectoryFrame {
  x:         number;
  x_dot:     number;
  theta:     number;
  theta_dot: number;
  action:    number;
  q0:        number;
  q1:        number;
}

export interface Episode {
  seed:       number;
  score:      number;
  trajectory: TrajectoryFrame[];
}

export interface NeuronInfo {
  id:       number;
  bias_q88: number;
  status:   string;
}

export interface SimulationData {
  model_info: {
    architecture:       string;
    training_episodes:  number;
    final_avg_score:    number;
  };
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
}
