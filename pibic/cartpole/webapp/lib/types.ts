// ─── Controller interface (LSP: float and Q8.8 are interchangeable) ─────────

export interface IController {
  getForce(state: [number, number, number, number]): number;
  readonly name: string;
  readonly isVerified: boolean;
}

// ─── Verification data (ISP: small focused interfaces) ──────────────────────

export interface NeuronInfo {
  id:       number;
  bias_q88: number;
  status:   string;
}

export interface NeuronVerification {
  total:   number;
  dead:    number[];
  neurons: NeuronInfo[];
}

export interface ClosedLoopProperty {
  result:         string;
  counterexample: string;
}

export interface DDPGClosedLoopVerification {
  property_a_right:  ClosedLoopProperty;
  property_a_left:   ClosedLoopProperty;
  property_b_safety: ClosedLoopProperty;
  property_c_bounds: ClosedLoopProperty;
}

export interface Counterexample {
  property:          string;
  description:       string;
  state_str:         string;
  expected_behavior: string;
}

export interface DDPGVerificationData {
  model_info: {
    architecture:      string;
    controller_type:   string;
    quantization:      string;
    training_episodes: number;
    final_avg_score:   number;
  };
  verification: {
    dead_neurons_l1: NeuronVerification;
    dead_neurons_l2: NeuronVerification;
    saturation: {
      saturated_neurons: number[];
      output_status:     string;
    };
  };
  closed_loop_verification: DDPGClosedLoopVerification;
  counterexamples:          Counterexample[];
}

// ─── Training ───────────────────────────────────────────────────────────────

export interface TrainingPoint {
  episode:  number;
  score:    number;
  avg100:   number;
  epsilon?: number;
}

// ─── Legacy DQN types ───────────────────────────────────────────────────────

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
  failed?:   boolean;
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

export interface SimulationData {
  model_info: {
    architecture:       string;
    training_episodes:  number;
    final_avg_score:    number;
  };
  training_history?: TrainingPoint[];
  episodes: Episode[];
  verification: {
    dead_neurons: NeuronVerification;
    saturation: {
      saturated_neurons: number[];
      output_status:     string;
    };
  };
  closed_loop_verification?: {
    property_a_right:  ClosedLoopProperty;
    property_a_left:   ClosedLoopProperty;
    property_b_safety: ClosedLoopProperty;
  };
}
