//
// 超シンプルなベイズ
//

// ベータ分布のパラメータ
struct Beta {
    alpha: f64,
    beta: f64,
}

#[allow(dead_code)]
impl Beta {
    fn new(alpha: f64, beta: f64) -> Self {
        Beta { alpha, beta }
    }

    #[allow(unused_parens)]
    fn update(&mut self, alpha: f64, beta: f64) {
        self.alpha += alpha;
        self.beta += (beta - alpha);
    }

    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    fn print_params(&self) {
        println!("alpha = {}, beta = {}", self.alpha, self.beta);
    }
}

#[allow(dead_code)]
pub fn run_bays_multi_observation() {
    println!("コイントスをn回試行し、m回成功したかを確認する実験を行います");
    println!("-------------------------------------");

    let mut beta = Beta::new(1.0, 1.0);

    let observations = vec![
        (5., 10.),
        (20., 100.),
        (30., 100.),
        (40., 100.),
        (90., 100.),
    ];

    for observation in observations {
        let observed_successes = observation.0;
        let total_trials = observation.1;

        println!(
            "観測された成功回数: {}, 試行回数: {}",
            observed_successes, total_trials
        );
        println!(
            "事後分布のパラメータ: α = {:.1}, β = {:.1}",
            beta.alpha, beta.beta
        );
        println!("推定される成功確率: {:.2}", beta.mean());
        println!("-------------------------------------");

        beta.update(observed_successes, total_trials);
    }
}

#[allow(dead_code)]
pub fn run_bays_single_observation() {
    let mut beta = Beta::new(1.0, 1.0);

    // 観測データ: 10回中7回成功
    let total_trials = 10.;
    let observed_successes = 7.;

    beta.update(observed_successes, total_trials);
    println!("mean = {:.2}", beta.mean());
}
