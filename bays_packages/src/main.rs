use rand::Rng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::ContinuousCDF;
use statrs::distribution::{Continuous, Normal as StatrsNormal};
mod modules;
use crate::modules::utils::*;

// ---------------------------------------------------------------------
// 事前、事後分布
trait BaysDistribution {
    fn pdf(&self, x: f64) -> f64;
    fn print(&self);
    fn update_parameters(&mut self, new_parameters: &Vec<f64>);
}
struct BaysNormalDistribution {
    mu: f64,
    sigma: f64,
}
impl BaysDistribution for BaysNormalDistribution {
    fn pdf(&self, x: f64) -> f64 {
        let normal = StatrsNormal::new(self.mu, self.sigma).unwrap();
        normal.pdf(x)
    }

    fn print(&self) {
        println!("parameter = {}, {}", self.mu, self.sigma);
    }

    fn update_parameters(&mut self, new_parameters: &Vec<f64>) {
        self.mu = new_parameters[0];
        self.sigma = new_parameters[1];
    }
}

fn normal(mu: f64, sigma: f64) -> Box<dyn BaysDistribution>{
    Box::new(
        BaysNormalDistribution{ mu: mu, sigma: sigma }
    )
}

// ---------------------------------------------------------------------
// 分布を取り出すインターフェイス関数
fn get_distribution(name: &str, parameters: &Vec<f64>) -> Box<dyn BaysDistribution> {
    if name == "normal" {
        normal(parameters[0], parameters[1])
    } else {
        normal(parameters[0], parameters[1]) // 暫定的
    }
}

// fn update_parameter_vector(parameters: &mut Vec<f64>, new_values: &[f64]) {
//     let len = parameters.len().min(new_values.len()); // 更新できる範囲の長さを決定

//     for i in 0..len {
//         parameters[i] = new_values[i];
//     }
// }

// ---------------------------------------------------------------------
fn metropolis_hastings(
    iterations: usize,
    burn_in: usize,
    init: f64,
    data: f64,
    sigma: f64,
    prior_mu: f64,
    prior_sigma: f64,
) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let mut current = init;
    let proposal_scale: f64 = 100.0;

    // 分布を取得
    let mut parameters = vec![current, sigma];
    let mut current_distribution  = get_distribution("normal", &parameters);
    let mut proposal_distribution = get_distribution("normal", &vec![prior_mu, prior_sigma]);

    // 提案分布としてrand_distrのNormalを使用
    let distribution = Normal::new(0.0, proposal_scale).unwrap();

    let mut acceptance_ratios = Vec::new();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + distribution.sample(&mut rng);

        // 尤度計算
        proposal_distribution.update_parameters(&vec![proposal, sigma]);
        let current_likelihood  = current_distribution.pdf(data);
        let proposal_likelihood = proposal_distribution.pdf(data);
        // 事前分布の計算
        let current_prior  = current_distribution.pdf(data);
        let proposal_prior = proposal_distribution.pdf(proposal);

        // 事後分布の計算
        let current_posterior  = current_likelihood * current_prior;
        let proposal_posterior = proposal_likelihood * proposal_prior;

        let acceptance_ratio = (proposal_posterior / current_posterior).min(1.0);

        if rng.gen::<f64>() < acceptance_ratio {
            current = proposal;
        }
        acceptance_ratios.push(acceptance_ratio);
        samples.push(current);
    }
    //    println!("MCMC acceptance_ratio : {:.4} +- {:.4}", mean_normal_dist(&acceptance_ratios), stddev_normal_dist(&acceptance_ratios));
    //バーンイン
    if samples.len() > burn_in {
        samples.split_off(burn_in)
    } else {
        samples
    }
}


// 薄化（間引き）を行う関数
pub fn thin_samples(samples: &[f64], interval: usize) -> Vec<f64> {
    samples.iter().step_by(interval).copied().collect()
}

pub fn autocorrelation(samples: &Vec<f64>, lag: usize) -> f64 {
    // samplesの平均
    let n = samples.len();
    let mean: f64 = samples.iter().sum::<f64>() / n as f64;

    let mut shifted_samples = samples[..n - lag].to_vec();
    let mut sliced_samples = samples[lag..].to_vec();

    subtract_constant(&mut shifted_samples, mean);
    subtract_constant(&mut sliced_samples, mean);

    // 自己相関の分子
    let numerator: f64 = hadamard_product(&sliced_samples, &shifted_samples)
        .iter()
        .sum();

    // 自己相関の分母
    let mut original_samples = samples.to_vec();
    subtract_constant(&mut original_samples, mean);
    let denominator: f64 = dot_product(&original_samples, &original_samples);
    numerator / denominator
}

// ---------------------------------------------------------------------

// Geweke診断
// MCMCで生成されたチェーンの前半と後半で平均値の差の検定を実施する。
// なぜなら、MCMCが収束している場合、チェーンのどこをとってもそこから計算された平均は同じはず。
// したがって平均の差の検定をやったとき、有意差がでていれば収束していないとなる

#[allow(dead_code)]
pub fn geweke_diagnostic_p_value(samples: &Vec<f64>, first_frac: f64, last_frac: f64) -> f64{
    let n = samples.len();
    let first_n = (n as f64 * first_frac) as usize;
    let last_n = (n as f64 * last_frac) as usize;

    // 前半と後半の平均と標準誤差の計算
    let mean_a = samples[0..first_n].iter().sum::<f64>() / first_n as f64;
    let mean_b = samples[(n - last_n)..n].iter().sum::<f64>() / last_n as f64;

    let se_a = (samples[0..first_n].iter()
        .map(|x| (x - mean_a).powi(2))
        .sum::<f64>() / (first_n as f64 - 1.0)).sqrt();
    let se_b = (samples[(n - last_n)..n].iter()
        .map(|x| (x - mean_b).powi(2))
        .sum::<f64>() / (last_n as f64 - 1.0)).sqrt();

    let z_score = (mean_a - mean_b) / ((se_a.powi(2) + se_b.powi(2)).sqrt());

    // 標準正規分布の作成
    let normal_dist = StatrsNormal::new(0.0, 1.0).unwrap();

    // p値の計算（両側検定）
    let p_value = 2.0 * (1.0 - normal_dist.cdf(z_score.abs()));
    p_value
}


// 非効率性因子(有効サンプル数を使った収束診断)
// 自己相関時間の計算
pub fn calculate_act(samples: &Vec<f64>, max_lag: usize) -> f64 {
    let n = samples.len() as f64;

    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|x|(x - mean).powi(2)).sum::<f64>() / n;

    let mut autocorrelations = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in 0..(samples.len() - lag) {
            sum += (samples[i] - mean) * (samples[i + lag] - mean);
        }
        autocorrelations[lag] = sum / (n * variance);
    }

    //　自己相関時間の計算
    let act: f64 = 1.0 + 2.0 * autocorrelations[1..].iter().sum::<f64>();
    act
}

// IF(非効率性因子)の計算
pub fn inefficiency_factor_diagnostic(samples: &Vec<f64>, max_lag: usize) -> f64 {
    let act = calculate_act(&samples, max_lag);
    let n = samples.len();
    let if_value = n as f64 / act;
    if_value
}

// 95%信用区間の計算
pub fn credible_interval(samples: &[f64]) -> (f64, f64) {
    let mut sorted_samples = samples.to_vec();
    sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (0.025 * sorted_samples.len() as f64).floor() as usize;
    let upper_idx = (0.975 * sorted_samples.len() as f64).floor() as usize;

    (sorted_samples[lower_idx], sorted_samples[upper_idx])
}

// ---------------------------------------------------------------------
fn main() {
    // let mut parameters = vec![1.0, 5.0];
    // let mut dist = get_distribution("normal", &parameters);
    // dist.print();
    // println!("{}", dist.pdf(1.0));

    // dist.update_parameters(&vec![10.0, 20.0]);
    // dist.print();
    // println!("{}", dist.pdf(1.0));

    // metropolis_hastings(100, 2, 10.0, 11.0, 100.0, 20.0, 400.0);

    println!("[サンプル] ベイズ推定で顧客の平均購入金額を推定する");
    // 目的：顧客の平均購入金額を推定する

    // ｎ回の観測データ（顧客の購入金額-ドルスプシで乱数から生成した値
    // =ARRAYFORMULA(NORMINV(RANDARRAY(20),D$11,D$12))
    let observations = vec![
                            41.0, 47.0, 57.0, 66.0, 49.0, 28.0, 47.0, 58.0, 54.0, 51.0,
                            51.0, 45.0, 44.0, 56.0, 61.0, 60.0, 51.0, 48.0, 57.0, 48.0,
                            65.0, 46.0, 51.0, 43.0, 43.0, 54.0, 40.0, 53.0, 75.0, 50.0,
                            35.0, 43.0, 53.0, 68.0, 48.0, 46.0, 44.0, 34.0, 48.0, 60.0
                        ];

    // 事前分布：平均1ドル、標準偏差100ドルの正規分布
    let mut prior_mu = 10.0;
    let mut prior_sigma = 100.0;

    // ベイズ更新のイテレーション数
    let iterations = 100000; // メトロポリス・ヘイスティングスのサンプル数 100000
    let sigma = 10.0; // 尤度の標準偏差（観測誤差として仮定）
    let mut posterior_mean = 0.;
    // MCMCの自己相関を排除するパラメータ
    let thinning_interval = 200; // 薄化の間隔（例：10サンプルに1つを選択） 1000
    let burn_in = 10000; // バーンイン

    // 初期値を異なる3つのチェーンで設定
    let init_values = vec![30.0, 50.0, 70.0]; // 初期値の異なる設定

    println!("自己相関抑制のためのパラメータ");
    println!("  薄化インターバル : {}", thinning_interval);
    println!("  バーンイン : {}", burn_in);

    for (i, &data) in observations.iter().enumerate() {
        println!("-------------------------------");
        println!("# Processing observation {}: {}", i + 1, data);
        // 各初期値で独立したチェーンを実行
        let mut all_samples = Vec::new();

        for &init in init_values.iter() {
            // メトロポリス・ヘイスティングスの実行
            // MCMCの初期値を1つにすると、生成した数値の分散が早期にゼロになってしまう
            let samples = metropolis_hastings(iterations, burn_in, init, data, sigma, prior_mu, prior_sigma);
            all_samples.extend(samples); // 結果を統合
        }

        // サンプルの薄化（間引き）
        let thinned_samples = thin_samples(&all_samples, thinning_interval);

        // 事後分布の平均を計算
        posterior_mean = mean_normal_dist(&thinned_samples);

        //
        // 次の更新に備えて事前分布の平均を事後分布の平均に更新
        prior_mu = posterior_mean;
        // 次の更新に備えて事前分布の標準偏差をサンプルの標準偏差に更新
        prior_sigma = stddev_normal_dist(&thinned_samples);

        println!(
            "Updated posterior mean after observing {}: {:.2} +- {:.4}",
            data, posterior_mean, prior_sigma
        );
        // 自己相関の計算と表示（遅延1～5まで）
        for lag in 1..=5 {
            let ac = autocorrelation(&thinned_samples, lag);
            println!("   Autocorrelation at lag {}: {:.4}", lag, ac);
        }

        let if_value = inefficiency_factor_diagnostic(&thinned_samples, 5);
        // Geweke診断
        println!("   Geweke p-value = {:.4}", geweke_diagnostic_p_value(&thinned_samples, 0.2, 0.5));
        // IF
        println!("   IF-value       = {:.2}({}個/{}の独立したサンプル)", if_value, iterations / if_value as usize, iterations);
        // 95%信用区間の計算
        // 例：平均日次売上金額が95%の確率で約60.12ドルから82.45ドルの間にある
        let (lower_bound, upper_bound) = credible_interval(&thinned_samples);
        println!("   95% Credible Interval: ({:.2}, {:.2})", lower_bound, upper_bound);
    }
    println!("-------------------------------");
    println!("観測値(顧客の購入金額-ドル) : {:?}", observations);
    println!("観測値の平均: {:.2}", mean_normal_dist(&observations));
    println!("ベイズ推定された事後分布の平均値: {:.2}", posterior_mean);
    println!("事後分布の平均値 - 単純平均: {:.2}", posterior_mean-mean_normal_dist(&observations));
}

