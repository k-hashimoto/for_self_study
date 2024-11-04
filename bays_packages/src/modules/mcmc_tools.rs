use statrs::distribution::Normal as StatrsNormal; // Continuous
use statrs::distribution::ContinuousCDF;
use crate::modules::utils::*;

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
