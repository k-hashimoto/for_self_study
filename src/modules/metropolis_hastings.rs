// メトロポリス・ヘイスティングスアルゴリズムの例
// メトロポリス・ヘイスティングス (Metropolis-Hastings, MH) アルゴリズムは、サンプリング対象の確率分布から直接サンプルを生成できない場合に、近似的にサンプルを生成するためのMCMC法の一つです。
// ここではMH法をつかって,正規分布に従うサンプルを生成する

use rand::Rng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::ContinuousCDF;
use statrs::distribution::{Continuous, Normal as StatrsNormal};

#[allow(unused_imports)]
use crate::modules::utils::{
    dot_product, hadamard_product, mean_normal_dist, stddev_normal_dist, subtract_constant,
};

#[allow(dead_code)]
pub fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let normal = StatrsNormal::new(mu, sigma).unwrap();
    normal.pdf(x)
}

#[allow(dead_code)]
pub fn simple_metropolis_hastings(iterations: usize, init: f64, data: f64, sigma: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let mut current = init;

    let proposal_distribution = Normal::new(0.0, 1.0).unwrap();

    for _ in 0..iterations {
        let proposal = current + proposal_distribution.sample(&mut rng);

        let current_likelihood = normal_pdf(data, current, sigma);
        let proposal_likelihood = normal_pdf(data, proposal, sigma);

        let acceptance_ratio = (proposal_likelihood / current_likelihood).min(1.0);

        if rng.gen::<f64>() < acceptance_ratio {
            current = proposal;
        }
        samples.push(current);
    }
    samples
}

#[allow(dead_code)]
pub fn metropolis_hastings_normal_dist(
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

    // 提案分布としてrand_distrのNormalを使用
    let proposal_distribution = Normal::new(0.0, proposal_scale).unwrap();

    let mut acceptance_ratios = Vec::new();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + proposal_distribution.sample(&mut rng);

        // 尤度計算
        let current_likelihood = normal_pdf(data, current, sigma);
        let proposal_likelihood = normal_pdf(data, proposal, sigma);

        // 事前分布の計算
        let current_prior = normal_pdf(current, prior_mu, prior_sigma);
        let proposal_prior = normal_pdf(proposal, prior_mu, prior_sigma);

        // 事後分布の計算
        let current_posterior = current_likelihood * current_prior;
        let proposal_posterior = proposal_likelihood * proposal_prior;

        let acceptance_ratio = (proposal_posterior / current_posterior).min(1.0);

        if rng.gen::<f64>() < acceptance_ratio {
            current = proposal;
        }
        acceptance_ratios.push(acceptance_ratio);
        samples.push(current);
    }
    println!("MCMC acceptance_ratio : {:.4} +- {:.4}", mean_normal_dist(&acceptance_ratios), stddev_normal_dist(&acceptance_ratios));
    //バーンイン
    if samples.len() > burn_in {
        samples.split_off(burn_in)
    } else {
        samples
    }
}

#[allow(dead_code)]
pub fn metropolis_hastings_normal_dist_dev(
    iterations: usize,
    burn_in: usize,
    init: f64,
    data: f64,
    sigma: f64,
    prior_mu: f64,
    prior_sigma: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let mut current = init;
    let proposal_scale: f64 = 10.0;

    let mut acceptance_ratios = Vec::new();

    // 提案分布としてrand_distrのNormalを使用
    let proposal_distribution = Normal::new(0.0, proposal_scale).unwrap();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + proposal_distribution.sample(&mut rng);

        // 尤度計算
        let current_likelihood = normal_pdf(data, current, sigma);
        let proposal_likelihood = normal_pdf(data, proposal, sigma);

        // 事前分布の計算
        let current_prior = normal_pdf(current, prior_mu, prior_sigma);
        let proposal_prior = normal_pdf(proposal, prior_mu, prior_sigma);

        // 事後分布の計算
        let current_posterior = current_likelihood * current_prior;
        let proposal_posterior = proposal_likelihood * proposal_prior;

        let acceptance_ratio = (proposal_posterior / current_posterior).min(1.0);

        if rng.gen::<f64>() < acceptance_ratio {
            current = proposal;
        }

        samples.push(current);
        acceptance_ratios.push(acceptance_ratio);
    }
    //バーンイン
    if samples.len() > burn_in {
        (samples.split_off(burn_in), acceptance_ratios)
    } else {
        (samples, acceptance_ratios)
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

// 自己相関の計算
// ChatGPTが出力した実装。確認用。
#[allow(dead_code)]
pub fn autocorrelation_v2(samples: &[f64], lag: usize) -> f64 {
    let n = samples.len();
    let mean: f64 = samples.iter().sum::<f64>() / n as f64;

    let numerator: f64 = (0..n - lag)
        .map(|i| (samples[i] - mean) * (samples[i + lag] - mean))
        .sum();

    let denominator: f64 = samples.iter().map(|&x| (x - mean).powi(2)).sum();
    //    println!("{} / {} {}", numerator, denominator, samples[1] - mean);


    // todo
    // samples[i] - meanがほぼ0なのでゼロ除算エラーが起きる
    // samplesのデータの分散がほぼゼロになってるはず。


    numerator / denominator
}

#[allow(dead_code)]
pub fn metropolis_hastings_normal_dist_v2(
    iterations: usize,
    burn_in: usize,
    init: f64,
    data: f64,
    sigma: f64,
    prior_mu: f64,
    prior_sigma: f64
) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let mut current = init;

    // 提案分布としてrand_distrのNormalを使用
    let proposal_distribution = Normal::new(0.0, 1.0).unwrap();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + proposal_distribution.sample(&mut rng);

        // 尤度の計算
        let current_likelihood = normal_pdf(data, current, sigma);
        let proposal_likelihood = normal_pdf(data, proposal, sigma);

        // 事前分布の計算
        let current_prior = normal_pdf(current, prior_mu, prior_sigma);
        let proposal_prior = normal_pdf(proposal, prior_mu, prior_sigma);

        // 事後分布の計算
        let current_posterior = current_likelihood * current_prior;
        let proposal_posterior = proposal_likelihood * proposal_prior;

        // 受け入れ率の計算
        let acceptance_ratio = (proposal_posterior / current_posterior).min(1.0);

        // 受け入れ判定
        if rng.gen::<f64>() < acceptance_ratio {
            current = proposal;
        }

        // サンプルを記録
        samples.push(current);
    }

    // バーンインを適用して初期サンプルを破棄
    if samples.len() > burn_in {
        samples.split_off(burn_in) // バーンイン以降のサンプルを残す
    } else {
        samples // バーンインがサンプル数を超える場合は全サンプル
    }
}

#[allow(dead_code)]
// 事後分布の計算に使うメトロポリス・ヘイスティングスアルゴリズム
pub fn metropolis_hastings_v4(
    iterations: usize,
    burn_in: usize,
    init: f64,
    data: f64,
    sigma: f64,
    prior_mu: f64,
    prior_sigma: f64,
    proposal_scale: f64
) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let mut current = init;

    let mut acceptance_ratios = Vec::new();
    let mut proposal_likelihoods = Vec::new();
    let mut proposal_priors = Vec::new();
    let mut prior_mus = Vec::new();
    let mut prior_sigmas = Vec::new();

    // 提案分布としてrand_distrのNormalを使用（スケール調整）
    let proposal_distribution = Normal::new(0.0, proposal_scale).unwrap();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + proposal_distribution.sample(&mut rng);

        // 尤度の計算
        let current_likelihood  = normal_pdf(data, current, sigma);
        let proposal_likelihood = normal_pdf(data, proposal, sigma);

        // 事前分布の計算
        let current_prior  = normal_pdf(current, prior_mu, prior_sigma);
        let proposal_prior = normal_pdf(proposal, prior_mu, prior_sigma);
        prior_mus.push(prior_mu);
        prior_sigmas.push(prior_sigma);

        // 事後分布の計算
        let current_posterior = current_likelihood * current_prior;
        let proposal_posterior = proposal_likelihood * proposal_prior;

        // 受け入れ率の計算
        let acceptance_ratio = (proposal_posterior / current_posterior).min(1.0);

        // 受け入れ判定
        if rng.gen::<f64>() < acceptance_ratio {
            current = proposal;
        }

        // サンプルを記録
        samples.push(current);
        acceptance_ratios.push(acceptance_ratio);
        proposal_likelihoods.push(proposal_likelihood);
        proposal_priors.push(proposal_prior);
    }
    let ac_mean  = mean_normal_dist(&acceptance_ratios);
    let ac_sigma = stddev_normal_dist(&acceptance_ratios);
    let ac_size = acceptance_ratios.len();
    println!("acceptance_ratio    : {:.4} +- {:.4} size = {}", ac_mean, ac_sigma, ac_size);
    println!("proposal_likelihood : {:.4} +- {:.4}", mean_normal_dist(&proposal_likelihoods), stddev_normal_dist(&proposal_likelihoods));
    println!("proposal_prior      : {:.4} +- {:.4}", mean_normal_dist(&proposal_priors), stddev_normal_dist(&proposal_priors));
    println!("prior_mu            : {:.4} +- {:.4}", mean_normal_dist(&prior_mus), stddev_normal_dist(&prior_mus));
    println!("prior_sigma         : {:.4} +- {:.4}", mean_normal_dist(&prior_sigmas), stddev_normal_dist(&prior_sigmas));

    // バーンインを適用して初期サンプルを破棄
    if samples.len() > burn_in {
        samples.split_off(burn_in) // バーンイン以降のサンプルを残す
    } else {
        samples // バーンインがサンプル数を超える場合は全サンプル
    }


}


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

// 尤度の計算
// ある時点からパラメータが変化する正規分布
pub fn likelihood(y: &[f64], tc: usize, mu1: f64, mu2: f64, sigma: f64) -> f64 {
    let mut ll = 0.0;
    for (i, &val) in y.iter().enumerate() {
        let mean = if i < tc { mu1 }  else { mu2 };
        ll += -0.5 * ((val - mean).powi(2) / sigma.powi(2)) ;
    }
    ll.exp()
}

// 同時事後分布の計算
pub fn joint_posterior(y: &[f64], tc : usize, mu1: f64, mu2: f64, sigma: f64) -> f64 {
    let prior_mu1 = normal_pdf(mu1, 50.0, 10.0);
    let prior_mu2 = normal_pdf(mu2, 50.0, 10.0);

    let prior_tc  = 1.0 / (y.len() as f64);
    let likelihood_val = likelihood(y, tc, mu1, mu2, sigma);

    likelihood_val * prior_mu1 * prior_mu2 * prior_tc
}

pub fn metropolis_hastings_changepoint(y: &[f64], iterations: usize) -> (usize, f64, f64) {
    let mut rng = rand::thread_rng();
    let n       = y.len();

    // 初期パラメータ
    let mut tc  = n / 2;
    let mut mu1 = 50.0;
    let mut mu2 = 60.0;
    let sigma   = 100.0;

    // 変化点tcの更新
    for _ in 0..iterations {
        let tc_new = rng.gen_range(1..n);
        let acceptance_ratio_tc = joint_posterior(y, tc_new, mu1, mu2, sigma)
                                / joint_posterior(y, tc, mu1, mu2, sigma);
        if rng.gen::<f64>() < acceptance_ratio_tc  {
            tc = tc_new;
        }

        let mu1_candidate = Normal::new(mu1, 100.0).unwrap().sample(&mut rng);
        let acceptance_ratio_mu1 = joint_posterior(y, tc, mu1_candidate, mu2, sigma)
                                 / joint_posterior(y, tc, mu1,           mu2, sigma);
        if rng.gen::<f64>() < acceptance_ratio_mu1  {
            mu1 = mu1_candidate;
        }

        let mu2_candidate = Normal::new(mu2, 100.0).unwrap().sample(&mut rng);
        let acceptance_ratio_mu2 = joint_posterior(y, tc, mu1, mu2_candidate, sigma)
                                 / joint_posterior(y, tc, mu1, mu2,           sigma);
        if rng.gen::<f64>() < acceptance_ratio_mu2  {
            mu2 = mu2_candidate;
        }
    }

    (tc, mu1, mu2)
}
