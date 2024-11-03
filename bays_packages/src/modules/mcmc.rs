
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::modules::mcmc_distributions::*;

// ---------------------------------------------------------------------
pub fn metropolis_hastings(
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

    // todo : 引数処理の部分が正規分布と分離できていない
    let mut proposal_distribution = get_distribution("normal", &vec![prior_mu, prior_sigma]);

    // 提案分布としてrand_distrのNormalを使用
    let distribution = Normal::new(0.0, proposal_scale).unwrap();

    let mut acceptance_ratios = Vec::new();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + distribution.sample(&mut rng);

        // 尤度計算
         // todo : 引数処理の部分が正規分布と分離できていない
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