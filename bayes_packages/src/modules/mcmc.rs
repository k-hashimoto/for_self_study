use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::modules::mcmc_distributions::*;

// ---------------------------------------------------------------------
pub fn metropolis_hastings_online(
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
    // todo : 引数処理の部分が正規分布と分離できていない
    let mut current_distribution = get_distribution("normal", &vec![current, sigma]);
    let prior_distribution = get_distribution("normal", &vec![prior_mu, prior_sigma]);

    // 提案分布としてrand_distrのNormalを使用
    let proposal_distribution = Normal::new(0.0, proposal_scale).unwrap();

    let mut acceptance_ratios = Vec::new();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + proposal_distribution.sample(&mut rng);

        // 尤度計算
        // todo : 引数処理の部分が正規分布と分離できていない
        let current_likelihood = current_distribution.pdf(data);
        current_distribution.update_parameters(&vec![proposal, sigma]);
        let proposal_likelihood = current_distribution.pdf(data);

        // 事前分布の計算
        let current_prior = prior_distribution.pdf(current);
        let proposal_prior = prior_distribution.pdf(proposal);

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
    //    println!("MCMC acceptance_ratio : {:.4} +- {:.4}", mean_normal_dist(&acceptance_ratios), stddev_normal_dist(&acceptance_ratios));
    //バーンイン
    if samples.len() > burn_in {
        samples.split_off(burn_in)
    } else {
        samples
    }
}

// ---------------------------------------------------------------------
pub fn metropolis_hastings_bulk(
    distribution_name: &str,
    iterations: usize,
    burn_in: usize,
    data: &[f64],
    parameters: Vec<f64>,
    prior_parameters: Vec<f64>,
    proposal_scale: f64,
) -> Vec<f64> {
    // [todo]
    //  - 事前・事後分布の実装はget_distributionで隠蔽できたので、stringの引数で分布を指定出来るようにする
    //  - この関数の引数は正規分布前提なのでこれも隠蔽する

    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let init = parameters[0];
    let mut current = init;

    // 分布を取得
    let current_distribution = get_distribution(distribution_name, &parameters);
    let mut proposal_distribution = get_distribution(distribution_name, &prior_parameters);

    // 提案分布としてrand_distrのNormalを使用
    let distribution = Normal::new(0.0, proposal_scale).unwrap();

    let mut acceptance_ratios = Vec::new();

    for _ in 0..iterations {
        // 提案された新しいサンプルを生成
        let proposal = current + distribution.sample(&mut rng);
        let proposal = if distribution_name == "poisson" {
            proposal.abs()
        } else {
            proposal
        };

        // 尤度計算
        // todo : 引数処理の部分が正規分布と分離できていない。むずい。
        // 分布によってパラメータの数が違う。正規分布なら2つ、ポアソン分布なら1つ
        // このMHの実装は共役分布が前提で、共役分布は限られていれていてif分岐出来るのでこれで良しよする
        if distribution_name == "normal" {
            let sigma = parameters[1];
            proposal_distribution.update_parameters(&vec![proposal, sigma]);
        } else if distribution_name == "poisson" {
            proposal_distribution.update_parameters(&vec![proposal]);
        }

        let current_likelihood: f64 = data.iter().map(|&x| current_distribution.pdf(x).ln()).sum();
        let proposal_likelihood: f64 = data
            .iter()
            .map(|&x| proposal_distribution.pdf(x).ln())
            .sum();

        // 事前分布の計算(対数空間)
        let current_prior = current_distribution.pdf(current).ln();
        let proposal_prior = proposal_distribution.pdf(proposal).ln();

        // 事後分布の計算(対数空間)
        let current_posterior = current_likelihood + current_prior;
        let proposal_posterior = proposal_likelihood + proposal_prior;

        let acceptance_ratio = (proposal_posterior - current_posterior).exp().min(1.0);

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
