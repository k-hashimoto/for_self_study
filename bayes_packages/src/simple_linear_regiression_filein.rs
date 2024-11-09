// use plotters::prelude::*;
use rand::Rng;
use rand_distr::Normal;
use statrs::distribution::{Continuous, Normal as StatrsNormal};
use crate::modules::utils::*;
use crate::modules::mcmc_tools::*;
use crate::modules::mcmc_visualizer::*;
// use std::error::Error;

fn split_vec(vec: &Vec<(f64, f64, f64)>, thinning_interval: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut z: Vec<f64> = Vec::new();

    for &(a, b, c) in vec {
        x.push(a);
        y.push(b);
        z.push(c);
    }
    (thin_samples(&x, thinning_interval), thin_samples(&y, thinning_interval) , thin_samples(&z, thinning_interval))
}

// ----------------------------------------------------------------------------------------------
struct BayesianLinearRegression {
    alpha_prior: StatrsNormal,
    beta_prior: StatrsNormal,
    sigma_prior: StatrsNormal,
}
impl BayesianLinearRegression {
    fn new(alpha_prior: StatrsNormal, beta_prior: StatrsNormal, sigma_prior: StatrsNormal) -> Self {
        BayesianLinearRegression {
            alpha_prior,
            beta_prior,
            sigma_prior,
        }
    }

    // fn likelihood(&self, x: &[f64], y: &[f64], alpha: f64, beta: f64, sigma: f64) -> f64 {
    //     // 尤度の計算。推定値とデータ点の差分が実現する確率を計算
    //     let mut likelihood = 1.0;
    //     for (&xi, &yi) in x.iter().zip(y.iter()) {
    //         let mean = alpha + beta * xi;
    //         let dist = StatrsNormal::new(mean, sigma).unwrap();
    //         likelihood *= dist.pdf(yi);
    //     }
    //     likelihood
    // }
    fn likelihood(&self, x: &[f64], y: &[f64], alpha: f64, beta: f64, sigma: f64) -> f64 {
        // 対数で計算しないと、likelihood *= dist.pdf(yi)といったふうにゼロに近い値をデータ点の数だけかけるので限りなくゼロに近い値にになってしまう
        let n = x.len() as f64;
        let mut log_likelihood = -n * (2.0 * std::f64::consts::PI).ln() / 2.0 - n * (sigma.powi(2)).ln() / 2.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let mean = alpha + beta * xi;
            log_likelihood -= (yi - mean).powi(2) / (2.0 * sigma.powi(2));
        }

        log_likelihood
    }


    // fn posterior(&self, x: &[f64], y: &[f64], alpha: f64, beta: f64, sigma: f64) -> f64 {
    //     // ベイズ更新
    //     self.likelihood(x, y, alpha, beta, sigma)
    //         * self.alpha_prior.pdf(alpha)
    //         * self.beta_prior.pdf(beta)
    //         * self.sigma_prior.pdf(sigma)
    // }

    fn posterior(&self, x: &[f64], y: &[f64], alpha: f64, beta: f64, sigma: f64) -> f64 {
        // 尤度の対数を計算
        let log_likelihood = self.likelihood(x, y, alpha, beta, sigma);

        // 事前分布の対数を計算
        let log_prior_alpha = self.alpha_prior.ln_pdf(alpha);
        let log_prior_beta = self.beta_prior.ln_pdf(beta);
        let log_prior_sigma = self.sigma_prior.ln_pdf(sigma);

        // 対数事後分布を返す
        log_likelihood + log_prior_alpha + log_prior_beta + log_prior_sigma
    }


    fn metropolis_hastings(&self, x: &[f64], y: &[f64], iterations: usize, burn_in: usize, proposal_scale: f64) -> Vec<(f64, f64, f64)> {
        // [メトロポリス・ヘイスティングスでベイズ線形回帰]
        // 尤度：alpha + beta * xiで計算される推定値とyiの差分が実現する確率として計算
        // その尤度を使ってベイズ更新する。ベイズ更新した事後分布と更新前のそれの比をつかって受け入れ判定をする

        let print_acceptance = true; // 最初の10行を出力するか

        let mut rng = rand::thread_rng();
        // 線形回帰モデルの各種パラメータの初期値
        // let mut alpha: f64 = rng.gen_range(-1.0..1.0);
        // let mut beta: f64 = rng.gen_range(-1.0..1.0);
//        let mut sigma: f64 = rng.gen_range(0.1..2.0);

        let mut alpha = y.iter().sum::<f64>() / y.len() as f64; // yの平均
        let mut beta = (y[y.len() - 1] - y[0]) / (x[x.len() - 1] - x[0]); // xとyの端点を使った傾きの推定
        let mut sigma = 300.0; // 初期値を固定値で設定

        let mena_init = 0.0;
        print!("initial values : {:.3} {:.3} {:.3}\n", alpha, beta,sigma );
        let mut num_accepted = 0; // MH法によってどのくらいのサンプルが採用されたかを確かめる為にカウントする
        // 確認用
        let mut acceptances:Vec<f64> = Vec::new();
        let mut accepte_thresholds:Vec<f64> = Vec::new();

        let mut samples = vec![];
        for i in 0..iterations {
            // 各種パラメータをランダムに"少しだけ"動かす
            let new_alpha = alpha + rng.sample(Normal::new(0.0, proposal_scale).unwrap());
            let new_beta = beta + rng.sample(Normal::new(0.0, proposal_scale).unwrap());
            let new_sigma = (sigma + rng.sample(Normal::new(0.0, proposal_scale).unwrap())).abs();


            // 尤度の計算とベイズ更新
            let current_posterior = self.posterior(x, y, alpha, beta, sigma);
            let proposed_posterior = self.posterior(x, y, new_alpha, new_beta, new_sigma);

            // 受け入れ判定
            let acceptance =  proposed_posterior - current_posterior; //対数なので割り算 => 引き算になる
            let accepte_threshold = rng.gen::<f64>().ln();
            if i < 10 && print_acceptance {
                print!("  acceptance = {:.5} ", acceptance);
            }

            if acceptance > accepte_threshold {
                alpha = new_alpha;
                beta = new_beta;
                sigma = new_sigma;

                if i < 10 && print_acceptance { print!(" +\n");}
                num_accepted += 1;
            } else {
                if i < 10 && print_acceptance { print!("\n");}
            }

            samples.push((alpha, beta, sigma));
            // 確認用
            acceptances.push(acceptance);
            accepte_thresholds.push(accepte_threshold);
        }
        print!("  number of accepted sample = {}({:.2}%)\n",num_accepted, 100. * (num_accepted as f64 / iterations as f64));
        let _ = write_vectors_to_csv("./mcmc.csv", &acceptances, &accepte_thresholds);
        if samples.len() > burn_in {
            samples.split_off(burn_in)
        } else {
            samples
        }

    }
}
// ----------------------------------------------------------------------------------------------
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (x, y) = match read_csv_to_float_vectors("./data/linear_regression/3-2-1-beer-sales-2.csv")
    {
        Ok((col1, col2)) => (col1, col2),
        Err(_err) => panic!("Somethig happen when reading csv."),
    };

    let iterations = 2000;
    let burn_in: usize = 1000;
    let thinning_interval = 1; // 薄化の間隔（例：10サンプルに1つを選択） 1000
    let proposal_scale: f64 = 0.5;

    let model = BayesianLinearRegression::new(
        StatrsNormal::new(0.0, 1.0).unwrap(),
        StatrsNormal::new(0.0, 1.0).unwrap(),
        StatrsNormal::new(0.0, 1.0).unwrap(),
    );

    // 4 チェーン生成する
    let mut samples: Vec<(f64, f64, f64)> = Vec::new(); // 平均値などの計算用
    let mut chains: Vec<Vec<f64>> = Vec::new(); // MCMCの収束判定用。どのchainかを特定したい
    for ichain in 0..4 {
        print!("processing #{}th chain...\n", ichain);
        let sample = model.metropolis_hastings(&x, &y, iterations, burn_in, proposal_scale);
        let (alpha_sample, beta_sample, sigma_sample) = split_vec(&sample, thinning_interval);
        chains.push(alpha_sample);

        samples.extend(sample);
    }
    let (alpha_samples, beta_samples, _sigma_samples) = split_vec(&samples, thinning_interval);
    print!("number of final sample is {}\n", alpha_samples.len());
    let alpha = mean_normal_dist(&alpha_samples);
    let beta = mean_normal_dist(&beta_samples);

    // let (alpha, beta, _sigma) = samples.last().unwrap();
    // プロットを作成
    plot_results_linear_regression("./plots/bayes_packages/bayesian_regression_plot.png", &x, &y, alpha, beta, &samples)?;


    // let (alpha_samples, beta_samples, _sigma_samples) = split_vec(&samples, thinning_interval);

    let _ = trace_plot(&chains, 0, (iterations - burn_in) / thinning_interval, 0.0, 20.0);

    println!("Plot saved as bayesian_regression_plot.png");
    Ok(())
}
