use std::error::Error;

mod modules;
use crate::modules::utils::read_csv_to_two_vectors;
use crate::modules::metropolis_hastings::{normal_pdf, likelihood};

use rand::Rng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::ContinuousCDF;
use statrs::distribution::{Continuous, Normal as StatrsNormal};
use std::convert::TryInto;

// 同時事後分布の計算
fn joint_posterior(y: &[f64], tc : usize, mu1: f64, mu2: f64, sigma: f64) -> f64 {
    let prior_mu1 = normal_pdf(mu1, 50.0, 10.0);
    let prior_mu2 = normal_pdf(mu2, 50.0, 10.0);

    let prior_tc  = 1.0 / (y.len() as f64);
    let likelihood_val = likelihood(y, tc, mu1, mu2, sigma);

    likelihood_val * prior_mu1 * prior_mu2 * prior_tc
}

fn metropolis_hastings_changepoint(y: &[f64], iterations: usize, burn_in: usize, thinning_interval: usize) -> (usize, f64, f64) {
    let mut rng = rand::thread_rng();
    let n       = y.len();

    // 初期パラメータ
    let mut tc  = n / 2;
    let mut mu1 = 50.0;
    let mut mu2 = 60.0;
    let sigma   = 100.0;

    let mut tc_samples = Vec::new();
    let mut m1_samples = Vec::new();
    let mut m2_samples = Vec::new();

    // 変化点tcの更新
    for i in 0..iterations {
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

        if i >= burn_in {
            tc_samples.push(tc);
            m1_samples.push(mu1);
            m2_samples.push(mu2);
        }

    }

    let thined_tc_samples: Vec<_> = tc_samples.iter().step_by(thinning_interval).copied().collect();
    let thined_m1_samples: Vec<_> = m1_samples.iter().step_by(thinning_interval).copied().collect();
    let thined_m2_samples: Vec<_> = m2_samples.iter().step_by(thinning_interval).copied().collect();

    let estimated_tc = thined_tc_samples.iter().sum::<usize>() / thined_tc_samples.len();
    let estimated_m1 = thined_m1_samples.iter().sum::<f64>() / thined_m1_samples.len() as f64;
    let estimated_m2 = thined_m2_samples.iter().sum::<f64>() / thined_m2_samples.len() as f64;

    (estimated_tc, estimated_m1, estimated_m2)
}

fn vec_to_array<T>(vec: Vec<T>) -> Result<Box<[T]>, Vec<T>> {
    let len = vec.len();

    // Vec<T>の長さに基づいて配列へ変換を試みる
    match len {
        4 => Ok(vec.into_boxed_slice().try_into().unwrap()),
        _ => Err(vec),
    }
}

fn main() -> Result<(), Box<dyn Error>>  {
    println!("# 英国炭鉱事故をベイズ変化点検知で分析する例");
    let path = "data/uk_coal_mining_accidents.csv";
    let (ts, numbers) = read_csv_to_two_vectors(path)?;

    match vec_to_array(numbers) {
        Ok(anumbers) => println!("Array: {:?}", anumbers),
        Err(vec) => println!("Failed to convert")
    }

    let iterations = 100000;
    let burn_in    = 50000;
    let thinning_interval = 10;

    let (estimated_tc, estimated_mu1, estimated_mu2) = metropolis_hastings_changepoint(&anumbers, iterations, burn_in, thinning_interval);

    println!("推定された変化点: {}(20)", estimated_tc);
    println!("推定された平均: mu1 = {:.2}, mu2 = {:.2}", estimated_mu1, estimated_mu2);
    Ok(())
}