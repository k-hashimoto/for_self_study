#[allow(unused_imports)]
// use crate::modules::metropolis_hastings::{
//     normal_pdf, simple_metropolis_hastings, metropolis_hastings_normal_dist, thin_samples, autocorrelation, autocorrelation_v2, metropolis_hastings_normal_dist_v2, geweke_diagnostic_p_value, inefficiency_factor_diagnostic, credible_interval, metropolis_hastings_normal_dist_dev, metropolis_hastings_v4
// };
use crate::modules::metropolis_hastings::*;

#[allow(unused_imports)]
use crate::modules::utils::{
    dot_product, hadamard_product, mean_normal_dist, stddev_normal_dist, subtract_constant,
};

#[allow(dead_code)]
pub fn run_business_example() {
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
            let samples = metropolis_hastings_normal_dist(iterations, burn_in, init, data, sigma, prior_mu, prior_sigma);
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

#[allow(dead_code)]
pub fn run_simple_normal_dist() {
    let data  = 5.5;
    let sigma = 1.0;
    let init  = 5.0; // MH法の初期値
    let iterations = 100000;

    //事前分布
    let prior_mu    = 1.0;
    let prior_sigma = 10.0;
    let burn_in = 5000;
    println!("prior dist: mu = {}, sigma = {}", prior_mu, prior_sigma);
    //let samples = simple_metropolis_hastings(iterations, init, data, sigma);

    let samples = metropolis_hastings_normal_dist(
        iterations,
        burn_in,
        init,
        data,
        sigma,
        prior_mu,
        prior_sigma,
    );
    println!("First 10 samples: {:?}", &samples[..10]);

    let proposal_mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("Estimated posterior mean: {}", proposal_mean);
}

#[allow(dead_code)]
pub fn run_simple_changepoint() {
    // ベイズ界隈で有名な英国炭鉱事故数のベイズ分析と同じもの
    // ここではrun_business_example()の発展型として、事前事後分布は正規分布とし、メトロポリス・ヘイスティングス法を使う

    // 目的：顧客の平均購入金額を推定する。ただし購入金額のデータは途中からジャンプしており例えば値上げが実施されたとする
    // 前半と後半で別々の尤度を定義し、変化点と確率をパラメータとして同時事後分布を計算する
    // 観測データ（真ん中から+10変化）
    const DATA: [f64; 40] = [
        41.0, 47.0, 57.0, 66.0, 49.0, 28.0, 47.0, 58.0, 54.0, 51.0,
        51.0, 45.0, 44.0, 56.0, 61.0, 60.0, 51.0, 48.0, 57.0, 48.0,
        65.0 + 10.0, 46.0 + 10.0, 51.0 + 10.0, 43.0 + 10.0, 43.0 + 10.0,
        54.0 + 10.0, 40.0 + 10.0, 53.0 + 10.0, 75.0 + 10.0, 50.0 + 10.0,
        35.0 + 10.0, 43.0 + 10.0, 53.0 + 10.0, 68.0 + 10.0, 48.0 + 10.0,
        46.0 + 10.0, 44.0 + 10.0, 34.0 + 10.0, 48.0 + 10.0, 60.0 + 10.0
    ];

    let iterations = 10000;
    let (estimated_tc, estimated_mu1, estimated_mu2) = metropolis_hastings_changepoint(&DATA, iterations);

    // 正解
    let correct_mu1 = DATA[1..20].iter().sum::<f64>() /  DATA[1..20].len() as f64;
    let correct_mu2 = DATA[21..].iter().sum::<f64>()  /  DATA[21..].len() as f64;

    // 結果の表示
    println!("推定された変化点: {}(20)", estimated_tc);
    println!("推定された平均: mu1 = {:.2}({:.2}), mu2 = {:.2}({:.2})", estimated_mu1, correct_mu1, estimated_mu2, correct_mu2);
}