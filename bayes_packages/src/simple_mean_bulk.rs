use crate::modules::mcmc::*;
use crate::modules::mcmc_tools::*;
use crate::modules::mcmc_visualizer::*;
use crate::modules::utils::*;

use rand_distr::{Distribution, Uniform};

// simple_mean_onlineのように逐次的にデータが入ってくるオンライン処理的な実装ではなく、stanで処理するときのようにバルクで処理する実装
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("[サンプル] ベイズ推定で顧客の平均購入金額を推定する(一括処理ver.)");

    // 目的：顧客の平均購入金額を推定する
    // ｎ回の観測データ（顧客の購入金額-ドルスプシで乱数から生成した値
    // =ARRAYFORMULA(NORMINV(RANDARRAY(20),D$11,D$12))
    let observations = vec![
        41.0, 47.0, 57.0, 66.0, 49.0, 28.0, 47.0, 58.0, 54.0, 51.0, 51.0, 45.0, 44.0, 56.0, 61.0,
        60.0, 51.0, 48.0, 57.0, 48.0, 65.0, 46.0, 51.0, 43.0, 43.0, 54.0, 40.0, 53.0, 75.0, 50.0,
        35.0, 43.0, 53.0, 68.0, 48.0, 46.0, 44.0, 34.0, 48.0, 60.0,
    ];

    let true_mean = mean_normal_dist(&observations);
    // 事前分布：平均1ドル、標準偏差100ドルの正規分布
    let prior_mu = 50.0;
    let prior_sigma = 10.0;

    // ベイズ更新のイテレーション数
    let iterations = 100000; // メトロポリス・ヘイスティングスのサンプル数 100000
    let sigma = 10.0; // 尤度の標準偏差（観測誤差として仮定）

    // let mut posterior_mean = 0.;
    // MCMCの自己相関を排除するパラメータ
    let thinning_interval = 10; // 薄化の間隔（例：10サンプルに1つを選択） 1000
    let burn_in = 50000; // バーンイン
    let proposal_scale = 5.0; // 提案分布のスケール

    // 初期値を異なる3つのチェーンで設定
    // 一様分布で初期値を生成
    let mut rng = rand::thread_rng();
    let uniform_dist = Uniform::new(true_mean - 10.0, true_mean + 10.0); // 初期値の範囲を20.0～80.0と設定
    let init_values: Vec<f64> = (0..4).map(|_| uniform_dist.sample(&mut rng)).collect();
    // let init_values = vec![30.0, 50.0, 70.0]; // 初期値の異なる設定

    println!("-------------------------------");
    println!("# 自己相関抑制のためのパラメータ");
    println!("  薄化インターバル : {}", thinning_interval);
    println!("  バーンイン : {}", burn_in);

    // 各初期値で独立したチェーンを実行
    let mut chains = Vec::new();
    // チェーンのループ

    for &init in init_values.iter() {
        let parameters = vec![init, sigma];
        let prior_parameters = vec![prior_mu, prior_sigma];
        // メトロポリス・ヘイスティングスの実行
        // MCMCの初期値を1つにすると、生成した数値の分散が早期にゼロになってしまう
        let samples = metropolis_hastings_bulk(
            "normal",
            iterations,
            burn_in,
            &observations,
            parameters,
            prior_parameters,
            proposal_scale,
        );
        // サンプルの薄化（間引き）
        //let thinned_samples = thin_samples(&samples, thinning_interval);
        chains.push(thin_samples(&samples, thinning_interval)); // 結果を統合
    }

    // 平均や各種統計量の計算用
    let all_samples: Vec<f64> = chains
        .iter()
        .flat_map(|chain| chain.iter().cloned())
        .collect();
    // 事後分布の平均を計算
    let posterior_mean = mean_normal_dist(&all_samples);
    println!("-------------------------------");
    println!("# MCMCのチェック");
    // 自己相関の計算と表示（遅延1～5まで）
    for lag in 1..=5 {
        let ac = autocorrelation(&all_samples, lag);
        println!("   Autocorrelation at lag {}: {:.4}", lag, ac);
    }

    let if_value = inefficiency_factor_diagnostic(&all_samples, 5);
    // Geweke診断
    println!(
        "   Geweke p-value = {:.4}",
        geweke_diagnostic_p_value(&all_samples, 0.2, 0.5)
    );
    // IF
    println!(
        "   IF-value       = {:.2}({}個/{}の独立したサンプル)",
        if_value,
        iterations / if_value as usize,
        iterations
    );
    // 95%信用区間の計算
    // 例：平均日次売上金額が95%の確率で約60.12ドルから82.45ドルの間にある
    let (lower_bound, upper_bound) = credible_interval(&all_samples);

    println!("-------------------------------");
    // 推定結果
    let _ = print_mcmc_summary_table(&true_mean, &posterior_mean, &lower_bound, &upper_bound);

    // トレースプロットを生成してPNGに保存
    let _ = trace_plot("./plots/bayes_packages/trace_plot.png", &chains, 0, (iterations - burn_in) / 10, 35.0, 65.0);

    plot_mu1_kde_and_data_mean(
        &chains[3],
        &observations,
        "./plots/bayes_packages/distribution.png",
        1.0,
    )
}
