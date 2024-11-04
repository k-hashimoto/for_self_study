//use modules::utils::*;
use crate::modules::utils::*;
use crate::modules::mcmc_tools::*;
use crate::modules::mcmc::*;

// stanで処理するときのような処理ではなく、逐次的にデータが入ってくるオンライン処理的な実装
pub fn run() {
    println!("[サンプル] ベイズ推定で顧客の平均購入金額を推定する(逐次処理ver.)");

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
    let thinning_interval = 300; // 薄化の間隔（例：10サンプルに1つを選択） 1000
    let burn_in = 50000; // バーンイン

    // 初期値を異なる3つのチェーンで設定
    let init_values = vec![30.0, 50.0, 70.0]; // 初期値の異なる設定

    println!("自己相関抑制のためのパラメータ");
    println!("  薄化インターバル : {}", thinning_interval);
    println!("  バーンイン : {}", burn_in);
    // データ点のループ
    // 1つのデータ点に対して3チェーンのサンプルを生成してしまっている
    for (i, &data) in observations.iter().enumerate() {
        println!("-------------------------------");
        println!("# Processing observation {} / {}", i + 1, observations.len());

        // 各初期値で独立したチェーンを実行
        let mut all_samples = Vec::new();
        for &init in init_values.iter() {
            // メトロポリス・ヘイスティングスの実行
            let samples = metropolis_hastings_online(iterations, burn_in, init, data, sigma, prior_mu, prior_sigma);

            // サンプルの薄化（間引き）
            let thinned_samples = thin_samples(&samples, thinning_interval);
            all_samples.extend(thinned_samples); // 結果を統合
        }

        //all_samplesには全部のチェーンのサンプルが含まれる

        // 事後分布の平均を計算
        posterior_mean = mean_normal_dist(&all_samples);

        //
        // 次の更新に備えて事前分布の平均を事後分布の平均に更新
        prior_mu = posterior_mean;
        // 次の更新に備えて事前分布の標準偏差をサンプルの標準偏差に更新
        prior_sigma = stddev_normal_dist(&all_samples);

        println!(
            "Updated posterior mean after observing {}: {:.2} +- {:.4}",
            data, posterior_mean, prior_sigma
        );
        // 自己相関の計算と表示（遅延1～5まで）
        for lag in 1..=5 {
            let ac = autocorrelation(&all_samples, lag);
            println!("   Autocorrelation at lag {}: {:.4}", lag, ac);
        }

        let if_value = inefficiency_factor_diagnostic(&all_samples, 5);
        // Geweke診断
        println!("   Geweke p-value = {:.4}", geweke_diagnostic_p_value(&all_samples, 0.2, 0.5));
        // IF
        println!("   IF-value       = {:.2}({}個/{}の独立したサンプル)", if_value, iterations / if_value as usize, iterations);
        // 95%信用区間の計算
        // 例：平均日次売上金額が95%の確率で約60.12ドルから82.45ドルの間にある
        let (lower_bound, upper_bound) = credible_interval(&all_samples);
        println!("   95% Credible Interval: ({:.2}, {:.2})", lower_bound, upper_bound);
    }
    println!("-------------------------------");
    println!("観測値(顧客の購入金額-ドル) : {:?}", observations);
    println!("観測値の平均: {:.2}", mean_normal_dist(&observations));
    println!("ベイズ推定された事後分布の平均値: {:.2}", posterior_mean);
    println!("事後分布の平均値 - 単純平均: {:.2}", posterior_mean-mean_normal_dist(&observations));
}