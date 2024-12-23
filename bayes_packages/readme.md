bays_studyで仮実装したやつをこっちに移植して正式版的にあつかう

*自己学習用なので、ChatGPTが生成したコードが大半*

## Build
```
cargo build -p bayes_packages
```

## Build & Run
```
ccargo run -p bayes_packages -- --<PACKAGE>
```

PACKAGEはsrc直下に `simple_mean_bulk.rs` のような形で実装し `main.rs` で読み出す

## 実装済

| 説明  | 分布 | MCMC |リンク |
| ------------- | ------------- | -------------  | ------------- |
| 平均値の推定 | 正規分布 | MH法 | [code](src/simple_mean_bulk.rs) |
| 平均値の推定 | 正規分布 | MH法(逐次処理) | [code](src/simple_mean_online.rs) |
| 平均値の推定 | ポアソン分布 | MH法 | [code](src/simple_poisson.rs) |
| 線形回帰 | 正規分布 | MH法 | [code](src/simple_linear_regiression.rs) |
| 線形回帰(csvからデータ読み込み) | 正規分布 | MH法 | [code](src/simple_linear_regiression_filein.rs) |
