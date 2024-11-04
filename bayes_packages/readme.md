bays_studyで仮実装したやつをこっちに移植して正式版的にあつかう

## Build
```
cargo build -p bayes_packages
```

## Build & Run
```
ccargo run -p bayes_packages -- --<PACKAGE>
```

PACKAGEはsrc直下に `simple_mean_bulk.rs` のような形で実装し `main.rs` で読み出す