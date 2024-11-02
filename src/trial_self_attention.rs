use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

type Matrix = Array2<f32>;

fn generate_random_matrix(rows: usize, cols: usize) -> Matrix {
    Array2::random((rows, cols), Uniform::new(0.0, 1.0))
}

fn self_attention(query: &Matrix, key: &Matrix, value: &Matrix) -> Matrix {
    // スコアを計算する。Query x Key^T
    let scores = query.dot(&key.t());

    // スコアを正規化
    let mut attention_weights = scores.map(|&x| x.exp()); // softmax

    // sum_axis(Axis(1))で行方向に和をとる
    // 各行一つ一つがクエリに対応している
    let sum_weights = attention_weights.sum_axis(Axis(1)).insert_axis(Axis(1));
    attention_weights = attention_weights / &sum_weights;
    attention_weights.dot(value)
}

fn main() {
    println!("# self-attentionの簡単な例");
    let seq_lne = 4;   // 例：4単語のシーケンス
    let embed_dim = 3; // 例：各単語は3次元のベクトルで表現される

    let query = generate_random_matrix(seq_lne, embed_dim);
    let key   = generate_random_matrix(seq_lne, embed_dim);
    let value = generate_random_matrix(seq_lne, embed_dim);
    // Self-Attentionを計算
    let attention_output = self_attention(&query, &key, &value);

    // 結果を表示
    println!("Query:\n{:?}", query);
    println!("Key:\n{:?}", key);
    println!("Value:\n{:?}", value);
    println!("Attention Output:\n{:?}", attention_output);
}