use std::any::type_name;

fn print_type_of<T>(_: &T) {
    println!("Type: {}", type_name::<T>());
}

#[derive(Debug)]
struct Sample {
    label: usize,
    features: Vec<f64>
}

fn split_x_sample(samples: &Vec<Sample>, threshold: f64, split_axis: usize) -> (Vec<Sample>, Vec<Sample>)
{
    // Sample構造体のvectorをSampleの中の値に応じて2つに分割して、2つのSample構造体のvectorを返す
    // 渡ってきた変数: Sample構造体のvector

    // 処理
    // vectorからsampleを取り出す
    // vectorの中のSampleの値を参照する。ここはread-onlyで良い
    // しきい値以上ならそのsampleを別のvectorに格納する

    (
        samples.iter().filter_map( | sample | {
            // vec![1.0, 3.5, 2.2, 4.0, 0.5] の中からthreshold以上の要素を取り出す
            let filtered_features: Vec<_> = sample.features.iter().cloned().filter(|&x| x >= threshold).collect();
            if !filtered_features.is_empty() {
                Some(
                    Sample {
                        label: sample.label,
                        features: filtered_features,
                    }
                )
            } else {
                None
            }
        }).collect(),

        samples.iter().filter_map( | sample | {
            // vec![1.0, 3.5, 2.2, 4.0, 0.5] の中からthreshold以上の要素を取り出す
            let filtered_features: Vec<_> = sample.features.iter().cloned().filter(|&x| x < threshold).collect();
            if !filtered_features.is_empty() {
                Some(
                    Sample {
                        label: sample.label,
                        features: filtered_features,
                    }
                )
            } else {
                None
            }
        }).collect()
    )
    // let mut return_sample_a = Vec::new();
    // let mut return_sample_b = Vec::new();
    // for v in vector {
    //     let _sample = v;
    //     if _sample.features[split_axis] > threshold {
    //         return_sample_a.push(_sample);
    //     } else {
    //         return_sample_b.push(_sample);
    //     }
    // }
    //    (return_sample_a, return_sample_b)
}

// fn filter_by_label(vector: &Vec<Sample>, label: usize) -> Vec<Sample>
// {
//     let mut return_sample = Vec::new();
//     for v in vector {
//         let _sample = v;
//         if _sample.label == label {
//             return_sample.push(_sample);
//         }
//     }
//     return_sample
// }

fn impurity(all_samples: &Vec<Sample>, split_axis: usize) -> f64
{
    // 不純度を計算する
    print_type_of(&all_samples);
    // 1. しきい値で分割する
    let (split_sample_a, split_sample_b) = split_x_sample(&all_samples, 3.0, split_axis);
    print_type_of(&split_sample_a);

    // // 2. ラベルで分割する
    // let split_sample_a_0 = filter_by_label(&split_sample_a, 0);
    // let split_sample_a_1 = filter_by_label(&split_sample_a, 1);

    println!("{:?}", split_sample_a);

    // gini(split_sample_a, );
    -1.
}

fn gini(sample_a: &Vec<f64>, sample_b: &Vec<f64>) -> f64
{
    // ジニ係数の計算
    let full_sample   = sample_a.clone();
    let n_full_sample = full_sample.len() as f64;
    let n_sample_a    = sample_a.len() as f64;
    let n_sample_b    = sample_b.len() as f64;

    1. - (
        ((n_sample_a / n_full_sample).powi(2)) + ((n_sample_b / n_full_sample).powi(2))
    )
}

fn main()
{
    let all_samples =  vec![
        Sample { label: 0, features: vec![1.0, 40.0]},
        Sample { label: 0, features: vec![2.0, 30.0]},
        Sample { label: 0, features: vec![3.0, 10.0]},
        Sample { label: 1, features: vec![5.0, 30.0]},
        Sample { label: 1, features: vec![8.0, 20.0]},
        Sample { label: 1, features: vec![4.0, 60.0]}
    ];

    println!("# 簡単な決定木アルゴリズム");
    let sample_a = vec![1., 2., 3., 5., 10., 2.];
    let sample_b = vec![40., 50., 60., 10., 30.,70.,];

    let gini_c = gini(&sample_a, &sample_b);
    println!("gini = {:.4}", gini_c);
    //    let (split_sample_a, split_sample_b) = split_x_sample(&all_samples, 3.0, 0);

    impurity(&all_samples, 0);


}