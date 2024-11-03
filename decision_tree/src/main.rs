use std::any::type_name;

fn print_type_of<T>(_: &T) {
    println!("Type: {}", type_name::<T>());
}

#[derive(Debug, Clone)]
struct Sample {
    label: usize,
    features: Vec<f64>
}
impl Sample {
    fn contain_value(&self, threshold: f64, split_axis: usize) -> bool {
        // featuresにthreshold以上の値がふくまれているか？
        if self.features[split_axis] > threshold {
            true
        } else {
            false
        }
    }
    fn check_label(&self, value: usize) -> bool {
        if self.label == value {
            true
        } else {
            false
        }
    }
}

fn split_samples(samples: &Vec<Sample>, threshold: f64, split_axis: usize) ->  (Vec<Sample>, Vec<Sample>) {
    (
        samples.iter().filter( | sample | {
            sample.contain_value(threshold, split_axis)
        }).cloned().collect(),

        samples.iter().filter( | sample | {
            !sample.contain_value(threshold, split_axis)
        }).cloned().collect()
    )
}

fn filter_by_label(samples: &Vec<Sample>, label: usize) -> Vec<Sample>{
    samples.iter().filter( | sample | {
        sample.check_label(label)
    }).cloned().collect()
}

fn impurity(all_samples: &Vec<Sample>, split_value: f64, split_axis: usize) -> f64
{
    // 不純度を計算する

    // 1. しきい値で分割する
    let (split_sample_a, split_sample_b) = split_samples(&all_samples, split_value, 0);

    // println!("split_sample_a = {:?}", split_sample_a);
    // println!("split_sample_b = {:?}", split_sample_b);

    // 2. ジニ係数計算のために、ラベルで分割する
    let split_sample_a_0 = filter_by_label(&split_sample_a, 0);
    let split_sample_a_1 = filter_by_label(&split_sample_a, 1);
    let split_sample_b_0 = filter_by_label(&split_sample_b, 0);
    let split_sample_b_1 = filter_by_label(&split_sample_b, 1);

    // println!("split_sample_a_1 = {:?}" , split_sample_a_1);
    // println!("split_sample_a_0 = {:?}" , split_sample_a_0);

    // 3. 分割したエリアごとにジニ係数を計算する
    let gini_a = gini(&split_sample_a_0, &split_sample_a_1);
    let gini_b = gini(&split_sample_b_0, &split_sample_b_1);

    // 4. データ点の数で重みを計算し、平均をとる
    let split_area_size_a = split_sample_a.len() as f64;
    let split_area_size_b = split_sample_b.len() as f64;
    let full_size = split_area_size_a + split_area_size_b;

    // 5. この分割での全データをもとにした平均のジニ係数
    (split_area_size_a / full_size) * gini_a + (split_area_size_b / full_size) * gini_b
}

fn gini(sample_a: &Vec<Sample>, sample_b: &Vec<Sample>) -> f64
{
    // ジニ係数の計算
    let n_sample_a    = sample_a.len() as f64;
    let n_sample_b    = sample_b.len() as f64;
    let n_full_sample = n_sample_a + n_sample_b;
    1. - (
        ((n_sample_a / n_full_sample).powi(2)) + ((n_sample_b / n_full_sample).powi(2))
    )
}

fn main()
{
    let all_samples =  vec![
        Sample { label: 0, features: vec![1.0, 40.0]},
        Sample { label: 0, features: vec![2.0, 30.0]},
        Sample { label: 0, features: vec![3.1, 10.0]},
        Sample { label: 1, features: vec![5.0, 30.0]},
        Sample { label: 1, features: vec![8.0, 20.0]},
        Sample { label: 1, features: vec![4.0, 60.0]}
    ];

    println!("# 簡単な決定木アルゴリズム");
    println!("{}", impurity(&all_samples, 3.0, 0));
}