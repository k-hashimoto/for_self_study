#[allow(dead_code)]
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

// use csv::ReaderBuilder;
// use std::error::Error;
// use std::fs::File;
use std::path::Path;

pub fn mean_normal_dist(vector: &Vec<f64>) -> f64 {
    vector.iter().sum::<f64>() / vector.len() as f64
}

#[allow(dead_code)]
pub fn stddev_normal_dist(vector: &Vec<f64>) -> f64 {
    let mean = mean_normal_dist(vector);
    vector
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        .sqrt()
        / vector.len() as f64
}

#[allow(dead_code)]
pub fn hadamard_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> Vec<f64> {
    // アダマール積
    vec1.iter().zip(vec2.iter()).map(|(&a, &b)| a * b).collect()
}

#[allow(dead_code)]
pub fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(&a, &b)| a * b).sum()
}

#[allow(dead_code)]
pub fn subtract_constant(vec: &mut Vec<f64>, constant: f64) {
    for value in vec.iter_mut() {
        *value -= constant;
    }
}

#[allow(dead_code)]
pub fn read_csv_to_two_vectors(path: &str) -> Result<(Vec<String>, Vec<f32>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut column1 = Vec::new();
    let mut column2 = Vec::new();

    for result in rdr.records() {
        let record = result?;
        if record.len() >= 2 {
            column1.push(record[0].to_string());
            column2.push(record[1].parse::<f32>().unwrap_or(0.0));
        }
    }

    Ok(( column1, column2 ))
    //    Ok(( Vec::from(column1), Vec::from(column2) ))
}

// ----------------------------------------------------------------------------------------------
pub fn read_csv_to_float_vectors(file_path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(Path::new(file_path))?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut col1 = Vec::new();
    let mut col2 = Vec::new();

    for result in rdr.records() {
        let record = result?;

        // 各行が2カラムであることを仮定して、それぞれをf64に変換
        if record.len() == 2 {
            let first_column: f64 = record[0]
                .parse()
                .map_err(|_| "Failed to parse first column as f64")?;
            let second_column: f64 = record[1]
                .parse()
                .map_err(|_| "Failed to parse second column as f64")?;

            col1.push(first_column);
            col2.push(second_column);
        } else {
            return Err("CSV file format error: Each row must have exactly 2 columns.".into());
        }
    }

    Ok((col1, col2))
}