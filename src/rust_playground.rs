trait Enemy {
    fn encount(&self);
    fn attack(&self);
}

//------------------------------------------------------------------------
//亡者
struct Undead {
    name: String,
    weapon : String,
}
impl Enemy for Undead {
    fn encount(&self) {
        println!("Encount with {}!", self.name);
    }
    fn attack(&self) {
        println!("[{}]: use {}", self.name, self.weapon);
    }
}
fn undead(name: &str, weapon: &str) -> Box<dyn Enemy> {
    Box::new(
        Undead{
            name: String::from(name),
            weapon: String::from(weapon)
        }
    )
}
//------------------------------------------------------------------------

//------------------------------------------------------------------------
//デーモン
struct Demon {
    name: String,
    weapon : String,
    demon_type: String,
}
impl Enemy for Demon {
    fn encount(&self) {
        println!("Encount with {}({})!", self.name, self.demon_type);
    }
    fn attack(&self) {
        println!("[{}]: use {}", self.name, self.weapon);
    }
}
fn demon(name: &str, weapon: &str, demon_type: &str) -> Box<dyn Enemy> {
    Box::new(
        Demon{
            name: String::from(name),
            weapon: String::from(weapon),
            demon_type: String::from(demon_type),
        }
    )
}
//------------------------------------------------------------------------

fn random_ememy(random_number: f64) -> Box<dyn Enemy> {
    if random_number < 0.5 {
        undead("undead1", "Short Sword")
    } else {
        demon("Demon", "Demon Great Machete", "Asylum")
    }
}

fn main() {
    let ememy = random_ememy(0.51);

    ememy.encount();
    ememy.attack();
}
