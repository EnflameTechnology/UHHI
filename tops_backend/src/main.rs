use driv::topsDevice_t;
use tops_raw as driv;

pub struct TopsDevice(topsDevice_t);
impl TopsDevice {
    pub fn new(d : topsDevice_t) -> Self
    {
        TopsDevice {0 : d}
    }
}
fn main() {
    println!("Test");
    unsafe {
        let mut device = TopsDevice {0:0};
        driv::topsDeviceGet(&mut device.0, 0 as i32);
        print!("{}", device.0 as i32);
    }

}