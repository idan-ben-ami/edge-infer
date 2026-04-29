#![no_std]
#![no_main]

use cortex_m_rt::{entry, pre_init};
use cortex_m_semihosting::{
    debug::{self, EXIT_SUCCESS},
    hprintln,
};
use panic_semihosting as _;

use generated::predict;

static INPUT: [[[f32; 28]; 28]; 1] = [[[0.0; 28]; 28]; 1];

const PAINT: u32 = 0xDEAD_BEEF;
// Don't paint our own pre_init / main frames. 256 bytes is enough
// headroom for pre_init's tiny frame and the path back up to main.
// Numbers reported below this offset will read as exactly 256
// (we can't see into the unpainted region); predict() functions
// smaller than 256 bytes of stack will report 256 as a lower
// bound. For the bundled examples that's only iris and
// vibration_anomaly (both pure MLPs).
const SAFETY_TOP_OFFSET: usize = 256;

extern "C" {
    static mut __ebss: u32;
    static mut _stack_start: u32;
}

#[pre_init]
unsafe fn paint_stack() {
    let bottom = &raw mut __ebss as *mut u32;
    let top = &raw mut _stack_start as *mut u32;
    let safe_top = (top as usize - SAFETY_TOP_OFFSET) as *mut u32;

    let mut p = bottom;
    while (p as usize) < (safe_top as usize) {
        p.write_volatile(PAINT);
        p = p.add(1);
    }
}

#[entry]
fn main() -> ! {
    // Run inference. This pushes whatever frames `predict` needs
    // onto the stack, overwriting our paint pattern as it goes.
    let output = predict(&INPUT);

    // Anchor output (don't let optimizer elide predict).
    let predicted = output
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // Find the first cell that is NOT still 0xDEADBEEF, walking
    // from the bottom of the stack region upward. That cell is
    // the lowest address the SP reached during predict.
    let bottom = &raw const __ebss as *const u32;
    let top = &raw const _stack_start as *const u32;

    let mut p = bottom;
    let mut high_water: *const u32 = top;
    while (p as usize) < (top as usize) {
        let v = unsafe { p.read_volatile() };
        if v != PAINT {
            high_water = p;
            break;
        }
        p = unsafe { p.add(1) };
    }

    let stack_top = top as usize;
    let stack_low = high_water as usize;
    let used = stack_top.saturating_sub(stack_low);

    hprintln!("Stack high-water mark: {} bytes", used);
    hprintln!("(measured by stack-painting; excludes 1 KB safety margin at top)");
    hprintln!("predicted class: {}", predicted);

    debug::exit(EXIT_SUCCESS);
    loop {}
}
