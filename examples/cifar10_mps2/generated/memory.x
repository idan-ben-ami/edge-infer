/* QEMU mps2-an386 (Cortex-M4F) -- 4 MB FLASH (SSRAM1) + 4 MB SRAM (SSRAM2).
 *
 * The cifar10_mps2 example was sized for this 4 MB-class memory map; with
 * the B.25 phase-scoping codegen the model also fits the 64 KB lm3s6965evb,
 * but the default kept here matches the example name so the QEMU recipe in
 * the README continues to work without further edits.
 *
 * Retargeting to real hardware:
 *   1. Replace LENGTH values with your chip's flash + RAM sizes.
 *      - STM32F411  : FLASH 512K, RAM 128K
 *      - STM32F446  : FLASH 512K, RAM 128K (+ optional CCM)
 *      - nRF52840   : FLASH 1024K, RAM 256K
 *      - RP2040     : FLASH n/a (XIP), RAM 264K
 *   2. Adjust ORIGIN if your chip's RAM/flash live elsewhere.
 *   3. Confirm the model fits: compare the lib's
 *      `PEAK_ACTIVATION_BYTES_UPPER_BOUND` constant to (RAM size
 *      MINUS bss/static MINUS heap reservation MINUS some safety
 *      margin). If it doesn't fit, your `predict()` call will
 *      hard-fault on stack overflow. Run `bin/stack_probe` on
 *      QEMU to get the real high-water mark, which is normally
 *      lower than the upper bound.
 *   4. cortex-m-rt places the stack at the top of RAM by default;
 *      that's correct for most targets. If you carve a heap or
 *      static buffer out, account for it.
 */
MEMORY
{
    FLASH : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}
