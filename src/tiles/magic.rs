//! Magic Bitboards for Sliders on 8×8 Tiles
//!
//! This module implements fast, deterministic magic bitboards for O(1) slider attack
//! generation within 8×8 tiles.
//!
//! ## Architecture
//! - Uses fixed Kannan-style magic numbers and shifts.
//! - Thread-safe initialization via `std::sync::Once`.
//! - Flat attack tables for cache efficiency.

use std::ptr::{addr_of, addr_of_mut};
use std::sync::Once;

// ============================================================================
// Public API
// ============================================================================

/// Get rook attacks for a square within a single tile.
#[inline(always)]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    debug_assert!(sq < 64);
    init();
    unsafe {
        let mask = *addr_of!(ROOK_MASKS).cast::<u64>().add(sq);
        let idx =
            (((occ & mask).wrapping_mul(*ROOK_MAGICS.get_unchecked(sq))) >> ROOK_SHIFT) as usize;
        *addr_of!(ROOK_ATTACKS).cast::<u64>().add((sq << 12) | idx)
    }
}

/// Get bishop attacks for a square within a single tile.
#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    debug_assert!(sq < 64);
    init();
    unsafe {
        let mask = *addr_of!(BISHOP_MASKS).cast::<u64>().add(sq);
        let shift = *BISHOP_SHIFTS.get_unchecked(sq) as u32;
        let idx = (((occ & mask).wrapping_mul(*BISHOP_MAGICS.get_unchecked(sq))) >> shift) as usize;
        *addr_of!(BISHOP_ATTACKS).cast::<u64>().add((sq << 9) | idx)
    }
}

/// Get queen attacks (union of bishop and rook attacks)
#[inline(always)]
pub fn queen_attacks(sq: usize, occ: u64) -> u64 {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

/// Get ray mask for a direction (0=N, 1=E, 2=S, 3=W, 4=NE, 5=SE, 6=SW, 7=NW)
pub fn get_ray_mask(sq: usize, dir: usize) -> u64 {
    init();
    unsafe { (*addr_of!(RAY_MASKS).cast::<[u64; 8]>().add(sq))[dir] }
}

/// Initialize magic bitboards (thread-safe, called automatically or at startup)
pub fn init() {
    INIT.call_once(|| unsafe { init_inner() });
}

/// Legacy alias for init()
pub fn init_magic_bitboards() {
    init();
}

// ============================================================================
// Internals
// ============================================================================

static INIT: Once = Once::new();

// Flat tables: [sq][index] but flattened for faster addressing
static mut ROOK_ATTACKS: [u64; 64 * 4096] = [0; 64 * 4096];
static mut BISHOP_ATTACKS: [u64; 64 * 512] = [0; 64 * 512];

static mut ROOK_MASKS: [u64; 64] = [0; 64];
static mut BISHOP_MASKS: [u64; 64] = [0; 64];

/// Ray masks for each square and direction.
/// Directions: 0=N, 1=E, 2=S, 3=W, 4=NE, 5=SE, 6=SW, 7=NW
static mut RAY_MASKS: [[u64; 8]; 64] = [[0; 8]; 64];

const ROOK_SHIFT: u32 = 52;

// Kannan-style “fixed shift” rook magics (64 entries)
const ROOK_MAGICS: [u64; 64] = [
    0x0080001020400080,
    0x0040001000200040,
    0x0080081000200080,
    0x0080040800100080,
    0x0080020400080080,
    0x0080010200040080,
    0x0080008001000200,
    0x0080002040800100,
    0x0000800020400080,
    0x0000400020005000,
    0x0000801000200080,
    0x0000800800100080,
    0x0000800400080080,
    0x0000800200040080,
    0x0000800100020080,
    0x0000800040800100,
    0x0000208000400080,
    0x0000404000201000,
    0x0000808010002000,
    0x0000808008001000,
    0x0000808004000800,
    0x0000808002000400,
    0x0000010100020004,
    0x0000020000408104,
    0x0000208080004000,
    0x0000200040005000,
    0x0000100080200080,
    0x0000080080100080,
    0x0000040080080080,
    0x0000020080040080,
    0x0000010080800200,
    0x0000800080004100,
    0x0000204000800080,
    0x0000200040401000,
    0x0000100080802000,
    0x0000080080801000,
    0x0000040080800800,
    0x0000020080800400,
    0x0000020001010004,
    0x0000800040800100,
    0x0000204000808000,
    0x0000200040008080,
    0x0000100020008080,
    0x0000080010008080,
    0x0000040008008080,
    0x0000020004008080,
    0x0000010002008080,
    0x0000004081020004,
    0x0000204000800080,
    0x0000200040008080,
    0x0000100020008080,
    0x0000080010008080,
    0x0000040008008080,
    0x0000020004008080,
    0x0000800100020080,
    0x0000800041000080,
    0x00FFFCDDFCED714A,
    0x007FFCDDFCED714A,
    0x003FFFCDFFD88096,
    0x0000040810002101,
    0x0001FFCE07800410,
    0x0000FFFFFE040008,
    0x00000FFFF8100040,
    0x0000FFFE81000040,
];

// Kannan-style bishop shifts + magics (64 each)
const BISHOP_SHIFTS: [u8; 64] = [
    58, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59, 59, 59, 57, 55, 55, 57, 59, 59, 59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 58,
];

const BISHOP_MAGICS: [u64; 64] = [
    0x0040201008040200,
    0x0000402010080400,
    0x0000804020100804,
    0x0001008040201008,
    0x0002010080402010,
    0x0004020100804020,
    0x0008040201008040,
    0x0010080402010080,
    0x0000020100804028,
    0x0000040201008050,
    0x0000080402010080,
    0x0000100804020100,
    0x0000200814040200,
    0x0000401028080400,
    0x0000802050100800,
    0x0001004020100800,
    0x0000020408102200,
    0x0000040810204400,
    0x0000081020408000,
    0x0000102040008000,
    0x0000204081010200,
    0x0000408102020400,
    0x0000810504020800,
    0x0001021008041000,
    0x0000040404080200,
    0x0000080808100400,
    0x0000101010200800,
    0x0000202020401000,
    0x0000404080810200,
    0x0000808101020400,
    0x0001010282040800,
    0x0002020405001000,
    0x0000100402010080,
    0x0000200805020100,
    0x0000401008040200,
    0x0000020020080080,
    0x0000040020100000,
    0x0000080040201000,
    0x0000100080402000,
    0x0000200100804000,
    0x0000050100802000,
    0x00000a0201004000,
    0x0000140402008000,
    0x0000280804010000,
    0x0000500810020000,
    0x0000a01020040000,
    0x0001402040080000,
    0x0002804081000000,
    0x0000080810200200,
    0x0000101020400400,
    0x0000202040008800,
    0x0000404080011000,
    0x0000808100022000,
    0x0001010200044000,
    0x0002020400088000,
    0x0004040800110000,
    0x0000010204102000,
    0x0000020408205000,
    0x0000040810400400,
    0x0000081020800800,
    0x0000102040011000,
    0x0000204800022000,
    0x0000408000044000,
    0x0001000000088000,
];

unsafe fn init_inner() {
    // Masks
    for sq in 0..64 {
        unsafe {
            *addr_of_mut!(ROOK_MASKS).cast::<u64>().add(sq) = gen_rook_mask(sq);
            *addr_of_mut!(BISHOP_MASKS).cast::<u64>().add(sq) = gen_bishop_mask(sq);

            // Ray masks
            let r = (sq / 8) as i32;
            let f = (sq % 8) as i32;

            let mut n = 0u64;
            for rr in (r + 1)..8 {
                n |= bit((rr as usize) * 8 + f as usize);
            }
            let mut e = 0u64;
            for ff in (f + 1)..8 {
                e |= bit((r as usize) * 8 + ff as usize);
            }
            let mut s = 0u64;
            for rr in (0..r).rev() {
                s |= bit((rr as usize) * 8 + f as usize);
            }
            let mut w = 0u64;
            for ff in (0..f).rev() {
                w |= bit((r as usize) * 8 + ff as usize);
            }

            let mut ne = 0u64;
            {
                let mut rr = r + 1;
                let mut ff = f + 1;
                while rr < 8 && ff < 8 {
                    ne |= bit((rr as usize) * 8 + ff as usize);
                    rr += 1;
                    ff += 1;
                }
            }
            let mut se = 0u64;
            {
                let mut rr = r - 1;
                let mut ff = f + 1;
                while rr >= 0 && ff < 8 {
                    se |= bit((rr as usize) * 8 + ff as usize);
                    rr -= 1;
                    ff += 1;
                }
            }
            let mut sw = 0u64;
            {
                let mut rr = r - 1;
                let mut ff = f - 1;
                while rr >= 0 && ff >= 0 {
                    sw |= bit((rr as usize) * 8 + ff as usize);
                    rr -= 1;
                    ff -= 1;
                }
            }
            let mut nw = 0u64;
            {
                let mut rr = r + 1;
                let mut ff = f - 1;
                while rr < 8 && ff >= 0 {
                    nw |= bit((rr as usize) * 8 + ff as usize);
                    rr += 1;
                    ff -= 1;
                }
            }

            let rms = addr_of_mut!(RAY_MASKS).cast::<[u64; 8]>().add(sq);
            (*rms)[0] = n;
            (*rms)[1] = e;
            (*rms)[2] = s;
            (*rms)[3] = w;
            (*rms)[4] = ne;
            (*rms)[5] = se;
            (*rms)[6] = sw;
            (*rms)[7] = nw;
        }
    }

    // Rook attacks
    for sq in 0..64 {
        unsafe {
            let mask = *addr_of!(ROOK_MASKS).cast::<u64>().add(sq);
            let mut occ = 0u64;
            let p_magic = *ROOK_MAGICS.get_unchecked(sq);
            loop {
                let idx = (((occ & mask).wrapping_mul(p_magic)) >> ROOK_SHIFT) as usize;
                *addr_of_mut!(ROOK_ATTACKS)
                    .cast::<u64>()
                    .add((sq << 12) | idx) = gen_rook_attacks(sq, occ);

                occ = occ.wrapping_sub(mask) & mask;
                if occ == 0 {
                    break;
                }
            }
        }
    }

    // Bishop attacks
    for sq in 0..64 {
        unsafe {
            let mask = *addr_of!(BISHOP_MASKS).cast::<u64>().add(sq);
            let shift = *BISHOP_SHIFTS.get_unchecked(sq) as u32;
            let p_magic = *BISHOP_MAGICS.get_unchecked(sq);
            let mut occ = 0u64;
            loop {
                let idx = (((occ & mask).wrapping_mul(p_magic)) >> shift) as usize;
                *addr_of_mut!(BISHOP_ATTACKS)
                    .cast::<u64>()
                    .add((sq << 9) | idx) = gen_bishop_attacks(sq, occ);

                occ = occ.wrapping_sub(mask) & mask;
                if occ == 0 {
                    break;
                }
            }
        }
    }
}

#[inline(always)]
fn bit(sq: usize) -> u64 {
    1u64 << sq
}

fn gen_rook_mask(sq: usize) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut m = 0u64;
    for rr in (r + 1)..7 {
        m |= bit((rr as usize) * 8 + (f as usize));
    }
    for rr in 1..r {
        m |= bit((rr as usize) * 8 + (f as usize));
    }
    for ff in (f + 1)..7 {
        m |= bit((r as usize) * 8 + (ff as usize));
    }
    for ff in 1..f {
        m |= bit((r as usize) * 8 + (ff as usize));
    }
    m
}

fn gen_bishop_mask(sq: usize) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut m = 0u64;
    {
        let mut rr = r + 1;
        let mut ff = f + 1;
        while rr < 7 && ff < 7 {
            m |= bit((rr as usize) * 8 + (ff as usize));
            rr += 1;
            ff += 1;
        }
    }
    {
        let mut rr = r + 1;
        let mut ff = f - 1;
        while rr < 7 && ff > 0 {
            m |= bit((rr as usize) * 8 + (ff as usize));
            rr += 1;
            ff -= 1;
        }
    }
    {
        let mut rr = r - 1;
        let mut ff = f + 1;
        while rr > 0 && ff < 7 {
            m |= bit((rr as usize) * 8 + (ff as usize));
            rr -= 1;
            ff += 1;
        }
    }
    {
        let mut rr = r - 1;
        let mut ff = f - 1;
        while rr > 0 && ff > 0 {
            m |= bit((rr as usize) * 8 + (ff as usize));
            rr -= 1;
            ff -= 1;
        }
    }
    m
}

fn gen_rook_attacks(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut a = 0u64;
    for rr in (r + 1)..=7 {
        let s = (rr as usize) * 8 + (f as usize);
        a |= bit(s);
        if (occ & bit(s)) != 0 {
            break;
        }
    }
    for rr in (0..r).rev() {
        let s = (rr as usize) * 8 + (f as usize);
        a |= bit(s);
        if (occ & bit(s)) != 0 {
            break;
        }
    }
    for ff in (f + 1)..=7 {
        let s = (r as usize) * 8 + (ff as usize);
        a |= bit(s);
        if (occ & bit(s)) != 0 {
            break;
        }
    }
    for ff in (0..f).rev() {
        let s = (r as usize) * 8 + (ff as usize);
        a |= bit(s);
        if (occ & bit(s)) != 0 {
            break;
        }
    }
    a
}

fn gen_bishop_attacks(sq: usize, occ: u64) -> u64 {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut a = 0u64;
    {
        let mut rr = r + 1;
        let mut ff = f + 1;
        while rr <= 7 && ff <= 7 {
            let s = (rr as usize) * 8 + (ff as usize);
            a |= bit(s);
            if (occ & bit(s)) != 0 {
                break;
            }
            rr += 1;
            ff += 1;
        }
    }
    {
        let mut rr = r + 1;
        let mut ff = f - 1;
        while rr <= 7 && ff >= 0 {
            let s = (rr as usize) * 8 + (ff as usize);
            a |= bit(s);
            if (occ & bit(s)) != 0 {
                break;
            }
            rr += 1;
            ff -= 1;
        }
    }
    {
        let mut rr = r - 1;
        let mut ff = f + 1;
        while rr >= 0 && ff <= 7 {
            let s = (rr as usize) * 8 + (ff as usize);
            a |= bit(s);
            if (occ & bit(s)) != 0 {
                break;
            }
            rr -= 1;
            ff += 1;
        }
    }
    {
        let mut rr = r - 1;
        let mut ff = f - 1;
        while rr >= 0 && ff >= 0 {
            let s = (rr as usize) * 8 + (ff as usize);
            a |= bit(s);
            if (occ & bit(s)) != 0 {
                break;
            }
            rr -= 1;
            ff -= 1;
        }
    }
    a
}

// ============================================================================
// Cross-Tile Slider Rays
// ============================================================================

/// Check if a ray from `from_sq` in direction `(dx, dy)` exits the tile at edge.
/// Returns the exit direction as (tile_dx, tile_dy, edge_bit_index).
#[inline]
pub fn ray_exit_info(from_sq: usize, dx: i32, dy: i32) -> Option<(i64, i64, usize)> {
    let file = (from_sq % 8) as i32;
    let rank = (from_sq / 8) as i32;
    let mut r = rank;
    let mut f = file;
    let mut last_sq = from_sq;
    loop {
        let next_r = r + dy;
        let next_f = f + dx;
        if next_r < 0 || next_r > 7 || next_f < 0 || next_f > 7 {
            break;
        }
        r = next_r;
        f = next_f;
        last_sq = (r * 8 + f) as usize;
    }
    let mut tile_dx = 0;
    let mut tile_dy = 0;
    let next_r_exit = r + dy;
    let next_f_exit = f + dx;
    if next_f_exit > 7 {
        tile_dx = 1;
    } else if next_f_exit < 0 {
        tile_dx = -1;
    }
    if next_r_exit > 7 {
        tile_dy = 1;
    } else if next_r_exit < 0 {
        tile_dy = -1;
    }
    if tile_dx == 0 && tile_dy == 0 {
        None
    } else {
        Some((tile_dx, tile_dy, last_sq))
    }
}
