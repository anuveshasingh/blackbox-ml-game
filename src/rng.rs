//! NumPy's `np.random.default_rng(seed)`, reproduced bit for bit.
//!
//! The Python game sampled every puzzle with `default_rng(42).uniform(lo, hi, n)`.
//! That generator is PCG64 seeded through `SeedSequence`; both are ported here
//! so the Rust game shows exactly the same sample rows.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

/// `SeedSequence(seed).generate_state(4, np.uint64)` for a seed below 2³².
fn seed_sequence_state(seed: u32) -> [u64; 4] {
    let entropy = [seed];
    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], &mut hash_const));
            }
        }
    }

    let mut words = [0u32; 8];
    let mut hash_const = INIT_B;
    for (i, word) in words.iter_mut().enumerate() {
        let mut value = pool[i % POOL_SIZE] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        value = value.wrapping_mul(hash_const);
        *word = value ^ (value >> XSHIFT);
    }
    let mut state = [0u64; 4];
    for (i, s) in state.iter_mut().enumerate() {
        *s = words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32;
    }
    state
}

/// PCG64 (XSL-RR 128/64), as in `numpy.random.PCG64`.
pub struct Rng {
    state: u128,
    inc: u128,
}

impl Rng {
    /// `np.random.default_rng(seed)`.
    pub fn new(seed: u32) -> Self {
        let s = seed_sequence_state(seed);
        let init_state = (s[0] as u128) << 64 | s[1] as u128;
        let init_seq = (s[2] as u128) << 64 | s[3] as u128;
        let mut rng = Rng { state: 0, inc: (init_seq << 1) | 1 };
        rng.step();
        rng.state = rng.state.wrapping_add(init_state);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    /// A double in [0, 1), as `Generator.random()`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `Generator.uniform(lo, hi, n)`.
    pub fn uniform(&mut self, lo: f64, hi: f64, n: usize) -> Vec<f64> {
        let range = hi - lo;
        (0..n).map(|_| lo + range * self.next_f64()).collect()
    }
}
