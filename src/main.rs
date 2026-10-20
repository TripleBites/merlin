// =====================================================================
//  TritVec basics: the math a ternary neural network needs
//
//  Run:  cargo run --release      Test:  cargo test
//
//  Every trit is stored as 2 bits, in two separate bit arrays ("planes"):
//
//      trit | nz bit | sign bit
//      -----+--------+---------
//        0  |   0    |   0
//       +1  |   1    |   0
//       -1  |   1    |   1
//
//  Rule we always keep true: sign bit = 0 wherever nz bit = 0,
//  and every bit past `len` is 0. Then padding never changes any result.
// =====================================================================

#[derive(Clone, Debug, PartialEq)]
struct TritVec {
    nz: Vec<u64>,   // bit i = 1  ->  trit i is nonzero
    sign: Vec<u64>, // bit i = 1  ->  trit i is negative
    len: usize,
}

impl TritVec {
    // ---------- 1. Create, read, write ---------------------------------

    fn zeros(len: usize) -> Self {
        let words = (len + 63) / 64; // 64 trits per u64 word, rounded up
        TritVec { nz: vec![0; words], sign: vec![0; words], len }
    }

    fn from_i8(vals: &[i8]) -> Self {
        let mut v = TritVec::zeros(vals.len());
        for (i, &t) in vals.iter().enumerate() {
            v.set(i, t);
        }
        v
    }

    fn set(&mut self, i: usize, t: i8) {
        assert!(i < self.len && (-1..=1).contains(&t));
        let word = i / 64;
        let bit = 1u64 << (i % 64);
        self.nz[word] &= !bit; // clear both bits first...
        self.sign[word] &= !bit;
        if t != 0 { self.nz[word] |= bit; } // ...then set what we need
        if t < 0 { self.sign[word] |= bit; }
    }

    fn get(&self, i: usize) -> i8 {
        let word = i / 64;
        let bit = 1u64 << (i % 64);
        if self.nz[word] & bit == 0 { 0 }
        else if self.sign[word] & bit != 0 { -1 }
        else { 1 }
    }

    fn to_i8(&self) -> Vec<i8> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    // ---------- 2. Negate: flip every sign ------------------------------
    //  Only nonzero trits get flipped, so we XOR sign with nz.
    //  (XOR with 1 flips a bit; XOR with 0 leaves it alone.)
    fn neg(&self) -> TritVec {
        let mut out = self.clone();
        for w in 0..out.nz.len() {
            out.sign[w] ^= out.nz[w];
        }
        out
    }

    // ---------- 3. Element-wise multiply --------------------------------
    //  Trit * trit is always a trit, so the result is another TritVec.
    //    nonzero  <=>  both nonzero          ->  a.nz & b.nz
    //    negative <=>  signs differ          ->  a.sign ^ b.sign
    //  (mask the sign with nz so the "sign = 0 where nz = 0" rule holds)
    fn mul(&self, other: &TritVec) -> TritVec {
        assert_eq!(self.len, other.len);
        let mut out = TritVec::zeros(self.len);
        for w in 0..self.nz.len() {
            out.nz[w] = self.nz[w] & other.nz[w];
            out.sign[w] = (self.sign[w] ^ other.sign[w]) & out.nz[w];
        }
        out
    }

    // ---------- 4. Sum of all trits -------------------------------------
    //  (#nonzero) - 2*(#negative):  every nonzero counts +1, and each
    //  negative one has to be turned from +1 into -1, a change of -2.
    fn sum(&self) -> i32 {
        let mut s = 0i32;
        for w in 0..self.nz.len() {
            s += self.nz[w].count_ones() as i32 - 2 * self.sign[w].count_ones() as i32;
        }
        s
    }

    // ---------- 5. Dot product ------------------------------------------
    //  dot(a, b) = sum(mul(a, b)), fused so no temporary vector is built.
    fn dot(&self, other: &TritVec) -> i32 {
        assert_eq!(self.len, other.len);
        let mut s = 0i32;
        for w in 0..self.nz.len() {
            let m = self.nz[w] & other.nz[w];
            let neg = m & (self.sign[w] ^ other.sign[w]);
            s += m.count_ones() as i32 - 2 * neg.count_ones() as i32;
        }
        s
    }

    fn count_nonzero(&self) -> usize {
        self.nz.iter().map(|w| w.count_ones() as usize).sum()
    }

    // ---------- 6. Visit each nonzero trit (used by the backward pass) ---
    //  `trailing_zeros` finds the lowest set bit; `w &= w - 1` clears it.
    //  This skips zeros entirely, so sparse vectors are cheap.
    fn for_each_nonzero(&self, mut f: impl FnMut(usize, i8)) {
        for word in 0..self.nz.len() {
            let mut bits = self.nz[word];
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                let i = word * 64 + b;
                let t = if self.sign[word] >> b & 1 == 1 { -1 } else { 1 };
                f(i, t);
                bits &= bits - 1; // clear the lowest set bit
            }
        }
    }
}

// ---------- 7. Activation: integer -> trit ------------------------------
//  The layer's output is a sum of trits (an i32). To pass it on as a trit,
//  threshold it:  > t  -> +1,   < -t -> -1,   otherwise 0.
fn ternarize(z: &[i32], t: i32) -> TritVec {
    let mut out = TritVec::zeros(z.len());
    for (i, &v) in z.iter().enumerate() {
        out.set(i, if v > t { 1 } else if v < -t { -1 } else { 0 });
    }
    out
}

// ---------- 8. A dense layer ----------------------------------------------
//  Weights are stored one TritVec per OUTPUT neuron (a row of W).
//  Then forward is just one dot product per output.
struct Layer {
    rows: Vec<TritVec>, // rows[j] = weights going into output neuron j
}

impl Layer {
    fn n_in(&self) -> usize { self.rows[0].len }
    fn n_out(&self) -> usize { self.rows.len() }

    // forward: z[j] = W[j] · x       (matrix * vector)
    fn forward(&self, x: &TritVec) -> Vec<i32> {
        self.rows.iter().map(|row| row.dot(x)).collect()
    }

    // backward: dx[i] = sum over j of W[j][i] * delta[j]   (transpose * vector)
    // delta is an integer error per output. We never build W-transpose:
    // walk each row's nonzero weights and add +-delta[j] where they point.
    fn backward(&self, delta: &[i32]) -> Vec<i32> {
        let mut dx = vec![0i32; self.n_in()];
        for (j, row) in self.rows.iter().enumerate() {
            if delta[j] == 0 { continue; }
            row.for_each_nonzero(|i, w| dx[i] += w as i32 * delta[j]);
        }
        dx
    }

    // weight "gradient" for one weight is just delta[j] * x[i].
    // (In Top this goes into the integer accumulator; here we only show it.)
    fn weight_grad(&self, x: &TritVec, delta: &[i32]) -> Vec<Vec<i32>> {
        let xs = x.to_i8();
        delta.iter().map(|&d| xs.iter().map(|&xi| d * xi as i32).collect()).collect()
    }
}

// ---------- printing helpers ----------------------------------------------
fn show(v: &TritVec) -> String {
    v.to_i8().iter().map(|&t| match t { 1 => " +", -1 => " -", _ => " 0" }).collect()
}
fn bits(w: u64, n: usize) -> String {
    (0..n).map(|i| if w >> i & 1 == 1 { " 1" } else { " ." }).collect()
}

fn main() {
    println!("\n----- 1. create / get / set -----");
    let mut a = TritVec::from_i8(&[1, -1, 0, 1, -1, 0, 0, 1]);
    println!("a        ={}", show(&a));
    println!("  nz     ={}", bits(a.nz[0], a.len));
    println!("  sign   ={}", bits(a.sign[0], a.len));
    a.set(2, -1);
    println!("a.set(2, -1)  ->  a ={}   a.get(2) = {}", show(&a), a.get(2));

    let b = TritVec::from_i8(&[1, 1, -1, -1, 0, 1, -1, 1]);
    println!("b        ={}", show(&b));

    println!("\n----- 2. negate -----");
    println!("-a       ={}", show(&a.neg()));

    println!("\n----- 3. element-wise multiply -----");
    println!("a        ={}", show(&a));
    println!("b        ={}", show(&b));
    println!("a * b    ={}", show(&a.mul(&b)));

    println!("\n----- 4. sum / count -----");
    println!("sum(a) = {}   nonzeros in a = {}", a.sum(), a.count_nonzero());

    println!("\n----- 5. dot product -----");
    println!("a . b = {}   (same as sum(a * b) = {})", a.dot(&b), a.mul(&b).sum());

    println!("\n----- 6. visiting nonzeros -----");
    print!("nonzero trits of b:");
    b.for_each_nonzero(|i, t| print!("  [{i}]={t:+}"));
    println!();

    println!("\n----- 7. activation (threshold t = 1) -----");
    let z = vec![5, 1, -1, -3, 0, 2];
    println!("z = {:?}  ->  ternarize ={}", z, show(&ternarize(&z, 1)));

    // ------------------------------------------------------------------
    println!("\n----- 8. a tiny network: 8 inputs -> 4 hidden -> 2 outputs -----");
    let x = TritVec::from_i8(&[1, 0, -1, 1, 1, 0, -1, 0]);

    let l1 = Layer { rows: vec![
        TritVec::from_i8(&[ 1,  0, -1,  1,  0,  0,  0,  0]),
        TritVec::from_i8(&[ 0,  1,  0,  0, -1,  1,  0,  0]),
        TritVec::from_i8(&[-1,  0,  0,  0,  1,  0,  1, -1]),
        TritVec::from_i8(&[ 1,  1,  1,  1,  1,  1,  1,  1]),
    ]};
    let l2 = Layer { rows: vec![
        TritVec::from_i8(&[ 1, -1,  0,  1]),
        TritVec::from_i8(&[-1,  1,  1,  0]),
    ]};
    println!("layer sizes: {}->{}, {}->{}", l1.n_in(), l1.n_out(), l2.n_in(), l2.n_out());

    // forward
    println!("x            ={}", show(&x));
    let z1 = l1.forward(&x);
    let h = ternarize(&z1, 0); // hidden activation
    println!("z1 = W1 . x  = {:?}", z1);
    println!("h  = tern(z1)={}", show(&h));
    let logits = l2.forward(&h); // last layer: keep the integers
    println!("logits = W2 . h = {:?}", logits);

    // backward with a simple integer error: target class 1
    // delta = output - target, target given as +3 for the right class, -3 otherwise
    let target = [-3, 3];
    let delta2: Vec<i32> = logits.iter().zip(&target).map(|(o, t)| o - t).collect();
    println!("\ntarget       = {:?}", target);
    println!("delta2 = logits - target = {:?}", delta2);

    let dh = l2.backward(&delta2);
    println!("dh = W2^T . delta2 = {:?}   (error sent back to the hidden layer)", dh);

    println!("\nweight grads for layer 2 (delta2[j] * h[i]):");
    for (j, g) in l2.weight_grad(&h, &delta2).iter().enumerate() {
        println!("  row {j}: {:?}", g);
    }
    println!("\n(Next step: feed these integer grads into the Top accumulators.)");
}

// =====================================================================
//  Tests: check every bit trick against plain i8 math.
// =====================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn rand_trits(seed: &mut u64, n: usize) -> Vec<i8> {
        (0..n).map(|_| {
            *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17;
            (*seed % 3) as i8 - 1
        }).collect()
    }

    #[test]
    fn ops_match_plain_math() {
        let mut s = 12345u64;
        for n in [1, 5, 63, 64, 65, 200] {
            for _ in 0..100 {
                let a = rand_trits(&mut s, n);
                let b = rand_trits(&mut s, n);
                let (ta, tb) = (TritVec::from_i8(&a), TritVec::from_i8(&b));

                assert_eq!(ta.to_i8(), a);
                let neg: Vec<i8> = a.iter().map(|x| -x).collect();
                assert_eq!(ta.neg().to_i8(), neg);
                let mul: Vec<i8> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
                assert_eq!(ta.mul(&tb).to_i8(), mul);
                assert_eq!(ta.sum(), a.iter().map(|&x| x as i32).sum::<i32>());
                assert_eq!(ta.dot(&tb), mul.iter().map(|&x| x as i32).sum::<i32>());
            }
        }
    }

    #[test]
    fn backward_is_transpose() {
        let mut s = 999u64;
        let rows: Vec<Vec<i8>> = (0..5).map(|_| rand_trits(&mut s, 70)).collect();
        let layer = Layer { rows: rows.iter().map(|r| TritVec::from_i8(r)).collect() };
        let delta = [3, -1, 0, 2, -5];
        let dx = layer.backward(&delta);
        for i in 0..70 {
            let expect: i32 = (0..5).map(|j| rows[j][i] as i32 * delta[j]).sum();
            assert_eq!(dx[i], expect);
        }
    }
}