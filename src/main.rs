// =====================================================================
//  TritVec + TritMatrix basics: the math a ternary neural network needs
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
//  and every bit past the logical length is 0. Then padding never
//  changes any result.
//
//  Key idea of this version: a TritVec and one row of a TritMatrix are
//  the SAME thing to the math -- a pair of word slices (nz, sign).
//  So all the bit tricks below are written once, on slices, and both
//  types just hand their slices to them.
// =====================================================================

// ---------------------------------------------------------------------
//  Part A: the bit math, on plain word slices
// ---------------------------------------------------------------------

/// Number of u64 words needed to hold `len` trits.
fn words_for(len: usize) -> usize {
    (len + 63) / 64
}

/// Write trit `t` at position `i` inside a pair of word slices.
fn set_trit(nz: &mut [u64], sign: &mut [u64], i: usize, t: i8) {
    assert!((-1..=1).contains(&t));
    let word = i / 64;
    let bit = 1u64 << (i % 64);
    nz[word] &= !bit; // clear both bits first...
    sign[word] &= !bit;
    if t != 0 { nz[word] |= bit; } // ...then set what we need
    if t < 0 { sign[word] |= bit; }
}

/// Read the trit at position `i`.
fn get_trit(nz: &[u64], sign: &[u64], i: usize) -> i8 {
    let word = i / 64;
    let bit = 1u64 << (i % 64);
    if nz[word] & bit == 0 { 0 }
    else if sign[word] & bit != 0 { -1 }
    else { 1 }
}

/// Ternary dot product: 3 bit ops + 2 popcounts per 64 trits.
///   m   = both nonzero
///   neg = both nonzero AND signs differ
///   dot = (#m) - 2*(#neg)
fn dot_words(a_nz: &[u64], a_sign: &[u64], b_nz: &[u64], b_sign: &[u64]) -> i32 {
    assert_eq!(a_nz.len(), b_nz.len());
    let mut s = 0i32;
    for w in 0..a_nz.len() {
        let m = a_nz[w] & b_nz[w];
        let neg = m & (a_sign[w] ^ b_sign[w]);
        s += m.count_ones() as i32 - 2 * neg.count_ones() as i32;
    }
    s
}

/// Call `f(index, trit)` for every nonzero trit, skipping zeros entirely.
/// `trailing_zeros` finds the lowest set bit; `bits &= bits - 1` clears it.
fn for_each_nonzero_words(nz: &[u64], sign: &[u64], mut f: impl FnMut(usize, i8)) {
    for word in 0..nz.len() {
        let mut bits = nz[word];
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            let t = if sign[word] >> b & 1 == 1 { -1 } else { 1 };
            f(word * 64 + b, t);
            bits &= bits - 1; // clear the lowest set bit
        }
    }
}

// ---------------------------------------------------------------------
//  Part B: TritVec -- one vector (we use it for activations: x, h, ...)
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TritVec {
    nz: Vec<u64>,   // bit i = 1  ->  trit i is nonzero
    sign: Vec<u64>, // bit i = 1  ->  trit i is negative
    len: usize,
}

impl TritVec {
    fn zeros(len: usize) -> Self {
        let words = words_for(len);
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
        assert!(i < self.len);
        set_trit(&mut self.nz, &mut self.sign, i, t);
    }

    fn get(&self, i: usize) -> i8 {
        assert!(i < self.len);
        get_trit(&self.nz, &self.sign, i)
    }

    fn to_i8(&self) -> Vec<i8> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    // Negate: flip the sign of every NONZERO trit -> XOR sign with nz.
    fn neg(&self) -> TritVec {
        let mut out = self.clone();
        for w in 0..out.nz.len() {
            out.sign[w] ^= out.nz[w];
        }
        out
    }

    // Element-wise multiply. Trit * trit is always a trit.
    //   nonzero  <=>  both nonzero   ->  a.nz & b.nz
    //   negative <=>  signs differ   ->  (a.sign ^ b.sign), masked by nz
    fn mul(&self, other: &TritVec) -> TritVec {
        assert_eq!(self.len, other.len);
        let mut out = TritVec::zeros(self.len);
        for w in 0..self.nz.len() {
            out.nz[w] = self.nz[w] & other.nz[w];
            out.sign[w] = (self.sign[w] ^ other.sign[w]) & out.nz[w];
        }
        out
    }

    // Sum: every nonzero counts +1, each negative one is a further -2.
    fn sum(&self) -> i32 {
        let mut s = 0i32;
        for w in 0..self.nz.len() {
            s += self.nz[w].count_ones() as i32 - 2 * self.sign[w].count_ones() as i32;
        }
        s
    }

    fn dot(&self, other: &TritVec) -> i32 {
        assert_eq!(self.len, other.len);
        dot_words(&self.nz, &self.sign, &other.nz, &other.sign)
    }

    fn count_nonzero(&self) -> usize {
        self.nz.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn for_each_nonzero(&self, f: impl FnMut(usize, i8)) {
        for_each_nonzero_words(&self.nz, &self.sign, f);
    }
}

// ---------------------------------------------------------------------
//  Part C: TritMatrix -- a whole weight matrix in TWO flat buffers
//
//  Row-major, one row per OUTPUT neuron, every row padded to a whole
//  number of u64 words (wpr = words per row):
//
//      nz:   [ row 0 words | row 1 words | row 2 words | ... ]
//      sign: [ row 0 words | row 1 words | row 2 words | ... ]
//
//  Row j lives at words  j*wpr .. (j+1)*wpr  in both buffers.
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TritMatrix {
    nz: Vec<u64>,
    sign: Vec<u64>,
    rows: usize, // number of output neurons
    cols: usize, // number of inputs (trits per row)
    wpr: usize,  // words per row = words_for(cols)
}

impl TritMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        let wpr = words_for(cols);
        TritMatrix { nz: vec![0; rows * wpr], sign: vec![0; rows * wpr], rows, cols, wpr }
    }

    /// Build from a list of rows, e.g. `TritMatrix::from_rows(&[[1, 0, -1], [0, 1, 1]])`.
    fn from_rows<const N: usize>(rows: &[[i8; N]]) -> Self {
        let mut m = TritMatrix::zeros(rows.len(), N);
        for (j, row) in rows.iter().enumerate() {
            for (i, &t) in row.iter().enumerate() {
                m.set(j, i, t);
            }
        }
        m
    }

    /// The word range that holds row j (same range in both buffers).
    fn row_range(&self, j: usize) -> std::ops::Range<usize> {
        assert!(j < self.rows);
        j * self.wpr..(j + 1) * self.wpr
    }

    /// Borrow row j as a pair of word slices -- no copying.
    fn row(&self, j: usize) -> (&[u64], &[u64]) {
        let r = self.row_range(j);
        (&self.nz[r.clone()], &self.sign[r])
    }

    fn set(&mut self, j: usize, i: usize, t: i8) {
        assert!(i < self.cols);
        let r = self.row_range(j);
        set_trit(&mut self.nz[r.clone()], &mut self.sign[r], i, t);
    }

    fn get(&self, j: usize, i: usize) -> i8 {
        assert!(i < self.cols);
        let (nz, sign) = self.row(j);
        get_trit(nz, sign, i)
    }

    fn row_to_i8(&self, j: usize) -> Vec<i8> {
        (0..self.cols).map(|i| self.get(j, i)).collect()
    }

    /// Bytes of actual weight storage (both planes).
    fn bytes(&self) -> usize {
        (self.nz.len() + self.sign.len()) * 8
    }
}

// ---------- Activation: integer -> trit --------------------------------
//  > t -> +1,   < -t -> -1,   otherwise 0.
fn ternarize(z: &[i32], t: i32) -> TritVec {
    let mut out = TritVec::zeros(z.len());
    for (i, &v) in z.iter().enumerate() {
        out.set(i, if v > t { 1 } else if v < -t { -1 } else { 0 });
    }
    out
}

// ---------------------------------------------------------------------
//  Part D: a dense layer, now backed by one TritMatrix
// ---------------------------------------------------------------------
struct Layer {
    w: TritMatrix, // w.rows = n_out, w.cols = n_in
}

impl Layer {
    fn n_in(&self) -> usize { self.w.cols }
    fn n_out(&self) -> usize { self.w.rows }

    // forward: z[j] = W[j] · x      (one dot product per row)
    // The rows are back to back in memory, so this streams straight
    // through the nz/sign buffers from start to end.
    fn forward(&self, x: &TritVec) -> Vec<i32> {
        assert_eq!(x.len, self.n_in());
        (0..self.n_out())
            .map(|j| {
                let (nz, sign) = self.w.row(j);
                dot_words(nz, sign, &x.nz, &x.sign)
            })
            .collect()
    }

    // backward: dx[i] = sum over j of W[j][i] * delta[j]   (W-transpose * delta)
    // Never build W-transpose: walk each row's nonzero weights and
    // scatter +-delta[j] into the input positions they point at.
    fn backward(&self, delta: &[i32]) -> Vec<i32> {
        assert_eq!(delta.len(), self.n_out());
        let mut dx = vec![0i32; self.n_in()];
        for j in 0..self.n_out() {
            if delta[j] == 0 { continue; }
            let (nz, sign) = self.w.row(j);
            for_each_nonzero_words(nz, sign, |i, w| dx[i] += w as i32 * delta[j]);
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

// ---------- printing helpers --------------------------------------------
fn show_i8(v: &[i8]) -> String {
    v.iter().map(|&t| match t { 1 => " +", -1 => " -", _ => " 0" }).collect()
}
fn show(v: &TritVec) -> String {
    show_i8(&v.to_i8())
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
    println!("\n----- 8. a TritMatrix: the 8 -> 4 layer's weights -----");
    let w1 = TritMatrix::from_rows(&[
        [ 1,  0, -1,  1,  0,  0,  0,  0],
        [ 0,  1,  0,  0, -1,  1,  0,  0],
        [-1,  0,  0,  0,  1,  0,  1, -1],
        [ 1,  1,  1,  1,  1,  1,  1,  1],
    ]);
    println!("rows = {}, cols = {}, words per row = {}", w1.rows, w1.cols, w1.wpr);
    println!("nz buffer has {} words, sign buffer has {} words  ->  {} bytes, 2 allocations",
             w1.nz.len(), w1.sign.len(), w1.bytes());
    println!("\n  row | trits            | nz word bits     | sign word bits");
    for j in 0..w1.rows {
        let (nz, sign) = w1.row(j);
        println!("  {j}   |{} |{} |{}  (words {:?})",
                 show_i8(&w1.row_to_i8(j)), bits(nz[0], 8), bits(sign[0], 8), w1.row_range(j));
    }
    println!("w1.get(2, 7) = {}", w1.get(2, 7));

    println!("\n----- 9. what a real MNIST layer costs (784 -> 256) -----");
    let big = TritMatrix::zeros(256, 784);
    println!("words per row = {}  ({} bits, {} of them padding)",
             big.wpr, big.wpr * 64, big.wpr * 64 - big.cols);
    println!("weights: {} bytes as TritMatrix  vs  {} bytes as f32",
             big.bytes(), big.rows * big.cols * 4);

    // ------------------------------------------------------------------
    println!("\n----- 10. a tiny network: 8 inputs -> 4 hidden -> 2 outputs -----");
    let x = TritVec::from_i8(&[1, 0, -1, 1, 1, 0, -1, 0]);
    let l1 = Layer { w: w1 };
    let l2 = Layer { w: TritMatrix::from_rows(&[
        [ 1, -1,  0,  1],
        [-1,  1,  1,  0],
    ])};
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

    fn rand_matrix(seed: &mut u64, rows: usize, cols: usize) -> (TritMatrix, Vec<Vec<i8>>) {
        let plain: Vec<Vec<i8>> = (0..rows).map(|_| rand_trits(seed, cols)).collect();
        let mut m = TritMatrix::zeros(rows, cols);
        for j in 0..rows {
            for i in 0..cols {
                m.set(j, i, plain[j][i]);
            }
        }
        (m, plain)
    }

    #[test]
    fn vec_ops_match_plain_math() {
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
    fn matrix_roundtrip_and_padding() {
        let mut s = 777u64;
        // 70 cols -> 2 words per row with 58 bits of padding in word 2
        let (m, plain) = rand_matrix(&mut s, 5, 70);
        for j in 0..5 {
            assert_eq!(m.row_to_i8(j), plain[j]);
            // padding bits past `cols` must stay zero in every row
            let (nz, sign) = m.row(j);
            let pad_mask = !0u64 << (70 % 64);
            assert_eq!(nz[1] & pad_mask, 0);
            assert_eq!(sign[1] & pad_mask, 0);
        }
    }

    #[test]
    fn forward_is_matrix_vector_product() {
        let mut s = 4242u64;
        for (rows, cols) in [(4, 8), (10, 63), (7, 64), (3, 65), (16, 784)] {
            let (m, plain) = rand_matrix(&mut s, rows, cols);
            let x = rand_trits(&mut s, cols);
            let layer = Layer { w: m };
            let z = layer.forward(&TritVec::from_i8(&x));
            for j in 0..rows {
                let expect: i32 = (0..cols).map(|i| plain[j][i] as i32 * x[i] as i32).sum();
                assert_eq!(z[j], expect, "rows={rows} cols={cols} j={j}");
            }
        }
    }

    #[test]
    fn backward_is_transpose() {
        let mut s = 999u64;
        let (m, plain) = rand_matrix(&mut s, 5, 70);
        let layer = Layer { w: m };
        let delta = [3, -1, 0, 2, -5];
        let dx = layer.backward(&delta);
        for i in 0..70 {
            let expect: i32 = (0..5).map(|j| plain[j][i] as i32 * delta[j]).sum();
            assert_eq!(dx[i], expect);
        }
    }
}