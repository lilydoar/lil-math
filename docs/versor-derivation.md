# Versor Types: Clifford Algebra Derivation

Reference for `Versor2Type` / `Versor3Type` in `src/linalg.zig`.

---

## Clifford Product Formulas

### Cl(3,0) — 3D versors

Even = `(a, p=e₁₂, q=e₁₃, r=e₂₃)`, Odd = `(x=e₁, y=e₂, z=e₃, w=e₁₂₃)`:

**Even × Even → Even**
- scalar: `a·a₂ − p·p₂ − q·q₂ − r·r₂`
- e₁₂: `a·p₂ + p·a₂ + r·q₂ − q·r₂`
- e₁₃: `a·q₂ + q·a₂ − r·p₂ + p·r₂`
- e₂₃: `a·r₂ + r·a₂ + q·p₂ − p·q₂`
- *Identical to `Rot3.mul`* ✓

**Even × Odd → Odd**
- e₁: `a·x + p·y + q·z − r·w`
- e₂: `a·y − p·x + q·w + r·z`
- e₃: `a·z − p·w − q·x − r·y`
- e₁₂₃: `a·w + p·z − q·y + r·x`

**Odd × Even → Odd**
- e₁: `a·x − p·y − q·z − r·w`
- e₂: `p·x + a·y − r·z + q·w`
- e₃: `q·x + r·y + a·z − p·w`
- e₁₂₃: `r·x − q·y + p·z + a·w`

**Odd × Odd → Even**
- scalar: `x₁·x₂ + y₁·y₂ + z₁·z₂ − w₁·w₂`
- e₁₂: `x₁·y₂ − y₁·x₂ + z₁·w₂ + w₁·z₂`
- e₁₃: `x₁·z₂ − y₁·w₂ − z₁·x₂ − w₁·y₂`
- e₂₃: `x₁·w₂ + y₁·z₂ − z₁·y₂ + w₁·x₂`

### Cl(2,0) — 2D versors

Even = `(a, b=e₁₂)`, Odd = `(x=e₁, y=e₂)`:

- Even×Even: `(a₁a₂ − b₁b₂, a₁b₂ + b₁a₂)` — complex multiplication
- Even×Odd: `(a·x + b·y, a·y − b·x)`
- Odd×Even: `(a·x − b·y, b·x + a·y)`
- Odd×Odd: `(x₁x₂ + y₁y₂, x₁y₂ − y₁x₂)`

---

## Sandwich Formula: `α(Ṽ) · v · V`

To apply a versor `V` to a vector `v`:

- `Ṽ` = reverse: negates grade-2 and grade-3 parts
- `α` = grade automorphism: negates odd-grade parts (grade 1, grade 3)
- For even V: `α(Ṽ) = Ṽ` (grade 0 and 2 are both even, both survive α)
- For odd V: `α(Ṽ) = −Ṽ` (grade 1 and 3 are both odd, both negated)

Vector `v` is embedded as grade-1: `vx·e₁ + vy·e₂ + vz·e₃`.

This differs from `Rot3`, which uses a bivector embedding (`vx·e₂₃ + vy·e₁₃ + vz·e₁₂`).
Both produce the same physical rotation for even versors, but only the grade-1 embedding supports reflections.

---

## Composition Order

The vector sandwich `α(Ṽ)·v·V` has **opposite** composition order from the bivector sandwich `R·v·R̃`.

- Bivector (Rot3): `P·Q` applies Q first, then P → `mul(a,b)` applies b first.
- Vector (Versor): `P·Q` applies P first, then Q → opposite!

**Fix:** define `Versor.mul(a, b)` as the Clifford product `b · a` (arguments swapped internally).
Result: `mul(a, b)` applies b first, then a — matching `Rot3.mul` semantics. Verified ✓

---

## Rot3 ↔ Versor3 Conversion: Negate e₁₃

The Hodge dual mapping introduces a sign difference on the e₁₃ component:

- **Rot3** stores: `(a, b01=e₁₂, b02=e₁₃, b12=e₂₃)` with `b02 = +sin(θ/2)·axis.y`
- **Versor (GA)** needs: `e₁₃ = −sin(θ/2)·axis.y` (negative, from `⋆e₂ = −e₁₃`)

```
Rot3(a, b01, b02, b12) → Versor3 even  {a, b01, -b02, b12}
Versor3 even  {a, p, q, r} → Rot3(a, p, -q, r)
```

**2D:** No sign flip needed. `Rot2(a, b)` maps directly to `Versor2 even (a, b)`.

---

## Reverse (Conjugate)

- Even `(a, p, q, r)` → Even `(a, −p, −q, −r)` — negate grade-2 (bivectors)
- Odd `(x, y, z, w)` → Odd `(x, y, z, −w)` — negate grade-3 (trivector), keep grade-1 (vector)

`V · Ṽ = |V|²` for unit versors ✓

---

## Summary Table

| Convention | Value |
|---|---|
| Sandwich formula | `α(Ṽ) · v · V` |
| `mul(a, b)` implementation | Clifford product `b · a` |
| `mul(a, b)` semantics | Applies b first, then a (matches Rot3) |
| 3D `fromAxisAngle` | `cos(θ/2) + sin(θ/2)·(az·e₁₂ − ay·e₁₃ + ax·e₂₃)` |
| Rot3 → Versor3 | Negate e₁₃ component |
| Versor3 → Rot3 | Negate e₁₃ component |
| 2D `fromAngle` | `cos(θ/2) + sin(θ/2)·e₁₂` — same as Rot2 |
| Rot2 → Versor2 | Direct copy, no sign change |
| Reflection from normal n | Odd versor `(nx, ny, nz, 0)` |

---

## Verification Summary

All formulas tested against reference implementations (200–400 random trials each):

- Clifford product formulas (all four parity cases) ✓
- Rodrigues rotation formula ✓
- Reflection formula `v − 2(v·n)n` ✓
- Determinant = +1 for even, −1 for odd ✓
- Result is always a pure grade-1 vector (no contamination) ✓
- Parity preservation: E×E→E, E×O→O, O×E→O, O×O→E ✓
- Associativity ✓
- Slerp within parity: interpolated odd versors stay unit length and produce det=−1 ✓
- 2D Cl(2,0) formulas ✓
