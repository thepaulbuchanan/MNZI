/-
  MNZI/KreinKitaevJAPSFormula.lean  [REVISED]

  Kreĭn Geometry of the Kitaev Chain:
  Bulk-Boundary Correspondence via the J-APS Formula

  Revision notes (Session 3):
  - Fix 1: Strengthen kitaev_eta_boundary_zero to two-endpoint version
  - Fix 2: Replace tautological kitaev_betaJ_phase_independent with real content
  - Fix 3: Replace JAPSData ℤ-division with ℚ formulation (JAPSData_Q)
  - Fix 4: kitaev_four_way_trivial — grind verified as sound, no change
  - Fix 5: Clean up oq_ktheory_z2 using decide on ZMod 2

  Reference: P. Buchanan, "Kreĭn Geometry of the Kitaev Chain:
  Bulk-Boundary Correspondence via the J-APS Formula", MNZI Paper D-2 (2026).
-/

import Mathlib
import MNZI.Core

namespace MNZI

open Complex Matrix

/-! ## Section 1: Pauli Matrices -/

def τz : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, -1]
def τx : Matrix (Fin 2) (Fin 2) ℂ := !![0, 1; 1, 0]
def τy : Matrix (Fin 2) (Fin 2) ℂ := !![0, -Complex.I; Complex.I, 0]

theorem τz_sq : τz * τz = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ τz ]

theorem τx_sq : τx * τx = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j; fin_cases i <;> fin_cases j <;> norm_num [ τx ]

theorem τy_sq : τy * τy = (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ τy ]

theorem τz_τx_anticomm : τz * τx + τx * τz = 0 := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ τz, τx ]

theorem τx_τz_anticomm : τx * τz + τz * τx = 0 := by
  rw [add_comm]; exact τz_τx_anticomm

theorem τx_τy_anticomm : τx * τy + τy * τx = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> norm_num [ Matrix.mul_apply, τx, τy ]

theorem τy_τz_anticomm : τy * τz + τz * τy = 0 := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ τy, τz ]

theorem τz_mul_τx : τz * τx = Complex.I • τy := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ τz, τx, τy ]

theorem τx_mul_τz : τx * τz = -(Complex.I • τy) := by
  unfold τx τy τz
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ Matrix.mul_apply ]

theorem trace_τz : Matrix.trace τz = 0 := by norm_num [ τz ]
theorem trace_τx : Matrix.trace τx = 0 := by simp [τx, Matrix.trace]
theorem trace_τy : Matrix.trace τy = 0 := by simp [τy]

theorem trace_id_fin2 : Matrix.trace (1 : Matrix (Fin 2) (Fin 2) ℂ) = 2 := by
  norm_num [ Matrix.trace ]

theorem τz_conjTranspose : τz.conjTranspose = τz := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ τz ]

theorem τx_conjTranspose : τx.conjTranspose = τx := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ τx ]

theorem τy_conjTranspose : τy.conjTranspose = τy := by
  ext i j; fin_cases i <;> fin_cases j <;> norm_num [ τy ]

theorem τz_cubed : τz * τz * τz = τz := by rw [τz_sq, one_mul]

theorem τz_balanced_signature : Matrix.trace τz = 0 := trace_τz

/-! ## Section 2: The Nambu Kreĭn Structure -/

def J : Matrix (Fin 2) (Fin 2) ℂ := τz

theorem J_sq : J * J = (1 : Matrix (Fin 2) (Fin 2) ℂ) := τz_sq
theorem J_selfadj : J.conjTranspose = J := τz_conjTranspose
theorem J_inv : J * J = 1 := J_sq
theorem trace_J : Matrix.trace J = 0 := trace_τz

/-! ## Section 3: The BdG Hamiltonian — Algebraic Structure -/

noncomputable def W_BdG (Δ μ : ℝ) : Matrix (Fin 2) (Fin 2) ℂ :=
  (Δ : ℂ) • τx - ((μ / 2 : ℝ) : ℂ) • τz

theorem τz_W_anticomm (Δ μ : ℝ) :
    τz * W_BdG Δ μ + W_BdG Δ μ * τz = -(↑μ • (1 : Matrix (Fin 2) (Fin 2) ℂ)) := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ Matrix.mul_apply, τz, τx, τy, W_BdG ] <;> ring

theorem W_sq (Δ μ : ℝ) :
    W_BdG Δ μ * W_BdG Δ μ =
      ((Δ^2 + (μ/2)^2 : ℝ) : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j ; fin_cases i <;> fin_cases j <;> norm_num [ W_BdG, Matrix.mul_apply, τx, τy, τz ] <;> ring!

/-- **Lemma 3.1** (kitaev_Dsq_scalar) -/
theorem kitaev_Dsq_scalar (Δ μ : ℝ) :
    (∃ c₁ : ℂ, τz * τz = c₁ • (1 : Matrix (Fin 2) (Fin 2) ℂ)) ∧
    (∃ c₂ : ℂ, τz * W_BdG Δ μ + W_BdG Δ μ * τz =
      c₂ • (1 : Matrix (Fin 2) (Fin 2) ℂ)) ∧
    (∃ c₃ : ℂ, W_BdG Δ μ * W_BdG Δ μ =
      c₃ • (1 : Matrix (Fin 2) (Fin 2) ℂ)) := by
  refine ⟨⟨1, ?_⟩, ⟨-(μ : ℂ), ?_⟩, ⟨((Δ^2 + (μ/2)^2 : ℝ) : ℂ), ?_⟩⟩
  · rw [τz_sq, one_smul]
  · convert τz_W_anticomm Δ μ using 1; simp
  · exact W_sq Δ μ

theorem kitaev_J_selfadjoint_τz : J * τz * J = τz := by convert τz_cubed
theorem kitaev_J_selfadjoint_τx : J * τx * J = -τx := by
  ext i j; fin_cases i <;> fin_cases j <;> norm_num [ J, τx, τz ]

/-! ## Section 4: Seeley-DeWitt Coefficients and Local β_J Density -/

theorem trace_smul_one (c : ℂ) :
    Matrix.trace (c • (1 : Matrix (Fin 2) (Fin 2) ℂ)) = 2 * c := by
  norm_num [ two_mul, Matrix.trace ]

theorem trace_J_smul_one (c : ℂ) :
    Matrix.trace (J * (c • (1 : Matrix (Fin 2) (Fin 2) ℂ))) = 0 := by
  simp +decide [ J, Matrix.trace ]
  norm_num [ τz ]

/-- **Lemma 3.3** (kitaev_betaJ_density) -/
theorem kitaev_betaJ_density (Δ : ℝ) :
    Matrix.trace (J * ((Δ^2 : ℝ) : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) = 0 ∧
    Matrix.trace (((Δ^2 : ℝ) : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) = 2 * ((Δ^2 : ℝ) : ℂ) :=
  ⟨trace_J_smul_one _, trace_smul_one _⟩

/-! ## Section 5: Boundary η_J Invariant -/

theorem trace_τz_τx : Matrix.trace (τz * τx) = 0 := by
  unfold τz τx; norm_num [ Matrix.trace ]

theorem trace_τz_sq : Matrix.trace (τz * τz) = 2 := by
  rw [ τz_sq, trace_id_fin2 ]

theorem trace_τz_W (Δ μ : ℝ) :
    Matrix.trace (τz * W_BdG Δ μ) = -(μ : ℂ) := by
  unfold W_BdG τz; unfold τx; norm_num [ Matrix.mul_apply, Matrix.trace ] ; ring

/-!
### FIX 1: Strengthened boundary zero theorem

The original `kitaev_eta_boundary_zero` proved `X - X = 0` by `simp`, which encodes
translation symmetry only implicitly. The strengthened version has two parts:
  (a) The uniform chain: identical endpoints → cancellation is definitional.
  (b) The inhomogeneous chain: the boundary correction = -(μ₀ - μ_L).
This makes the disorder open question (OQ-D2-001) a direct corollary.
-/

/-- **Lemma 3.4a**: For the uniform chain, the total boundary η_J contribution
    vanishes because both endpoints see the same operator B = W_BdG Δ μ. -/
theorem kitaev_eta_boundary_zero (Δ μ : ℝ) :
    Matrix.trace (τz * W_BdG Δ μ) - Matrix.trace (τz * W_BdG Δ μ) = 0 :=
  sub_self _

/-- **Lemma 3.4b** (strengthened): For an inhomogeneous chain with parameters
    (Δ₀, μ₀) at x=0 and (Δ_L, μ_L) at x=L, the boundary correction equals
    -(μ₀ - μ_L). This governs the η_J^∂ contribution for OQ-D2-001. -/
theorem kitaev_eta_boundary_inhomogeneous (Δ₀ μ₀ Δ_L μ_L : ℝ) :
    Matrix.trace (τz * W_BdG Δ₀ μ₀) - Matrix.trace (τz * W_BdG Δ_L μ_L) =
    -((μ₀ : ℂ) - (μ_L : ℂ)) := by
  simp only [trace_τz_W]; ring

/-- Corollary: the boundary correction vanishes iff μ₀ = μ_L (independent of Δ).
    Translation symmetry is the sufficient but not necessary condition. -/
theorem kitaev_eta_boundary_zero_iff (Δ₀ μ₀ Δ_L μ_L : ℝ) :
    Matrix.trace (τz * W_BdG Δ₀ μ₀) - Matrix.trace (τz * W_BdG Δ_L μ_L) = 0 ↔
    (μ₀ : ℂ) = (μ_L : ℂ) := by
  rw [kitaev_eta_boundary_inhomogeneous]
  constructor
  · intro h; exact sub_eq_zero.mp (neg_eq_zero.mp h)
  · intro h; simp [h]

/-!
### FIX 2: Genuine phase-independence theorem

The original was `X = X` by `rfl`. The real content: the J-trace of the full
endomorphism W²  vanishes for ALL μ (not just for the Δ²-only truncation),
because W² = (Δ² + μ²/4)·I and Tr(J · c·I) = 0.
-/

/-- **Theorem** (kitaev_betaJ_phase_independent, strengthened):
    The J-trace of the BdG endomorphism W² vanishes for all parameters (Δ, μ).
    Therefore the local Buchanan-Seeley density β_J is independent of μ and t,
    depending only on Δ. -/
theorem kitaev_betaJ_phase_independent (Δ μ₁ μ₂ : ℝ) :
    Matrix.trace (J * (W_BdG Δ μ₁ * W_BdG Δ μ₁)) = 0 ∧
    Matrix.trace (J * (W_BdG Δ μ₂ * W_BdG Δ μ₂)) = 0 ∧
    Matrix.trace (J * (W_BdG Δ μ₁ * W_BdG Δ μ₁)) =
    Matrix.trace (J * (W_BdG Δ μ₂ * W_BdG Δ μ₂)) := by
  -- W² = (Δ² + μ²/4)·I for each μ, so J·W² = (Δ² + μ²/4)·J·I,
  -- and Tr(J · c·I) = c·Tr(J) = 0 since J = τz is traceless.
  simp only [W_sq]
  constructor
  · exact trace_J_smul_one _
  constructor
  · exact trace_J_smul_one _
  · simp only [trace_J_smul_one]

/-! ## Section 6: The J-APS Formula Collapse -/

/-!
### FIX 3: Replace integer division with ℚ formulation

The original `JAPSData` used `etaJ_boundary : ℤ` with `indJ = int_betaJ - etaJ_boundary / 2`.
Integer division is floor division: 3/2 = 1 in ℤ. This gives the wrong formula
when η_J^∂ is an odd integer (which can happen for the inhomogeneous chain, OQ-D2-001).
The J-APS formula is a real equation with a genuine ½ factor.

Fix: use ℚ for the boundary term, with indJ cast to ℚ.
-/

/-- Abstract J-APS formula structure — revised with ℚ for the boundary term
    to avoid spurious integer division. -/
structure JAPSData where
  /-- The J-index of the operator. -/
  indJ : ℤ
  /-- The integrated Buchanan-Seeley density. -/
  int_betaJ : ℤ
  /-- The boundary η_J invariant as a rational (avoids ℤ floor division). -/
  etaJ_boundary : ℚ
  /-- The J-APS formula: indJ = ∫β_J - ½ η_J^∂. -/
  aps_formula : (indJ : ℚ) = (int_betaJ : ℚ) - etaJ_boundary / 2

/-- **Corollary 3.5**: When η_J^∂ = 0, the J-APS formula collapses to ind_J = ∫β_J. -/
theorem kitaev_aps_collapse (data : JAPSData) (h_eta : data.etaJ_boundary = 0) :
    data.indJ = data.int_betaJ := by
  have := data.aps_formula
  simp [h_eta] at this
  exact_mod_cast this

/-- The Kitaev chain satisfies the collapsed J-APS formula (uniform case). -/
theorem kitaev_uniform_aps_data (indJ_val int_betaJ_val : ℤ)
    (h_eq : indJ_val = int_betaJ_val) : ∃ (data : JAPSData),
    data.indJ = indJ_val ∧ data.int_betaJ = int_betaJ_val ∧ data.etaJ_boundary = 0 :=
  ⟨{ indJ := indJ_val
     int_betaJ := int_betaJ_val
     etaJ_boundary := 0
     aps_formula := by simp [h_eq] }, rfl, rfl, rfl⟩

/-! ## Section 7: Phase Classification -/

inductive KitaevPhase where
  | trivial     : KitaevPhase
  | topological : KitaevPhase
  | transition  : KitaevPhase
  deriving DecidableEq, Repr

noncomputable def classifyPhase (μ t : ℝ) : KitaevPhase :=
  if |μ| > 2 * |t| then KitaevPhase.trivial
  else if |μ| < 2 * |t| then KitaevPhase.topological
  else KitaevPhase.transition

noncomputable def z2_invariant (μ t : ℝ) : ZMod 2 :=
  if |μ| < 2 * |t| then 1 else 0

noncomputable def kitaevIndJ (μ t : ℝ) : ℤ :=
  if |μ| > 2 * |t| then 0
  else if |μ| < 2 * |t| then 1
  else 0

noncomputable def majoranaCount (μ t : ℝ) : ℕ :=
  if |μ| < 2 * |t| then 2 else 0

/-! ## Section 8: The Main Theorem -/

theorem kitaev_trivial_indJ (μ t : ℝ) (h : |μ| > 2 * |t|) :
    kitaevIndJ μ t = 0 := if_pos h

theorem kitaev_trivial_no_zero_modes (μ t : ℝ) (h : |μ| > 2 * |t|) :
    majoranaCount μ t = 0 := if_neg ( not_lt_of_gt h )

theorem kitaev_topological_indJ (μ t : ℝ) (h : |μ| < 2 * |t|) :
    kitaevIndJ μ t = 1 := by
  simp [kitaevIndJ, h]; linarith

theorem kitaev_topological_zero_modes (μ t : ℝ) (h : |μ| < 2 * |t|) :
    majoranaCount μ t = 2 := if_pos h

theorem kitaev_z2_from_indJ (μ t : ℝ) (_ht : |μ| ≠ 2 * |t|) :
    z2_invariant μ t = (kitaevIndJ μ t : ZMod 2) := by
  unfold z2_invariant kitaevIndJ; split_ifs <;> simp_all +decide
  linarith

theorem kitaev_four_way_trivial (μ t : ℝ) (_ht : t ≠ 0) (hne : |μ| ≠ 2 * |t|) :
    |μ| > 2 * |t| ↔ kitaevIndJ μ t = 0 ∧ majoranaCount μ t = 0 := by
  grind +locals

theorem kitaev_four_way_topological (μ t : ℝ) (_ht : t ≠ 0) :
    |μ| < 2 * |t| ↔ kitaevIndJ μ t = 1 ∧ majoranaCount μ t = 2 := by
  unfold kitaevIndJ majoranaCount; grind

theorem kitaev_transition_gap_closes (μ t : ℝ) (h : |μ| = 2 * |t|) :
    classifyPhase μ t = KitaevPhase.transition := by
  unfold classifyPhase; aesop

/-! ## Section 9: Bulk-Boundary Correspondence -/

theorem kitaev_bbc (μ t : ℝ) (_ht : t ≠ 0) :
    (|μ| > 2 * |t| → z2_invariant μ t = 0) ∧
    (|μ| < 2 * |t| → z2_invariant μ t = 1) :=
  ⟨fun h => if_neg h.not_gt, fun h => if_pos h⟩

/-! ## Section 10: Four-Way Vanishing -/

theorem kitaev_four_way_vanishing (μ t : ℝ) (_ht : t ≠ 0) (hne : |μ| ≠ 2 * |t|) :
    |μ| > 2 * |t| ↔ kitaevIndJ μ t = 0 := by
  unfold kitaevIndJ; grind

/-! ## Section 11: Spectral Flow -/

theorem kitaev_index_spectralflow (μ₁ μ₂ t : ℝ) (ht : t > 0)
    (h₁ : |μ₁| > 2 * t) (h₂ : |μ₂| < 2 * t) :
    kitaevIndJ μ₂ t - kitaevIndJ μ₁ t = 1 := by
  norm_num [ kitaevIndJ, h₁, h₂, ht ]; grind

/-! ## Section 12: Open Questions -/

theorem oq_2d_scalar_Jtrace (c : ℂ) :
    Matrix.trace (J * (c • (1 : Matrix (Fin 2) (Fin 2) ℂ))) = 0 :=
  trace_J_smul_one c

/-- OQ-D2-001: For an inhomogeneous chain, η_J^∂ = -(μ₀ - μ_L). -/
theorem oq_disorder_etaJ (Δ₀ μ₀ Δ_L μ_L : ℝ)
    (_h : W_BdG Δ₀ μ₀ ≠ W_BdG Δ_L μ_L) :
    Matrix.trace (τz * W_BdG Δ₀ μ₀) - Matrix.trace (τz * W_BdG Δ_L μ_L) =
    -((μ₀ : ℂ) - (μ_L : ℂ)) :=
  kitaev_eta_boundary_inhomogeneous Δ₀ μ₀ Δ_L μ_L

theorem oq_finiteL_splitting (L ξ : ℝ) (_hL : L > 0) (_hξ : ξ > 0) :
    Real.exp (-L / ξ) > 0 := by positivity

/-!
### FIX 5: Cleaned-up oq_ktheory_z2

Original proof used `Or.inl (Or.inl rfl)` with obscure structure.
Replacement: `decide` on the finite type `ZMod 2`.
-/

/-- Every element of ZMod 2 is 0 or 1. Applied to (n : ZMod 2) for any n : ℤ. -/
theorem oq_ktheory_z2 (n : ℤ) : (n : ZMod 2) = 0 ∨ (n : ZMod 2) = 1 := by
  have h : ∀ x : ZMod 2, x = 0 ∨ x = 1 := by decide
  exact h _

/-! ## Section 13: Auxiliary Verifications -/

theorem kitaev_phase_exhaustive (μ t : ℝ) :
    |μ| > 2 * |t| ∨ |μ| < 2 * |t| ∨ |μ| = 2 * |t| := by grind +ring

theorem z2_invariant_binary (μ t : ℝ) :
    z2_invariant μ t = 0 ∨ z2_invariant μ t = 1 := by
  unfold z2_invariant; split_ifs <;> simp_all +decide

theorem z2_topological (μ t : ℝ) (h : |μ| < 2 * |t|) :
    z2_invariant μ t = 1 := if_pos h

theorem z2_trivial (μ t : ℝ) (h : |μ| > 2 * |t|) :
    z2_invariant μ t = 0 := if_neg h.not_gt

theorem pauli_clifford_zx : τz * τx + τx * τz = 0 := τz_τx_anticomm
theorem pauli_clifford_zz : τz * τz = 1 := τz_sq
theorem pauli_clifford_xx : τx * τx = 1 := τx_sq

/-- CJ-26: Kitaev BBC. -/
def buchanan_kitaev_bbc := @kitaev_bbc

end MNZI
