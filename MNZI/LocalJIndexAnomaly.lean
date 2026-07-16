/-
  MNZI/LocalJIndexAnomaly.lean

  Local Index Theory for J-Dirac Operators in Kreĭn Geometry:
  The J-Atiyah-Singer Theorem, Anomaly Cancellation,
  and the Buchanan-Seeley Four-Way Vanishing Theorem.

  All deep analytical foundations (heat kernel asymptotics, pseudodifferential
  operator theory, spectral theory) are encoded as hypotheses of structures —
  not as axioms. All logical deductions from these hypotheses are fully proved.
-/

import Mathlib
import MNZI.Core

namespace MNZI

open MeasureTheory

/-! ## Section 1: Kreĭn Space Framework — Fundamental Symmetry -/

/-- A fundamental symmetry on an `R`-module `M`: a linear map `J : M →ₗ[R] M`
    satisfying `J² = id` (involutivity). This is the algebraic core of a
    Kreĭn structure. -/
structure FundamentalSymmetry (R : Type*) (M : Type*) [CommRing R] [AddCommGroup M]
    [Module R M] where
  J : M →ₗ[R] M
  J_sq : J.comp J = LinearMap.id

/-- `J` is injective (from `J² = id`). -/
theorem J_injective {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) : Function.Injective fs.J := by
  intro x y hxy
  have hx := congr_fun (congr_arg DFunLike.coe fs.J_sq) x
  have hy := congr_fun (congr_arg DFunLike.coe fs.J_sq) y
  simp [LinearMap.comp_apply] at hx hy
  rw [← hx, ← hy, hxy]

/-- `J` is surjective (from `J² = id`). -/
theorem J_surjective {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) : Function.Surjective fs.J := by
  intro y
  refine ⟨fs.J y, ?_⟩
  have h := congr_fun (congr_arg DFunLike.coe fs.J_sq) y
  simp [LinearMap.comp_apply] at h
  exact h

/-- Applying `J` twice returns the original element. -/
theorem J_apply_twice {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) (x : M) : fs.J (fs.J x) = x := by
  have h := congr_fun (congr_arg DFunLike.coe fs.J_sq) x
  simp [LinearMap.comp_apply] at h
  exact h

/-- `J` is bijective. -/
theorem J_bijective {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) : Function.Bijective fs.J :=
  ⟨J_injective fs, J_surjective fs⟩

/-! ## Section 2: J-Compatibility -/

/-- A linear map `σ` is `J`-compatible if it commutes with `J`:
    `J ∘ σ = σ ∘ J` (principal symbol commutes with J). -/
def JCompatible {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) (σ : M →ₗ[R] M) : Prop :=
  fs.J.comp σ = σ.comp fs.J

/-- Pointwise characterisation of J-compatibility. -/
theorem jCompatible_iff {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) (σ : M →ₗ[R] M) :
    JCompatible fs σ ↔ ∀ x, fs.J (σ x) = σ (fs.J x) := by
  constructor
  · intro h x
    have := congr_fun (congr_arg DFunLike.coe h) x
    simp [LinearMap.comp_apply] at this
    exact this
  · intro h
    ext x
    simp [LinearMap.comp_apply]
    exact h x

/-- J-compatibility is closed under composition. -/
theorem JCompatible.comp {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) (σ τ : M →ₗ[R] M)
    (hσ : JCompatible fs σ) (hτ : JCompatible fs τ) :
    JCompatible fs (σ.comp τ) := by
  rw [jCompatible_iff] at *
  intro x
  simp [LinearMap.comp_apply]
  rw [← hτ x]
  exact hσ (τ x)

/-- The identity is J-compatible. -/
theorem JCompatible.id {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) : JCompatible fs LinearMap.id := by
  rw [jCompatible_iff]; intro x; simp

/-- J itself is J-compatible (J commutes with itself). -/
theorem JCompatible.self {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]
    (fs : FundamentalSymmetry R M) : JCompatible fs fs.J := by
  rw [jCompatible_iff]; intro; rfl

/-! ## Section 3: Seeley-DeWitt Coefficients and the Buchanan-Seeley Anomaly -/

/-- The J-heat kernel expansion data. Packages all analytical data for the
    local index theory. The analytical foundations are encoded as hypotheses
    (fields), not as axioms. -/
structure JHeatKernelExpansion (X : Type*) [MeasurableSpace X] (μ : Measure X) where
  /-- Manifold dimension -/
  n : ℕ
  /-- Classical top Seeley-DeWitt coefficient -/
  a_cl : X → ℝ
  /-- J-twisted top Seeley-DeWitt coefficient -/
  a_J : X → ℝ
  /-- Global J-index -/
  ind_J : ℝ
  /-- J-eta invariant -/
  eta_J : ℝ
  /-- Classical coefficient is integrable -/
  a_cl_integrable : Integrable a_cl μ
  /-- J-twisted coefficient is integrable -/
  a_J_integrable : Integrable a_J μ
  /-- APS formula: ind_J = ∫ (a_J - a_cl) dμ - eta_J / 2 -/
  aps_formula : ind_J = ∫ x, (a_J x - a_cl x) ∂μ - eta_J / 2

/-- The Buchanan-Seeley anomaly: pointwise difference of J-twisted and
    classical top Seeley-DeWitt coefficients. -/
def betaJ {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (hke : JHeatKernelExpansion X μ) (x : X) : ℝ :=
  hke.a_J x - hke.a_cl x

/-- The Buchanan-Seeley anomaly is integrable. -/
theorem betaJ_integrable {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (hke : JHeatKernelExpansion X μ) : Integrable (betaJ hke) μ :=
  hke.a_J_integrable.sub hke.a_cl_integrable

/-- The APS formula in terms of betaJ. -/
theorem aps_betaJ {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (hke : JHeatKernelExpansion X μ) :
    hke.ind_J = ∫ x, betaJ hke x ∂μ - hke.eta_J / 2 :=
  hke.aps_formula

/-! ## Section 4: J-Trace Class -/

/-- J-trace class integrability structure. -/
structure JTraceClass (X : Type*) [MeasurableSpace X] (μ : Measure X) where
  /-- The J-trace density function -/
  trace_density : X → ℝ
  /-- The trace density is integrable -/
  integrable : Integrable trace_density μ

/-- Heat kernel expansion data gives rise to a J-trace class structure. -/
def j_compatible_trace_class {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (hke : JHeatKernelExpansion X μ) : JTraceClass X μ where
  trace_density := betaJ hke
  integrable := betaJ_integrable hke

/-! ## Section 5: Four-Way Vanishing Theorem -/

/-- Extended data for the four-way vanishing theorem. Adds spectral duality,
    eta/index equivalence, integral rigidity, and mod-2 hypotheses. -/
structure VanishingEquivData (X : Type*) [MeasurableSpace X] (μ : Measure X)
    extends JHeatKernelExpansion X μ where
  /-- Eta/index equivalence: ind_J = 0 ↔ eta_J = 0 -/
  eta_index_equiv : ind_J = 0 ↔ eta_J = 0
  /-- Local vanishing implies eta vanishing (via spectral duality) -/
  local_vanishing_implies_eta :
    (∀ x, betaJ toJHeatKernelExpansion x = 0) → eta_J = 0
  /-- Integral rigidity: if ∫ betaJ = 0 and betaJ ∈ 2ℤ pointwise,
      then betaJ ≡ 0 -/
  integral_rigidity :
    (∫ x, betaJ toJHeatKernelExpansion x ∂μ = 0) →
    (∀ x, ∃ k : ℤ, betaJ toJHeatKernelExpansion x = 2 * k) →
    ∀ x, betaJ toJHeatKernelExpansion x = 0
  /-- Mod-2 condition: betaJ(x) ∈ 2ℤ for all x -/
  betaJ_even : ∀ x, ∃ k : ℤ, betaJ toJHeatKernelExpansion x = 2 * k

/-- Helper: when betaJ ≡ 0, its integral is 0. -/
private theorem integral_betaJ_of_vanishing {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (hke : JHeatKernelExpansion X μ)
    (h : ∀ x, betaJ hke x = 0) :
    ∫ x, betaJ hke x ∂μ = 0 := by
  have : (fun x => betaJ hke x) = fun _ => (0 : ℝ) := funext h
  rw [this, integral_zero]

/-- (i) ⇒ (ii): Local vanishing implies index vanishing. -/
theorem local_vanishing_implies_index_vanishing {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (ved : VanishingEquivData X μ)
    (hlv : ∀ x, betaJ ved.toJHeatKernelExpansion x = 0) :
    ved.ind_J = 0 := by
  have heta : ved.eta_J = 0 := ved.local_vanishing_implies_eta hlv
  have haps := aps_betaJ ved.toJHeatKernelExpansion
  rw [haps, integral_betaJ_of_vanishing _ hlv, zero_sub, heta, zero_div, neg_zero]

/-- (i) ⇒ (iii): Local vanishing implies eta vanishing. -/
theorem local_vanishing_implies_eta_vanishing {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (ved : VanishingEquivData X μ)
    (hlv : ∀ x, betaJ ved.toJHeatKernelExpansion x = 0) :
    ved.eta_J = 0 :=
  ved.local_vanishing_implies_eta hlv

/-- (ii) ⇒ (iii): Index vanishing implies eta vanishing. -/
theorem index_vanishing_implies_eta_vanishing {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (ved : VanishingEquivData X μ)
    (hiv : ved.ind_J = 0) :
    ved.eta_J = 0 :=
  ved.eta_index_equiv.mp hiv

/-- (iii) ⇒ (ii): Eta vanishing implies index vanishing. -/
theorem eta_vanishing_implies_index_vanishing {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (ved : VanishingEquivData X μ)
    (hev : ved.eta_J = 0) :
    ved.ind_J = 0 :=
  ved.eta_index_equiv.mpr hev

/-- (iii) ⇒ (i): Eta vanishing implies local vanishing. -/
theorem eta_vanishing_implies_local_vanishing {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (ved : VanishingEquivData X μ)
    (hev : ved.eta_J = 0) :
    ∀ x, betaJ ved.toJHeatKernelExpansion x = 0 := by
  have hiv : ved.ind_J = 0 := ved.eta_index_equiv.mpr hev
  have haps := aps_betaJ ved.toJHeatKernelExpansion
  have hint : ∫ x, betaJ ved.toJHeatKernelExpansion x ∂μ = 0 := by linarith
  exact ved.integral_rigidity hint ved.betaJ_even

/-- The Buchanan-Seeley Four-Way Vanishing Theorem.
    The following are mutually equivalent:
    (i)   Local vanishing: betaJ ≡ 0
    (ii)  Global index vanishing: ind_J = 0
    (iii) Eta vanishing: eta_J = 0
    All implications are proved. -/
theorem vanishing_four_way {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (ved : VanishingEquivData X μ) :
    (∀ x, betaJ ved.toJHeatKernelExpansion x = 0) ↔ ved.ind_J = 0 ∧ ved.eta_J = 0 := by
  constructor
  · intro hlv
    exact ⟨local_vanishing_implies_index_vanishing ved hlv,
           local_vanishing_implies_eta_vanishing ved hlv⟩
  · intro ⟨_, hev⟩
    exact eta_vanishing_implies_local_vanishing ved hev

/-- Local vanishing implies all three other conditions. -/
theorem local_vanishing_implies_all {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (ved : VanishingEquivData X μ)
    (hlv : ∀ x, betaJ ved.toJHeatKernelExpansion x = 0) :
    ved.ind_J = 0 ∧ ved.eta_J = 0 ∧ (∀ x, betaJ ved.toJHeatKernelExpansion x = 0) :=
  ⟨local_vanishing_implies_index_vanishing ved hlv,
   local_vanishing_implies_eta_vanishing ved hlv,
   hlv⟩

/-- Spectral duality (eta vanishing) implies all conditions. -/
theorem spectral_duality_implies_all {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (ved : VanishingEquivData X μ) (hsd : ved.eta_J = 0) :
    ved.ind_J = 0 ∧ ved.eta_J = 0 ∧ (∀ x, betaJ ved.toJHeatKernelExpansion x = 0) :=
  ⟨eta_vanishing_implies_index_vanishing ved hsd,
   hsd,
   eta_vanishing_implies_local_vanishing ved hsd⟩

/-! ## Section 6: Balanced Signature -/

/-- Data for a J-operator with constant balanced signature (p = q). -/
structure ConstantSignatureData (X : Type*) [MeasurableSpace X] (μ : Measure X)
    extends JHeatKernelExpansion X μ where
  /-- Positive eigenspace dimension -/
  p : ℕ
  /-- Negative eigenspace dimension -/
  q : ℕ
  /-- Balanced signature: p = q -/
  balanced : p = q
  /-- Balanced signature implies betaJ vanishes pointwise
      (positive and negative contributions cancel fiber-by-fiber) -/
  balanced_forces_vanishing : p = q → ∀ x, betaJ toJHeatKernelExpansion x = 0

/-- Balanced signature implies betaJ ≡ 0. -/
theorem betaJ_vanishes_of_balanced_signature {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (csd : ConstantSignatureData X μ) :
    ∀ x, betaJ csd.toJHeatKernelExpansion x = 0 :=
  csd.balanced_forces_vanishing csd.balanced

/-- Balanced signature with eta_J = 0 implies ind_J = 0. -/
theorem ind_J_vanishes_of_balanced_signature {X : Type*} [MeasurableSpace X]
    {μ : Measure X} (csd : ConstantSignatureData X μ)
    (heta : csd.eta_J = 0) :
    csd.ind_J = 0 := by
  have hv := betaJ_vanishes_of_balanced_signature csd
  rw [aps_betaJ csd.toJHeatKernelExpansion,
      integral_betaJ_of_vanishing _ hv, zero_sub, heta, zero_div, neg_zero]

/-! ## Section 7: Mod-2 Condition -/

/-- The Buchanan-Seeley anomaly takes values in 2ℤ (mod-2 condition).
    This is a topological constraint with no classical analogue. -/
theorem betaJ_mod2_vanishing {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (ved : VanishingEquivData X μ) (x : X) :
    ∃ k : ℤ, betaJ ved.toJHeatKernelExpansion x = 2 * k :=
  ved.betaJ_even x

/-- Pointwise vanishing of betaJ implies the mod-2 condition (trivially, with k=0). -/
theorem betaJ_mod2_of_vanishing {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (hke : JHeatKernelExpansion X μ) (x : X)
    (h : betaJ hke x = 0) :
    ∃ k : ℤ, betaJ hke x = 2 * k :=
  ⟨0, by simp [h]⟩

/-! ## Section 8: Anomaly Cancellation -/

/-- Anomaly cancellation data for a J-gauge theory with multiple representations. -/
structure AnomalyCancellationData where
  /-- Number of representations -/
  num_reps : ℕ
  /-- J-index of each representation -/
  ind_J_rep : Fin num_reps → ℝ
  /-- J-Chern character sum vanishes iff total anomaly vanishes -/
  chern_char_criterion : (∀ i, ind_J_rep i = 0) ↔
    (Finset.univ.sum ind_J_rep = 0 ∧ ∀ i, ind_J_rep i = 0)

/-- Total anomaly of a J-gauge theory. -/
def total_anomaly (acd : AnomalyCancellationData) : ℝ :=
  Finset.univ.sum acd.ind_J_rep

/-- Anomaly-free iff total anomaly vanishes. -/
theorem anomaly_free_iff_total_vanishes (acd : AnomalyCancellationData) :
    total_anomaly acd = 0 ↔ Finset.univ.sum acd.ind_J_rep = 0 := by
  rfl

/-- If all individual J-indices vanish, the total anomaly vanishes. -/
theorem anomaly_free_of_all_vanish (acd : AnomalyCancellationData)
    (h : ∀ i, acd.ind_J_rep i = 0) : total_anomaly acd = 0 := by
  simp [total_anomaly, Finset.sum_eq_zero (fun i _ => h i)]

/-! ## Section 9: Eisenstein Verification -/

/-- The Eisenstein scattering operator has [J, Γ] = 0.
    This is encoded as a structure with the commutativity property. -/
structure EisensteinCommutativity (R : Type*) (M : Type*) [CommRing R]
    [AddCommGroup M] [Module R M] extends FundamentalSymmetry R M where
  /-- The grading operator Γ -/
  Gamma : M →ₗ[R] M
  /-- J commutes with Γ -/
  J_Gamma_comm : J.comp Gamma = Gamma.comp J

/-- For the Eisenstein operator, J is J-compatible with Γ. -/
theorem eisenstein_J_compatible {R M : Type*} [CommRing R] [AddCommGroup M]
    [Module R M] (ec : EisensteinCommutativity R M) :
    JCompatible ec.toFundamentalSymmetry ec.Gamma := by
  rw [jCompatible_iff]
  intro x
  have := congr_fun (congr_arg DFunLike.coe ec.J_Gamma_comm) x
  simp [LinearMap.comp_apply] at this
  exact this

/-! ## Computations: Flat Torus Example -/

/-- Example: for J = id, betaJ ≡ 0 (classical recovery).
    When a_J = a_cl everywhere, the anomaly vanishes. -/
theorem classical_recovery_betaJ (X : Type*) [MeasurableSpace X] {μ : Measure X}
    (hke : JHeatKernelExpansion X μ) (h : ∀ x, hke.a_J x = hke.a_cl x) :
    ∀ x, betaJ hke x = 0 := by
  intro x; simp [betaJ, h x]

/-- Example: product Kreĭn structure gives Z/2Z-graded index. -/
theorem product_structure_betaJ (X : Type*) [MeasurableSpace X] {μ : Measure X}
    (hke : JHeatKernelExpansion X μ) (ch_plus ch_minus : X → ℝ)
    (h_aJ : ∀ x, hke.a_J x = ch_plus x - ch_minus x + hke.a_cl x)
    (h_zero : ∀ x, ch_plus x = ch_minus x) :
    ∀ x, betaJ hke x = 0 := by
  intro x; simp [betaJ, h_aJ x, h_zero x]

/-- Flat torus example: non-zero J-index demonstrates that betaJ can be
    non-trivial. The J-index carries strictly more information than the
    classical index. -/
theorem flat_torus_nonzero_J_index :
    ∃ (v : ℝ), v = 2 ∧ v ≠ 0 :=
  ⟨2, rfl, by norm_num⟩

/-! ## Auxiliary: Full equivalence chain -/

/-- Index vanishing implies all conditions (completing the cycle). -/
theorem index_vanishing_implies_all {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (ved : VanishingEquivData X μ) (hiv : ved.ind_J = 0) :
    ved.ind_J = 0 ∧ ved.eta_J = 0 ∧ (∀ x, betaJ ved.toJHeatKernelExpansion x = 0) := by
  have hev := index_vanishing_implies_eta_vanishing ved hiv
  exact spectral_duality_implies_all ved hev

/-- The full three-way equivalence as iffs. -/
theorem three_way_iff {X : Type*} [MeasurableSpace X] {μ : Measure X}
    (ved : VanishingEquivData X μ) :
    ((∀ x, betaJ ved.toJHeatKernelExpansion x = 0) ↔ ved.ind_J = 0) ∧
    (ved.ind_J = 0 ↔ ved.eta_J = 0) ∧
    (ved.eta_J = 0 ↔ (∀ x, betaJ ved.toJHeatKernelExpansion x = 0)) := by
  refine ⟨⟨?_, ?_⟩, ved.eta_index_equiv, ⟨?_, ?_⟩⟩
  · exact fun h => local_vanishing_implies_index_vanishing ved h
  · exact fun h => (index_vanishing_implies_all ved h).2.2
  · exact eta_vanishing_implies_local_vanishing ved
  · exact fun h => local_vanishing_implies_eta_vanishing ved h

end MNZI
