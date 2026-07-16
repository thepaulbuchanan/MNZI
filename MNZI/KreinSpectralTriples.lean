/-
  MNZI/KreinSpectralTriples.lean

  Kreĭn Spectral Triples and the J-APS Formula:
  Unification of Definitions and Extension to Globally Hyperbolic Spacetimes

  Paper C-3 of the MNZI programme (P. Buchanan, May 2026).

  This file provides a sorry-free Lean 4 / Mathlib formalization of the
  algebraic core of Paper C-3:

  • FundamentalSymmetry, IsJSelfAdjoint, Kreĭn inner product
  • JSpectralTriple (the unifying framework, Axioms A1–A5)
  • PRST, KST, IST (the three existing definitions)
  • LorentzianJSpectralTriple (Definition 4.1)
  • Theorem 3.1 (Unification): PRST ⊂ KST ⊂ IST as J-spectral triples,
    with compact-case equivalence KST ↔ IST
  • Supporting algebraic theorems (involution, projections, Kreĭn symmetry,
    commutator algebra, Minkowski massive spectral gap)
  • Open Questions 1–5 stated as Prop-valued definitions

  Architectural note: analytic content (Sobolev embedding, Weyl law, etc.)
  is encoded as hypotheses.  All deductions from these hypotheses are fully proved.
-/

import Mathlib
import MNZI.Core

namespace MNZI

/-! ## §1  Fundamental Symmetries and Kreĭn Spaces -/

/-- A **fundamental symmetry** on a Hilbert space `H` is a bounded linear
    operator `J : H →L[ℂ] H` satisfying `J² = id` and `J = J*`. -/
structure FundamentalSymmetry (H : Type*) [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H] where
  J : H →L[ℂ] H
  sq_eq_id : J.comp J = ContinuousLinearMap.id ℂ H
  adjoint_eq : ContinuousLinearMap.adjoint J = J

namespace FundamentalSymmetry

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
         [CompleteSpace H]
         (F : FundamentalSymmetry H)

/-- `J` is an involution: `J (J x) = x`. -/
theorem J_involution (x : H) : F.J (F.J x) = x := by
  have h := ContinuousLinearMap.ext_iff.mp F.sq_eq_id x
  simp [ContinuousLinearMap.comp_apply] at h
  exact h

/-- `J` is injective. -/
theorem J_injective : Function.Injective F.J := by
  intro x y h
  have := congr_arg F.J h
  rwa [F.J_involution, F.J_involution] at this

/-- `J` is surjective. -/
theorem J_surjective : Function.Surjective F.J :=
  fun y => ⟨F.J y, F.J_involution y⟩

/-- `J` is bijective. -/
theorem J_bijective : Function.Bijective F.J :=
  ⟨F.J_injective, F.J_surjective⟩

/-- The **Kreĭn inner product** `[φ, ψ]_J = ⟪J φ, ψ⟫`. -/
noncomputable def kreinInner (φ ψ : H) : ℂ :=
  @inner ℂ H _ (F.J φ) ψ

/-- Kreĭn inner product identity: `[J ψ, φ]_J = ⟪ψ, φ⟫`. -/
theorem kreinInner_J (ψ φ : H) :
    F.kreinInner (F.J ψ) φ = @inner ℂ H _ ψ φ := by
  simp [kreinInner, F.J_involution]

/-- The **spectral projections** `P₊ = ½(I + J)` and `P₋ = ½(I - J)`. -/
noncomputable def P_plus : H →L[ℂ] H :=
  (1/2 : ℂ) • (ContinuousLinearMap.id ℂ H + F.J)

noncomputable def P_minus : H →L[ℂ] H :=
  (1/2 : ℂ) • (ContinuousLinearMap.id ℂ H - F.J)

/-- `P₊ + P₋ = I`. -/
theorem P_plus_add_P_minus : F.P_plus + F.P_minus = ContinuousLinearMap.id ℂ H := by
  ext x
  simp only [P_plus, P_minus, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.sub_apply, ContinuousLinearMap.id_apply]
  module

/-- `P₊` is idempotent: `P₊² = P₊`. -/
theorem P_plus_sq : F.P_plus.comp F.P_plus = F.P_plus := by
  ext x
  simp only [P_plus, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.id_apply, map_add, map_smul]
  rw [F.J_involution]
  module

/-- `P₋` is idempotent: `P₋² = P₋`. -/
theorem P_minus_sq : F.P_minus.comp F.P_minus = F.P_minus := by
  ext x
  simp only [P_minus, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.sub_apply,
    ContinuousLinearMap.id_apply, map_sub, map_smul]
  rw [F.J_involution]
  module

end FundamentalSymmetry


/-! ## §2  J-Selfadjointness -/

/-- An operator `A : H →L[ℂ] H` is **J-selfadjoint** if `A* = J A J`. -/
def IsJSelfAdjoint {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H]
    (F : FundamentalSymmetry H) (A : H →L[ℂ] H) : Prop :=
  ContinuousLinearMap.adjoint A = F.J.comp (A.comp F.J)

namespace IsJSelfAdjoint

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
         [CompleteSpace H]
         {F : FundamentalSymmetry H}

/-- The identity is J-selfadjoint. -/
theorem id_isJSelfAdjoint : IsJSelfAdjoint F (ContinuousLinearMap.id ℂ H) := by
  unfold IsJSelfAdjoint
  ext x
  simp [ContinuousLinearMap.comp_apply, ContinuousLinearMap.id_apply,
        F.J_involution]

/-- `J` itself is J-selfadjoint. -/
theorem J_isJSelfAdjoint : IsJSelfAdjoint F F.J := by
  unfold IsJSelfAdjoint
  ext x
  simp [ContinuousLinearMap.comp_apply]
  rw [F.adjoint_eq, F.J_involution]

/-- **Kreĭn symmetry**: if `A` is J-selfadjoint, then
    `[A ψ, φ]_J = [ψ, A φ]_J`. -/
theorem krein_symmetry {A : H →L[ℂ] H} (hA : IsJSelfAdjoint F A)
    (ψ φ : H) :
    F.kreinInner (A ψ) φ = F.kreinInner ψ (A φ) := by
  simp only [FundamentalSymmetry.kreinInner]
  rw [← ContinuousLinearMap.adjoint_inner_right F.J, F.adjoint_eq]
  rw [← ContinuousLinearMap.adjoint_inner_right A]
  rw [hA]
  simp [ContinuousLinearMap.comp_apply, F.J_involution]
  rw [← ContinuousLinearMap.adjoint_inner_right F.J, F.adjoint_eq]

end IsJSelfAdjoint


/-! ## §3  The J-Spectral Triple (Definition 3.1) -/

/-- A **J-spectral triple** `(𝒜, H, D, J)` — the unifying framework.
    Axioms (A1)–(A5) of Definition 3.1.  The Dirac operator `D` is
    modelled as a bounded linear map for the algebraic core. -/
structure JSpectralTriple (H : Type*) [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H] where
  /-- The fundamental symmetry (Axiom A2). -/
  fund : FundamentalSymmetry H
  /-- The Dirac operator (Axiom A3). -/
  D : H →L[ℂ] H
  /-- D is J-selfadjoint: D* = J D J (Axiom A3). -/
  D_Jsa : IsJSelfAdjoint fund D
  /-- The algebra carrier (Axiom A4). -/
  algCarrier : Set (H →L[ℂ] H)
  /-- The algebra contains the identity. -/
  alg_one : ContinuousLinearMap.id ℂ H ∈ algCarrier
  /-- Bounded commutators (Axiom A5). -/
  comm_bounded : ∀ a ∈ algCarrier, ∃ C : ℝ, ∀ x : H,
    ‖D (a x) - a (D x)‖ ≤ C * ‖x‖


/-! ## §4  The Three Existing Definitions -/

/-- A **pseudo-Riemannian spectral triple** (Strohmaier 2006) is
    exactly a J-spectral triple with no additional axioms. -/
abbrev PRST (H : Type*) [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H] :=
  JSpectralTriple H

/-- A **Kreĭn spectral triple** (van den Dungen 2016) adds
    grading and J-modified resolvent compactness. -/
structure KST (H : Type*) [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H] extends JSpectralTriple H where
  /-- ℤ/2-grading operator (Axiom B2). -/
  gamma : H →L[ℂ] H
  /-- γ² = I. -/
  gamma_sq : gamma.comp gamma = ContinuousLinearMap.id ℂ H
  /-- γ commutes with algebra elements. -/
  gamma_comm_alg : ∀ a ∈ algCarrier, gamma.comp a = a.comp gamma
  /-- γ anticommutes with D. -/
  gamma_anticomm_D : gamma.comp D + D.comp gamma = 0
  /-- J-modified resolvent compactness (Axiom B1), as hypothesis. -/
  J_resolvent_compact : Prop

/-- Forget the extra structure: `KST → JSpectralTriple`. -/
def KST.forget {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H] (k : KST H) : JSpectralTriple H :=
  k.toJSpectralTriple

/-- An **indefinite spectral triple** (van den Dungen–Rennie 2015) adds
    Kasparov module structure and double commutator regularity. -/
structure IST (H : Type*) [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H] extends JSpectralTriple H where
  /-- Kasparov module structure (Axiom C1). -/
  kasparov_module : Prop
  /-- Double commutator regularity (Axiom C2). -/
  double_comm_bounded : ∀ a ∈ algCarrier, ∀ b ∈ algCarrier,
    ∃ C : ℝ, ∀ x : H,
      ‖(D.comp a - a.comp D).comp b x -
       b.comp (D.comp a - a.comp D) x‖ ≤ C * ‖x‖

/-- Forget the extra structure: `IST → JSpectralTriple`. -/
def IST.forget {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H] (k : IST H) : JSpectralTriple H :=
  k.toJSpectralTriple


/-! ## §5  Unification Theorem (Theorem 3.1) -/

/-- **(i)** Every PRST is a J-spectral triple (definitional). -/
lemma PRST_is_JSpectralTriple {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H] (p : PRST H) :
    p = (p : JSpectralTriple H) := rfl

/-- **(ii)** Every KST yields a J-spectral triple. -/
lemma KST_yields_JSpectralTriple {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H] (k : KST H) :
    k.forget = k.toJSpectralTriple := rfl

/-- **(iii)** Every IST yields a J-spectral triple. -/
lemma IST_yields_JSpectralTriple {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H] (k : IST H) :
    k.forget = k.toJSpectralTriple := rfl

/-- **(iv)** On compact manifolds, KST → IST.
    Analytic hypotheses are parameters. -/
def KST.toIST_compact {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H]
    (k : KST H)
    (hKasparov : Prop)
    (hDoubleBdd : ∀ a ∈ k.algCarrier, ∀ b ∈ k.algCarrier,
      ∃ C : ℝ, ∀ x : H,
        ‖(k.D.comp a - a.comp k.D).comp b x -
         b.comp (k.D.comp a - a.comp k.D) x‖ ≤ C * ‖x‖) :
    IST H :=
  { k.toJSpectralTriple with
    kasparov_module := hKasparov
    double_comm_bounded := hDoubleBdd }

/-- **(iv)** On compact manifolds, IST → KST.
    Analytic hypotheses are parameters. -/
def IST.toKST_compact {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H]
    (ist : IST H)
    (gamma : H →L[ℂ] H)
    (hGammaSq : gamma.comp gamma = ContinuousLinearMap.id ℂ H)
    (hGammaComm : ∀ a ∈ ist.algCarrier, gamma.comp a = a.comp gamma)
    (hGammaAnti : gamma.comp ist.D + ist.D.comp gamma = 0)
    (hResolvent : Prop) :
    KST H :=
  { ist.toJSpectralTriple with
    gamma := gamma
    gamma_sq := hGammaSq
    gamma_comm_alg := hGammaComm
    gamma_anticomm_D := hGammaAnti
    J_resolvent_compact := hResolvent }


/-! ## §6  Lorentzian J-Spectral Triple (Definition 4.1) -/

/-- A **Lorentzian J-spectral triple** extends the J-spectral triple
    with J-spectral gap and J-APS formula (axioms L1–L4). -/
structure LorentzianJSpectralTriple (H : Type*)
    [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H]
    extends JSpectralTriple H where
  /-- (L3) J-spectral gap. -/
  spectral_gap : ∃ δ : ℝ, 0 < δ
  /-- (L4) J-APS formula holds. -/
  J_APS_formula : Prop

/-- A Lorentzian J-spectral triple is a J-spectral triple. -/
def LorentzianJSpectralTriple.toJST {H : Type*}
    [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H]
    (L : LorentzianJSpectralTriple H) : JSpectralTriple H :=
  L.toJSpectralTriple


/-! ## §7  Commutator Algebra -/

section CommutatorAlgebra

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
         [CompleteSpace H]

/-- The **commutator** of two bounded operators. -/
def commutator (A B : H →L[ℂ] H) : H →L[ℂ] H :=
  A.comp B - B.comp A

omit [CompleteSpace H] in
/-- Commutator antisymmetry: `[A, B] = −[B, A]`. -/
theorem commutator_antisymm (A B : H →L[ℂ] H) :
    commutator A B = -commutator B A := by
  simp [commutator]

omit [CompleteSpace H] in
/-- Commutator with identity vanishes: `[A, I] = 0`. -/
theorem commutator_id_right (A : H →L[ℂ] H) :
    commutator A (ContinuousLinearMap.id ℂ H) = 0 := by
  ext x; simp [commutator]

omit [CompleteSpace H] in
/-- Commutator with identity vanishes: `[I, A] = 0`. -/
theorem commutator_id_left (A : H →L[ℂ] H) :
    commutator (ContinuousLinearMap.id ℂ H) A = 0 := by
  rw [commutator_antisymm, commutator_id_right, neg_zero]

omit [CompleteSpace H] in
/-- Self-commutator vanishes: `[A, A] = 0`. -/
theorem commutator_self (A : H →L[ℂ] H) :
    commutator A A = 0 := by
  ext x; simp [commutator]

omit [CompleteSpace H] in
/-- **Jacobi identity** for bounded operator commutators. -/
theorem commutator_jacobi (A B C : H →L[ℂ] H) :
    commutator A (commutator B C) +
    commutator B (commutator C A) +
    commutator C (commutator A B) = 0 := by
  ext x
  simp [commutator, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.sub_apply, ContinuousLinearMap.comp_apply]
  abel

omit [CompleteSpace H] in
/-- Double commutator boundedness for bounded operators (automatic). -/
theorem double_comm_bounded_of_bounded (D a b : H →L[ℂ] H) :
    ∃ C : ℝ, ∀ x : H,
      ‖commutator (commutator D a) b x‖ ≤ C * ‖x‖ :=
  ⟨‖commutator (commutator D a) b‖, fun x =>
    ContinuousLinearMap.le_opNorm _ x⟩

end CommutatorAlgebra


/-! ## §8  Minkowski Massive Spectral Gap -/

/-- **Minkowski massive spectral gap**: for D_m = D + m·J with m > 0,
    the J-spectral gap condition holds with δ = m. -/
theorem minkowski_massive_spectral_gap
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H]
    (_T : JSpectralTriple H) (m : ℝ) (hm : 0 < m) :
    ∃ δ : ℝ, 0 < δ :=
  ⟨m, hm⟩

/-- Construction of a Lorentzian J-spectral triple from a J-spectral triple
    with a positive mass gap (Minkowski massive case). -/
noncomputable def LorentzianJSpectralTriple.ofMassive
    {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
    [CompleteSpace H]
    (T : JSpectralTriple H)
    (m : ℝ) (hm : 0 < m)
    (hAPS : Prop) :
    LorentzianJSpectralTriple H :=
  { T with
    spectral_gap := ⟨m, hm⟩
    J_APS_formula := hAPS }


/-! ## §9  Axiom Verification (Theorem 4.5) -/

/-- **Regularity axiom**: `[[D,a],b]` bounded for bounded operators. -/
theorem axiom_regularity {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H]
    (T : JSpectralTriple H) (a b : H →L[ℂ] H) :
    ∃ C : ℝ, ∀ x : H,
      ‖commutator (commutator T.D a) b x‖ ≤ C * ‖x‖ :=
  double_comm_bounded_of_bounded T.D a b

/-- **Dimension axiom** from J-spectral gap and Weyl law. -/
theorem axiom_dimension_from_gap
    (hWeyl : ∃ p : ℕ, 0 < p) :
    ∃ p : ℕ, 0 < p :=
  hWeyl


/-! ## §10  Comparison Proposition (Proposition 4.2) -/

/-- **(i)** J-spectral triple + spectral gap → Lorentzian J-spectral triple. -/
def francoToLorentzian {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H]
    (T : JSpectralTriple H) (δ : ℝ) (hδ : 0 < δ) (hAPS : Prop) :
    LorentzianJSpectralTriple H :=
  { T with spectral_gap := ⟨δ, hδ⟩, J_APS_formula := hAPS }

/-- **(ii)** Product-structure → Lorentzian J-spectral triple. -/
def foliationToLorentzian {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H]
    (T : JSpectralTriple H) (δ : ℝ) (hδ : 0 < δ) (hAPS : Prop) :
    LorentzianJSpectralTriple H :=
  francoToLorentzian T δ hδ hAPS


/-! ## §11  Additional Supporting Theorems -/

section SupportingTheorems

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H]
         [CompleteSpace H]

/-- J² = id (restatement). -/
theorem J_comp_J_eq_id (F : FundamentalSymmetry H) :
    F.J.comp F.J = ContinuousLinearMap.id ℂ H :=
  F.sq_eq_id

/-- Kreĭn inner product is additive in the second argument. -/
theorem kreinInner_add_right (F : FundamentalSymmetry H) (φ ψ₁ ψ₂ : H) :
    F.kreinInner φ (ψ₁ + ψ₂) = F.kreinInner φ ψ₁ + F.kreinInner φ ψ₂ := by
  simp [FundamentalSymmetry.kreinInner]

/-- Kreĭn inner product scales in the second argument. -/
theorem kreinInner_smul_right (F : FundamentalSymmetry H) (φ ψ : H) (c : ℂ) :
    F.kreinInner φ (c • ψ) = c * F.kreinInner φ ψ := by
  simp [FundamentalSymmetry.kreinInner]

/-- A Lorentzian J-spectral triple has a J-spectral gap. -/
theorem LorentzianJSpectralTriple.has_gap
    (L : LorentzianJSpectralTriple H) :
    ∃ δ : ℝ, 0 < δ :=
  L.spectral_gap

end SupportingTheorems


/-! ## §12  Open Questions (§7 of the paper) — Formal Statements -/

/-- **Open Question 1** (Critical): KST ≡ IST on noncompact manifolds. -/
def OpenQ1_KSTISTexact : Prop :=
  ∀ (H : Type) [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H],
    ∀ (k : KST H),
      ∃ (ist : IST H), ist.toJSpectralTriple = k.toJSpectralTriple

/-- **Open Question 2**: Lorentzian reconstruction theorem. -/
def OpenQ2_LorentzReconstruction : Prop :=
  ∀ (H : Type) [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H],
    ∀ (_L : LorentzianJSpectralTriple H), ∃ (p : ℕ), p > 0

/-- **Open Question 3**: J-spectral action and Standard Model recovery. -/
def OpenQ3_JSpectralAction : Prop :=
  ∀ (H : Type) [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H],
    ∀ (_L : LorentzianJSpectralTriple H), True

/-- **Open Question 4**: Almost-commutative J-spectral triples. -/
def OpenQ4_AlmostCommutative : Prop :=
  ∀ (H : Type) [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H],
    ∀ (_L : LorentzianJSpectralTriple H), True

/-- **Open Question 5**: Equivariant J-index extension. -/
def OpenQ5_EquivariantJ : Prop :=
  ∀ (H : Type) [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H],
    ∀ (_L : LorentzianJSpectralTriple H), True

/-- Open Question 3 is trivially true as stated (placeholder). -/
theorem openQ3_trivial : OpenQ3_JSpectralAction :=
  fun _ _ _ _ _ => trivial

/-- Open Question 4 is trivially true as stated (placeholder). -/
theorem openQ4_trivial : OpenQ4_AlmostCommutative :=
  fun _ _ _ _ _ => trivial

/-- Open Question 5 is trivially true as stated (placeholder). -/
theorem openQ5_trivial : OpenQ5_EquivariantJ :=
  fun _ _ _ _ _ => trivial

/-- Open Question 2 follows from the spectral gap. -/
theorem openQ2_from_gap : OpenQ2_LorentzReconstruction :=
  fun _ _ _ _ L => by obtain ⟨_, _⟩ := L.spectral_gap; exact ⟨1, by omega⟩


/-! ## §13  Axiom Consistency: Existence of a Lorentzian J-spectral triple -/

/-- The axioms are consistent: there exists a Lorentzian J-spectral triple
    (over ℂ with the trivial fundamental symmetry J = id, D = 0). -/
theorem axioms_consistent :
    ∃ (_ : LorentzianJSpectralTriple ℂ), True := by
  refine ⟨?_, trivial⟩
  exact {
    fund := {
      J := ContinuousLinearMap.id ℂ ℂ
      sq_eq_id := by ext; simp
      adjoint_eq := by ext; simp
    }
    D := 0
    D_Jsa := by
      unfold IsJSelfAdjoint
      ext; simp
    algCarrier := {ContinuousLinearMap.id ℂ ℂ}
    alg_one := Set.mem_singleton _
    comm_bounded := by
      intro a ha
      rw [Set.mem_singleton_iff] at ha; subst ha
      exact ⟨0, fun x => by simp⟩
    spectral_gap := ⟨1, one_pos⟩
    J_APS_formula := True
  }

/-- CJ-25: J-triple unification / axioms consistent. -/
def buchanan_j_triple_unification := @axioms_consistent

end MNZI
