# The Topological Pipeline for MLP/CNN Latent-Space Analysis
### A formal reference for the method implemented in the `ltep` package

*Definition-level statements of every step, intended to be lifted into the methods
section. **Construction** = a choice we make; **Fact** = an external result;
**Remark** = interpretation.*

---

## 0. Scope and summary

Let $F=f_{m}\circ\cdots\circ f_{1}$ be a trained feed-forward network and
$X\subset\mathbb R^{n_0}$ a finite sample. The pipeline assigns to $F$ a small set of
reproducible barcodes and scalars quantifying **where, along the depth of $F$, the
latent representations stop reorganising topologically**.

Significance enters **once**, on the **scale axis**: a bootstrap confidence band at
level $1-\alpha$ fixes, per layer, the resolution(s) $\varepsilon_i$ at which features
count as real. The **layer axis** is then pure *tracking* — a single full-data
barcode whose births/deaths give the convergence depth $d^\*$. $H_0$ (components) and
$H_1$ (loops) live at different scales, so each gets its **own** $\varepsilon$
sequence and its **own** barcode; the two are never merged.

---

## 1. Setup and notation

For $i=0,\dots,L-1$ let $X_0=X$ and $X_i=F_i(X)\subset\mathbb R^{n_i}$
($F_i=f_i\circ\cdots\circ f_1$) be the per-layer **representations**, $X_{L-1}$ the
output. Every $X_i$ shares the row index set $[N]$: row $r$ of each $X_i$ is the image
of $x_r$. This shared indexing is what makes the pullback tower (2.2) well defined.

$\mathrm{VR}_\varepsilon(Y)$ is the Vietoris–Rips complex of a cloud $Y$ at scale
$\varepsilon$ (a cheaper surrogate for Čech/nerve; qualitatively equivalent here).
$\mathrm{Dgm}_k\{K_t\}$ is the degree-$k$ persistence diagram of a filtration,
each point $(b,d)$ with persistence $d-b$; $W_\infty$ is the bottleneck distance.

Two filtrations on **incomparable** axes — bottleneck distances between them are not
comparable:

| object | filtration parameter | units |
|---|---|---|
| layer persistence | scale $\varepsilon$ | distance |
| MLP persistence | layer index $i$ | layers |

---

## 2. The two persistence objects

### 2.1 Layer persistence (scale axis)

**Construction 1.** For cloud $X_i$, $\mathrm{Dgm}_k(X_i):=\mathrm{Dgm}_k\{\mathrm{VR}_\varepsilon(X_i)\}_{\varepsilon\ge0}$,
births/deaths in distance units. Used to select $\varepsilon_i$ (Section 4) and to
test the output for a surviving loop (Section 8).

### 2.2 MLP persistence (layer axis)

Given scales $\varepsilon_0,\dots,\varepsilon_{L-1}$, build the **pullback tower**:
take $\mathrm{VR}_{\varepsilon_{L-1}}(X_{L-1})$ and pull it back through the layer maps
to get complexes $K_0,\dots,K_{L-1}$ on the shared vertex set $[N]$, where a simplex
survives at layer $i$ iff (i) its vertices are pairwise within $\varepsilon_i$ in
$X_i$ and (ii) its image survives at layer $i{+}1$.

**Construction 2 (combined filtration).** With
$\phi(\sigma)=\min\{i:\sigma\in K_i\}$ (first layer a simplex appears), the sublevel
sets $\{\phi\le t\}$ form a monotone filtration over the layer index. Its diagram
$\mathrm{Dgm}^{\mathrm{MLP}}_k:=\mathrm{Dgm}_k\{\phi\le t\}_{t=0}^{L-1}$ is the
**MLP-persistence barcode**, births/deaths in *layer units*.

**Remark 1 (semantics).** An $H_1$ class is born where its cycle is complete with no
filler and dies where a bounding $2$-chain first appears — i.e. where the network
*contracts* the loop; the death layer localises loop resolution. A class never filled
(death capped at the last layer) is **essential**: for $H_1$ an unresolved loop, for
$H_0$ a surviving component.

**Convention 1 (caps).** $H_0$ read over all $L$ representations (cap $L-1$); $H_1$
over loop-bearing layers excluding the $\approx 1$-D output (cap $L-2$).

---

## 3. Stage 0 — preprocessing of representations

Per dataset; recorded with the run.

**Construction 3 (principal-subspace normalisation, Hiraoka et al.).** Optional. From
the centred SVD $X_i=U\Sigma V^\top$ keep $k$ scores $Z=(U\Sigma)_{:,1:k}$ and rescale:
`global` $Z/\mathrm{rms}(Z)$, `whiten` per-PC unit variance, or `none`. **Fact:** in
high dimension pairwise distances concentrate and PH degrades; a linear projection to
a conservative $k$ near the signal dimension restores it (use `global` unless $k$ is
known). On ResNet block features the carrier flips $H_0\leftrightarrow H_1$ between
`global`-PCA and raw — evidence the raw $H_1$ is concentration noise.

**Construction 4 (diameter normalisation).** For COIL each $X_i$ is divided by
$\operatorname{diam}(X_i)$ before filtration. **Remark:** $\varepsilon$ selection is
exactly invariant under a global rescaling (it rescales $\tau$, the Betti curve and
the plateau identically), so this only fixes the numerical scale; it does **not**
change fallbacks, component counts, or $d^\*$.

**Construction 5 (sparsification).** Optional $\delta$-net $X_\delta\subseteq X$.
**Fact (stability):** $W_\infty(\mathrm{Dgm}_k(X),\mathrm{Dgm}_k(X_\delta))\le\delta$,
with $\delta$ the covering radius in **actual** distance units (not $\delta^2$).

---

## 4. Scale axis: two decoupled resolutions per layer

For each layer the pipeline computes **two** scales, $\varepsilon^{H_0}_i$ and
$\varepsilon^{H_1}_i$, and never combines them (no $\max$): a single $\varepsilon$
cannot be optimal for both, and $\max$ can only raise $\varepsilon$ — the one
direction along which a loop dies.

### 4.1 Bootstrap confidence band

**Construction 6 ($\tau$).** For diagram $\hat D_i=\mathrm{Dgm}_k(X_i)$, draw $B$ row
resamples with replacement, form $\hat D_i^{*(b)}$, and set
$$\tau_i=2\,\hat q_{1-\alpha}\big(\{W_\infty(\hat D_i,\hat D_i^{*(b)})\}_{b=1}^B\big).$$
**Fact (Fasy et al. 2014).** The band of half-width $\tau_i$ about the diagonal is an
asymptotically valid $(1-\alpha)$ confidence set; features with $d-b>\tau_i$ are
**significant** at level $\alpha$.

### 4.2 Selection: one three-tier ladder for both dimensions

Let $\beta_k(\varepsilon)$ be the degree-$k$ Betti curve and "plateau" the midpoint of
the widest $\varepsilon$-interval realising a target count. The **same ladder** is
applied to $H_0$ and $H_1$; the only asymmetry is how $\tau_i$ enters — a *scale* for
$H_0$ (components are born at $0$, so $\tau_i$ thresholds the death/merge level), a
*persistence* for $H_1$ (loops are scored by lifespan $d-b>\tau_i$).

**Construction 7 (the ladder).** *(This single ladder replaces the former separate
$H_0$/$H_1$ selectors and the median-only fallback.)* For layer $i$ and dimension $k$:
- **Tier 1 — significant.** If features with $\mathrm{pers}>\tau_i$ exist,
  $\varepsilon^{H_k}_i$ is the plateau where exactly those features are alive
  ($H_0$: the Betti$_0$ plateau at the significant count, restricted to
  $\varepsilon>\tau_i$ and tail-capped, C9; $H_1$: the widest plateau of the
  significant-loops-only Betti$_1$ curve, i.e. inside the alive window).
- **Tier 2 — most persistent (sub-threshold).** If none clears $\tau_i$ but degree-$k$
  features exist, $\varepsilon^{H_k}_i$ is the **alive-window midpoint of the single
  most persistent feature**, $\tfrac12(b^\*+d^\*)$ with $(b^\*,d^\*)=\arg\max(d-b)$
  (essential deaths capped at the grid max). **Flagged sub-threshold.**
- **Tier 3 — none.** If there is no degree-$k$ feature at all,
  $\varepsilon^{H_k}_i=\operatorname{median}\operatorname{pdist}(X_i)$ (geometric
  heuristic). **Flagged.**

**Remark (no $H_1\to H_0$ borrow).** Tier 2 is what makes the two scales fully
independent: when no loop is significant, $\varepsilon^{H_1}_i$ is the most-persistent
loop's own window — it is **never** set to $\varepsilon^{H_0}_i$. This matters at the
tower **anchor** (the last representation): the $H_1$ pullback tower is anchored at
$\mathrm{VR}_{\varepsilon^{H_1}_{L-1}}(X_{L-1})$, so the anchor must be a genuine $H_1$
scale for a loop to thread the pullback. A layer whose loop is real but sub-threshold
(e.g. an autoencoder bottleneck at a strict $\alpha$) now anchors at its loop's
alive-window rather than collapsing to a tiny $H_0$ scale.

### 4.3 Guards (each flagged, never silent)

- **Construction 9 ($H_0$ tail cap).** For a **Tier-1** $H_0$ selection, cap
  $\varepsilon^{H_0}_i$ at the last **significant** merge $\max\{e_j>\tau_i\}$ ($e_j$
  the MST edge weights / finite $H_0$ deaths), so $\varepsilon$ cannot sit in the
  fully-merged tail. Flag **capped**.
- **Construction 10 (degenerate).** Flag layer $i$ **degenerate** if
  $\tau_i<\rho\,\operatorname{diam}(X_i)$ ($\rho=\texttt{TAU\_FLOOR\_FRAC}$): a
  near-collapsed cloud whose significance test is unreliable.
- The **tier** itself is reported per layer per dimension ($\texttt{h0\_tier}$,
  $\texttt{h1\_tier}\in\{1,2,3\}$); Tier 3 sets the **fallback** flag.

### 4.4 Manual override (heuristic off)

**Construction 12.** A user-supplied per-layer vector (read off the layer diagrams)
replaces 4.1–4.3, yielding one barcode at hand-set scales. Length is checked against
$L$ (loud error on mismatch). Used to probe robustness of $d^\*$ to the scale choice.

**Per-layer audit.** Every layer reports
$\varepsilon^{H_0}_i,\varepsilon^{H_1}_i,\tau_i,n^{H_0}_{\mathrm{sig}},n^{H_1}_{\mathrm{sig}}$
and its flags.

---

## 5. Layer axis: tracking, no resampling

A layer-lifespan threshold is **not** used: a loop resolved in one transition has
lifespan $1$ yet is real, while boundary jitter can split a real feature across
adjacent integer layer-cells. **No resampling/recurrence test is applied either.**

**Construction 13.** Significance already lives in the per-layer $\varepsilon$ choice
(Section 4): the $\tau$ band has discarded noise, so **every** feature present at the
chosen $\varepsilon$ is genuine information from that layer. The MLP-persistence
barcode (C2) is therefore read directly — a single full-data barcode, all bars drawn,
repeated cells shown as multiplicities. From it, per dimension $k$:

- **resolution events** $\{(b,d):d<\mathrm{cap}_k\}$ (features that die);
- **resolved-by layer** $\rho_k=\max\{d:(b,d),\,d<\mathrm{cap}_k\}$;
- **unresolved (essential)** $u_k=\#\{(b,d):d=\mathrm{cap}_k\}$ (for $H_1$, the failure flag);
- **onset layers** $\{b\}$.

**Remark 2.** An optional `significance=True` path additionally qualifies bars by
bootstrap recurrence over row-resamples (a layer-axis stability gate using
`CONV_N_RESAMPLE`, `CONV_SUBSAMPLE_FRAC`, `AGREEMENT_MIN`). It is **not** the default
and is unused by the experiments; the no-resampling reading above is the method.

---

## 6. Convergence depth and the prunable tail

**Construction 14.** Over both dimensions of the single barcode,
$$d^\*=\max\{\,b,\ d\cdot\mathbf 1[d<\mathrm{cap}_k]\,:\ (b,d)\in\mathrm{Dgm}^{\mathrm{MLP}}_k\},$$
the last layer carrying any **genuine birth or finite (real) death**. Essential bars
(capped at the last layer) contribute only their birth, never inflating $d^\*$. The
**inert / prunable tail** is $\{d^\*+1,\dots,L-1\}$: trailing representations across
which no feature is born or dies, i.e. the same identities are carried unchanged.

**Remark 3 (robustness).** $d^\*$ is read per scale scheme ($\varepsilon^{H_0}$,
$\varepsilon^{H_1}$, or manual). Feature *counts* vary with the scale; the prunable
tail is typically stable across schemes — that stability is itself a reported result.

---

## 7. Homology reading: two barcodes and the carrier

**Construction 15 (relevant dimension for $\varepsilon$).** $k=0$ for the output and
any $X_i$ with $n_i<2$; $H_1$ is tracked on the interior loop-bearing layers up to a
cap `max_hom_dim`.

**Construction 16 (signal / simplification / carrier).** From a barcode:
- **signal dimension** — a degree in which a feature is *present*;
- **simplification dimension** — a degree in which a feature is *resolved* (finite
  death), hence drives $d^\*$;
- **carrier** $\kappa$ — the degree carrying the network's reorganisation.

Two barcodes are produced (one at $\varepsilon^{H_0}$, one at $\varepsilon^{H_1}$),
each displaying $H_0$ and $H_1$ at *its* scale. **Each experiment reads the barcode for
its carrier:** COIL autoencoder $\to H_1$ (a designed rotation loop, preserved);
COIL classifier and ResNet $\to H_0$; cardio / CIFAR-dense $\to H_0$. A carrier that
only appears at fallback layers ($n^{H_1}_{\mathrm{sig}}=0$) is noise, not signal.

---

## 8. Corroboration

**Construction 17 (cross-check).** For the carrier $\kappa$, compute consecutive
**diameter-normalised** layer-persistence diagrams and their bottleneck distances
$W_\infty(\widehat{\mathrm{Dgm}}_\kappa(X_i),\widehat{\mathrm{Dgm}}_\kappa(X_{i+1}))$.
Beyond $d^\*$ these should lie in the noise; a large value past $d^\*$ contradicts the
barcode. The last-hidden$\to$output transition is excluded (collapse, not
reorganisation).

**Construction 18 (output-loop anomaly).** If $\mathrm{Dgm}_1(X_{L-1})$ has a feature
with $\mathrm{pers}>\tau_{L-1}$, the output retains a significant loop — the network
has not linearised the data (a concrete failure signal).

---

## 9. Pruning validation (empirical falsifier of $d^\*$)

**Construction 19.** Representations are $[\text{input},h_1,\dots,h_m,\text{output}]$,
so $d^\*$ hidden layers do the topological work. Build the **pruned** network with
hidden widths `hidden_widths[:d*]` (inert tail removed), retrain from scratch under
the same data/seed/epochs, and compare test accuracy to the full network. A **control**
at `hidden_widths[:d*-1]` (one layer shorter, cutting into the working region) is also
retrained. The claim is supported iff pruned $\approx$ full while the control degrades
— accuracy, not topology, adjudicates inertness. (Implemented for the trained-MLP
datasets; not applicable to the pretrained ResNet family or to the loop-preserving
autoencoder, where accuracy is not the relevant metric.)

---

## 10. Algorithm

1. **Stage 0.** Extract $X_0,\dots,X_{L-1}$; preprocess (C3–C5), recording choices.
2. **Scale axis.** Per layer: $\tau_i$ (C6) $\Rightarrow$ both scales via the
   three-tier ladder (C7), guards (C9, C10); or manual (C12). Pre-commit $\alpha,B$.
3. **Two towers + barcodes.** Pullback tower (C2) at $\{\varepsilon^{H_0}_i\}$ and at
   $\{\varepsilon^{H_1}_i\}$ $\Rightarrow$ two MLP barcodes.
4. **Layer axis.** Read each barcode directly, no resampling (C13).
5. **Convergence.** $d^\*$ and inert tail per scheme (C14); carrier $\kappa$ (C16).
6. **Corroborate.** Cross-check (C17); output anomaly (C18).
7. **Validate.** Pruned retrain vs full and control (C19).

---

## 11. Reported quantities

| symbol | axis / units | meaning |
|---|---|---|
| $\tau_i$ | scale / distance | bootstrap noise floor at layer $i$ |
| $\varepsilon^{H_0}_i,\varepsilon^{H_1}_i$ | scale / distance | the two resolutions |
| $n^{H_0}_{\mathrm{sig}},n^{H_1}_{\mathrm{sig}}$, flags | — | significant counts; tiers $\{1,2,3\}$; {capped, fallback (tier 3), sub-$\tau$ (tier 2), degenerate} |
| $\mathrm{Dgm}^{\mathrm{MLP}}_k$ | layer | the two barcodes (per scheme) |
| $\rho_k$, $u_k$ | layer | resolved-by layer; unresolved/essential count |
| $d^\*$, inert tail | layer | convergence depth, prunable layers |
| $\kappa$ | — | carrier ($H_1$ vs $H_0$) |
| cross-check $W_\infty$ | scale | corroboration of $d^\*$ |
| pruned vs full / control acc | — | empirical prunability (C19) |

---

## 12. Pre-committed parameters

Fixed and reported; **never** retuned after seeing an accuracy curve. $\alpha$ is a
per-dataset significance level (set once, reported in the run header), not a tuning
knob: lower $\alpha\Rightarrow$ higher $\tau\Rightarrow$ fewer significant features.

| symbol | code | default | role |
|---|---|---|---|
| $\alpha$ | `ALPHA` (per-dataset; `--alpha`) | $0.01$ | scale-axis confidence level |
| $B$ | `N_BOOT` | $100$ | $\tau$-band replicates (precision of $\tau$ only) |
| — | `MAX_DIMENSION` | $2$ | VR expansion degree |
| $\rho$ | `TAU_FLOOR_FRAC` | $10^{-2}$ | degenerate-layer floor (fraction of diameter) |
| $R,\gamma,a_{\min}$ | `CONV_N_RESAMPLE`, `CONV_SUBSAMPLE_FRAC`, `AGREEMENT_MIN` | $50,\,0.8,\,0.8$ | **only** for the optional `significance=True` path (Remark 2) |

**Remark 4.** $B$ controls only the precision of $\tau$; robust quantities ($d^\*$,
well-separated features) are invariant to it. Per-dataset $\alpha$ matters: cardio
uses a low $\alpha$ (fewer noise $H_0$); COIL's loops are weak, so a higher
$\alpha\,(0.05)$ captures more of them.

---

## 13. Caveats / threats to validity

1. **First-appearance filtration** (C2) over non-nested pullback complexes — a stated
   modelling choice that makes the union monotone.
2. **High-dimensional concentration** — mitigated, not eliminated, by Stage 0; report
   raw-vs-PCA.
3. **$\varepsilon$ reuse** — $\varepsilon_i$ chosen on $X_i$ alone is reused in the
   pullback; a feature significant in layer persistence need not surface in the tower.
4. **Saturated late layers** give $\tau\to0$ (flagged degenerate); the $H_0$ tail cap
   prevents $\varepsilon$ landing in the merged tail. Fallback layers are honest "no
   significant structure", not a bug.
5. **Compact networks** (e.g. ResNet) show $d^\*\approx L-1$ with a short inert tail:
   the honest reading is "uses nearly full depth"; pruning is **not** claimed there.
6. **Run-to-run variation.** Trained-from-scratch nets can shift $d^\*$ by a layer
   across seeds when late layers sit at the significance boundary; report the range or
   fix the seed.

---

## 14. Function map (`ltep` package)

| concept | Construction | location |
|---|---|---|
| relevant dimension | C15 | `ltep.pipeline.relevant_dimension` |
| preprocessing | C3–C5 | `ltep.pipeline.preprocess_latents`; `ltep.datasets.coil100.diameter_normalize`; `ltep.runtime` |
| $\tau$ band + two $\varepsilon$ | C6–C10 | `ltep.pipeline.select_epsilon` (per-layer via `ltep.metrics`) → `epsilons_H0`, `epsilons_H1` |
| manual override | C12 | `ltep.pipeline.parse_manual_epsilons` |
| per-dataset $\alpha$ | §12 | `ltep.pipeline.set_alpha` |
| pullback tower + barcode | C2 | `ltep.pipeline.mlp_persistence` |
| barcode reading + $d^\*$ | C13–C14 | `ltep.pipeline.convergence_depth(significance=False)` |
| signal / simplification / carrier | C16 | `ltep.pipeline.{signal,simplification,carrier}_dimension` |
| cross-check / output anomaly | C17 / C18 | `ltep.pipeline.{cross_check_bottleneck, output_loop_anomaly}` |
| report | — | `ltep.pipeline.pretty_print`; `ltep.output` (per-run folder + log + params) |
| visual checks | — | `ltep.plots.{plot_layer_persistence, plot_mlp_persistence, plot_betti0_diagnostic}` |
| pruning validation | C19 | `experiments.cardio.validate_pruning` |

---

### References
- B. T. Fasy, F. Lecci, A. Rinaldo, L. Wasserman, S. Balakrishnan, A. Singh,
  *Confidence sets for persistence diagrams*, Ann. Statist. 42(6), 2014.
- Y. Hiraoka, Y. Imoto, S. Kanazawa, E. Liu, *Curse of dimensionality on persistence
  diagrams*.
- E. Paluzo-Hidalgo, *Latent Space Topology Evolution in Multilayer Perceptrons*,
  arXiv:2506.01569, 2025.
