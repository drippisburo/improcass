# Context for a new Claude chat: thermal hotspot classifier + report (NTU IE4476)

Paste this whole file at the start of a new chat. Also paste `writing-style.md` (my writing style) when asking for report text.

## Working rules (keep these)
- I write the code myself. Do NOT write or edit code unless I explicitly ask; explain, review, suggest checks instead. Code shown in chat is fine.
- Never touch `functions.cpp`, `preproc.cpp`, `functions.h`. Scratch files go in Claude's scratchpad, named `claude_*` if in the project.
- The professor explicitly allows Claude to write the report text. Write it in my style (writing-style.md) and follow `z_conventions.txt`.
- I guide the report: do one chapter at a time when I ask. Commit locally, but ASK before `git push` (pushing reaches Overleaf).
- Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>` (or whatever attribution the session specifies).

## Assignment
"Dimensionality Reduction for Classification" (20%), due 15 Nov 2026. Own program: PCA and/or LDA, classifier (Mahalanobis, k-NN), accuracy vs number of dimensions plot, proper preprocessing (alignment, normalisation), proper train/test split (all parameters from training data only). Report with figures/tables/references, code as appendix (official guide says 4-5 pages; length is my call).

Motivation (for the introduction, which I write myself): search and rescue in the Alps; a drone with a thermal camera finds warm spots, the program decides whether each warm spot is one person, a tight group of people (cluster), or something else (clutter).

## Paths
- Code project: `C:\Users\alber\OneDrive\Desktop\school\candcplusplus\cplpl\HIT-UAV-Infrared-Thermal-Dataset-main` (git remote github.com/drippisburo/improcass)
  - `functions.h/.cpp` (all functions), `preproc.cpp` (one setting), `pooledpreproc.cpp` (the sweep -> `results.csv`), `plotting.py` (4 figures from results.csv)
  - Data: `normal_xml/JPEGImages`, `normal_xml/Annotations` (Pascal VOC XML, standard boxes)
  - Build: `g++ -std=c++20 -O2 -I C:/msys64/ucrt64/include/eigen3 pooledpreproc.cpp functions.cpp -o pooledpreproc.exe; if ($?) { .\pooledpreproc.exe }` (about 2 min)
  - Python: `C:\Program Files\Python313\python.exe` (has matplotlib; the MSYS2 python does not)
  - Existing images in project root: `eigenvector_top_*_*.png` (eigen-patches), `lda_vector_1/2.png` (LDA filters), `outputmask.png`, `outputblobs.png`
- Report repo: `C:\Users\alber\OneDrive\Desktop\school\candcplusplus\cplpl\Report_improc` (github.com/drippisburo/Report_improc, linked to Overleaf)
  - Overleaf does not pull automatically: Menu -> GitHub -> Pull GitHub changes into Overleaf, then Recompile. No TeX installed locally.

## Pipeline as implemented (numbers to use in the report)
1. Frames: HIT-UAV night images only (1981 of 2898), camera angles 70/80/90 deg, all altitudes 60-130 m (step 10) -> 762 frames, 3561 person boxes. 24 altitude-angle groups.
2. Detection: global threshold pixel >= 1.52 x frame mean; blob grouping of hot pixels within radius r = 6; keep blobs with area 12-500 px. Settings chosen by hand.
3. Labels: count person-box centres inside the blob bounding box: 0 clutter, 1 person, >= 2 cluster. Other classes (Car, Bicycle, OtherVehicle, DontCare) unused.
4. Patches: square 38 x 38 px (rad 16 + margin 3, size = 2(rad+margin)) centred on blob centroid, sampled bilinearly with altitude scale k = 100/alt (reference 100 m), border pixels repeated at frame edge.
5. Rotation: angle from intensity-weighted second moments (in-circle pixels), theta = 0.5 atan2(2 mu11, mu20 - mu02); rotate (inverse map, bilinear) so the long axis is vertical. Circular mask.
6. Normalisation: in-circle pixels z-scored per patch (zero mean, unit variance), outside circle 0.
7. PCA: mean + eigenvectors from TRAINING patches only (incl. mirrors), solved once per fold at dmax = 150 (Eigen returns ascending: top d = rightCols(d)).
8. PCA -> LDA: LDA on the d PCA dims, at most 2 directions (3 classes), needs d >= 2.
9. Classifiers: Mahalanobis with per-class mean + covariance, score = -1/2 d_M^2 - 1/2 log det, no prior; small eigenvalues adjusted with notsmall = 20 (only acts when d > 20). k-NN: squared Euclidean, majority vote, ties to the closest neighbour.
10. Evaluation: grouped 5-fold CV; the 24 groups assigned greedily to 5 buckets (group with most clusters first, into the bucket with fewest clusters). Left-right mirrored copies of TRAINING patches only. Metric: balanced accuracy (mean per-class recall), pooled confusion matrix over folds. Official HIT-UAV split NOT used (103 of 105 flight groups in both train and test).

Sweep: d in {1,2,3,5,10,20,40,80,150} with k = 5; k in {1,3,5,9,15} at d = 40. 42 settings. results.csv columns: `LDAPCA,classifier,d,k,BA,recall_clutter,recall_person,recall_cluster`.

## Data numbers
| Angle | Frames | Clutter | Person | Cluster |
|---|---|---|---|---|
| 70 | 301 | 3027 | 648 | 74 |
| 80 | 274 | 2304 | 1672 | 246 |
| 90 | 187 | 1680 | 339 | 32 |
| Total | 762 | 7011 | 2659 | 352 |
Imbalance about 20 : 7.5 : 1. Group 60 m / 80 deg alone has 190 clusters and 1199 persons. Groups 60 m / 90 deg and 130 m / 90 deg have no person boxes at all.

## Results
| Method | Best BA | At |
|---|---|---|
| PCA + Mahalanobis | 78.9% | d = 10 (plateau 10-80, collapses to 63.1% at d = 150) |
| PCA + k-NN | 73.2% | d = 40, k = 1 (73.1% at d = 20 with k = 5) |
| PCA -> LDA + Mahalanobis | 76.3% | d = 20 |
| PCA -> LDA + k-NN | 62.2% | d = 40, k = 15 |
- Effect of k (d = 40): PCA BA falls with k (73.2 -> 71.3); LDA BA rises with k (58.6 -> 62.2); for LDA k = 1 and k = 3 are identical.
- Why k=1 = k=3 in LDA: the training set is mirror-symmetric, so LDA directions are symmetric and a patch and its mirror project to the same point (difference ~1e-6): every training point is a duplicate pair.
- k-NN carries the class prior (under-predicts clusters); Mahalanobis has no prior, which balanced accuracy rewards.
- At d = 150, per-class covariances from few cluster vectors are unreliable; almost everything goes to "person".
- At d = 2, PCA and PCA->LDA Mahalanobis give the same result (affine invariance).

## Report: structure and status
Template: TU Delft class (`tudelft-report.cls`), biblatex style `aiaa`. Title "Dimensionality Reduction for Thermal Hotspot Classification", subtitle "PCA and LDA on UAV infrared imagery for search and rescue", author "Alberto Rigaudo N2602293K".
Conventions (`z_conventions.txt`): no personal pronouns, British English, capitalised titles, `\Cref`, labels lowercase with underscores, the fixed table template (booktabs, `[H]`), captions without full stop.
Bibliography (`bibliography.bib`): `suo2023hituav` (dataset, CC BY 4.0), `turk1991eigenfaces`, `belhumeur1997fisherfaces`.

| File | Sections (labels) | Status |
|---|---|---|
| `mainmatter/introduction.tex` | - | I write it myself |
| `mainmatter/chapters/data.tex` | sec:dataset, sec:frame_selection, sec:labelling (eq:labelling), sec:class_distribution (tab:class_distribution, fig:person_boxes = `figures/data/person_boxes.png`) | DONE (commit e9d0fad, pushed) |
| `mainmatter/chapters/method.tex` | sec:pipeline, sec:detection, sec:patches, sec:normalisation, sec:pca, sec:lda, sec:classifiers (sec:mahalanobis, sec:knn), sec:evaluation | skeleton with comments |
| `mainmatter/chapters/results.tex` | sec:accuracy_vs_d, sec:per_class_recall, sec:effect_of_k, sec:best_configurations | skeleton |
| `mainmatter/chapters/discussion.tex` | sec:pca_vs_lda, sec:class_imbalance, sec:high_dimension, sec:mirror_augmentation, sec:limitations | skeleton (ends with my old `%%` question line, delete it) |
| `mainmatter/conclusion.tex` | - | skeleton |
| Appendix "Source Code" (`appendix/`) | functions.h, functions.cpp, pooledpreproc.cpp, plotting.py | copies; re-copy final code before submission |

Figures available: `figures/results/fig1_ba_vs_d.png` (BA vs d, 4 methods), `fig2_recall_vs_d.png` (per-class recall, PCA+Mahalanobis), `fig3_cluster_recall.png` (cluster recall, Mahalanobis vs k-NN), `fig4_ba_vs_k.png` (BA vs k at d = 40).

### Plan, in order (I decide when each starts)
1. Method chapter: pipeline figure (frame -> mask -> blobs -> patches -> rotated patches; could use outputmask/outputblobs + example patches per class), detection, patches + rotation (equation for theta), normalisation, PCA (equation, eigen-patch figure), LDA (S_B w = lambda S_W w, LDA filter figure), Mahalanobis score equation, k-NN, evaluation protocol.
2. Results chapter: the 4 figures, best-configuration table (template), pooled confusion matrix of PCA + Mahalanobis d = 10 (needs a quick run to print it).
3. Discussion: PCA vs LDA, class imbalance and the classifiers, high-dimension covariance, mirror augmentation in LDA space, limitations (hand-set detector thresholds not fitted per fold, accuracy conditional on detection, cold vehicles ignored, neighbouring frames repeat the same people).
4. Conclusion (recap, recommendation PCA + Mahalanobis d about 10, next steps: balanced training for k-NN, learned detector thresholds, more camera angles), final modest one-sentence paragraph.
5. Front matter (summary/preface/nomenclature: keep or remove), cover image (replace `figures/cover_placeholder.jpeg` with a thermal frame), re-copy appendix code, compile in Overleaf and fix errors, spellcheck.
