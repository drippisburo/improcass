# Project context: thermal hotspot classifier (NTU assignment)

## Working rules for Claude
- I write the code myself. Do NOT write or edit code unless I explicitly ask.
  Explain, review my code, suggest checks, and answer questions instead.
- Language: C++ (g++ 16, MSYS2, Windows). No CMake; single g++ command.
- Libraries: stb_image.h / stb_image_write.h (already in albwork/), Eigen later for PCA/LDA.
- My work so far is in albwork/ (explore.cpp, preproc.cpp).

## The assignment (fixed)
"Dimensionality Reduction for Classification" (20%), due 15 Nov 2026.
- My own program: dimensionality reduction (PCA, LDA, or both), classification
  (minimum Mahalanobis distance, linear, or nearest neighbour), accuracy/error rate.
- Plot accuracy vs. number of reduced dimensions.
- Proper preprocessing (alignment, normalisation).
- Proper train/test split: ALL parameters (means, projections, covariances, detector
  thresholds) come from training data only; test data only gives the final accuracy.
- Report 4-5 pages incl. figures/tables/references, code as appendix.

## Motivation
Search and rescue: I'm a lifelong skier; people lost in the Alps are hard to see.
A drone with a thermal camera finds warm spots; my program decides whether each
warm spot is one person, a tight group of people, or something else.
Limitation to state: thermal sees people on the surface / partly buried, not deeply buried.

## Data: HIT-UAV (thermal drone dataset, CC BY 4.0)
Cite: Suo et al., "HIT-UAV: A high-altitude infrared thermal dataset for UAV-based
object detection", Scientific Data 10, 227 (2023). Name the licence, say I cropped/normalised.
- Night split (what I use): split_daynight/night/JPEGImages (1,981 grayscale JPGs, 640x512)
  and split_daynight/night/Annotations (Pascal VOC XML, human-drawn boxes).
- Filename: daynight_altitude_angle_0_frame.jpg, e.g. 1_60_90_0_00009.jpg
  (1 = night, altitude 60-130 m, camera angle 30-90 deg).
- People are white-hot. At night cars/bicycles usually look COLDER than the ground,
  so a warm-spot detector mostly ignores them (fine for SAR; report it).
- No "cluster" label exists: people are boxed one by one. I derive clusters.
- Night counts per angle (touching = person boxes overlap or gap <= 1 px):

  | angle | images | persons | touching pairs |
  |---|---|---|---|
  | 80 | 274 | 2278 | 304 |
  | 90 | 187 | 435 | 72 |
  | all 30-90 | 1981 | 9573 | 1746 |

  Touching boxes != merged blobs; recount clusters from real blobs once the detector runs.
- DO NOT use the official ImageSets train/test txt: 103 of 105 flight groups are in both.
  Split by flight group (first 4 filename fields) and use grouped 5-fold cross-validation.
  Neighbouring frames repeat the same people, so unique clusters < counted clusters.

## Current strategy
### 1. Frame selection
Night images with camera angle 80 or 90 deg only (near top-down; tunable, e.g. 70-90 later).
Top-down: a person is a roughly round blob; two touching people form a rectangle in any orientation.

### 2. Hotspot detection (per frame)
1. Load as 1-channel float.
2. Background B = box blur, radius r ~15, via integral image (cost independent of r;
   at borders average only the in-image part).
3. D = I - B, negatives clamped to 0 (big uniform hot roofs vanish; only rims remain).
4. Mask = D > mean(D) + k*std(D) (population std), start k = 3.
5. Connected components, 8-connectivity, flood fill with explicit stack (no recursion);
   each pixel visited once. Per blob: area, bbox, centroid, peak D, fill ratio, elongation.
6. Altitude-specific blob filter (limits from TRAINING boxes per altitude):
   noise (< 1 person) -> drop; 1-person size -> keep; merged 2-3 people -> keep;
   too big / long-thin / outline -> drop (roofs, edges). Filter BEFORE anything else.
7. Tune r, k and the limits on training groups: maximise person recall (aim >= 90%).

### 3. Classes and automatic labels
One candidate = one blob. Count person-box CENTRES inside the blob:
0 -> clutter, 1 -> person, >= 2 -> cluster. Skip blobs overlapping DontCare boxes.
Cluster = people right in front of each other, that is why the blob will be as large as the diameter of two people plus some margin for the space bwtween them.
Close-but-separate people are
separate blobs -> each is a "person". Labels depend on detector settings: fix them first.

### 4. Patch extraction (rotation + altitude scaling + crop in one step)
- Reference resolution = 60 m: patch size N = S(60) (~38 px).
- S(alt) = 2*d(alt) + 0.5*d(alt), d(alt) = median person diameter at that altitude
  from TRAINING boxes (use a table, not a 1/altitude formula; real sizes don't follow it).
- Scale s = S(alt)/N (s = 1 at 60 m; s < 1 at higher altitude = upsampling, adds no detail).
- theta = long-axis angle from intensity-weighted second moments of the blob:
  theta = 0.5*atan2(2*mu11, mu20 - mu02). Rotate so the long axis is VERTICAL
  (check on a horizontal pair: it must come out vertical). 180 deg ambiguity is fine.
- Sample each patch pixel (u,v) (centred coords) straight from the frame, bilinear:
  x = cx + s*(cos t*u - sin t*v), y = cy + s*(sin t*u + cos t*v), with t chosen so the
  long axis maps to vertical. No missing corners because the frame has real pixels.
  Blobs too near the frame edge: pad with B or skip.
- Circular mask radius N/2: use only in-circle pixels as the vector (~pi/4*N^2 values).
  Makes every patch cover the same area at any rotation, so ALL blobs can be rotated
  (round single people look the same at any angle).
- Normalise each patch (in-circle pixels) to zero mean, unit variance.

### 5. Training set handling
Balance classes (subsample person/clutter), flip augmentation on TRAIN only.
Report balanced accuracy + confusion matrix.

### 6. Dimensionality reduction
- Fit PCA on patch vectors, not whole images: ONE PCA on all training patches pooled
  (all 3 classes, labels not used), refit inside each fold. No per-class PCA: classes
  need one shared space to be compared. Labels only come in at LDA and the classifiers.
- PCA: mean patch, covariance, eigenvectors sorted DESCENDING (many solvers return
  ascending), z = W_d^T (x - mu). d = 1..~200. Eigen-patches = template images.
- PCA -> LDA (Fisherfaces): PCA to m dims first (m <= N_total - C so S_W is invertible;
  choose m ~10-200 on training folds), then LDA: S_B w = lambda S_W w, at most C-1 = 2 dims.
  LDA directions map back to pixel filters: w_pixels = W_PCA * v.

### 7. Classifiers (in the reduced space)
- 1-NN / k-NN, Euclidean (non-linear, piecewise-linear boundaries).
- Minimum Mahalanobis with ONE pooled covariance (linear boundaries; per-class
  covariance would need more samples per class than dims).

## Status (updated 5 Nov 2026)
### Done in albwork/preproc.cpp + functions.cpp (one image at a time)
- Patches: one square per blob centred on the centroid, size = 2*(rad + margin), now
  rad 26, margin 3 -> 58 px (plan says ~38). Edge patches are padded with the mean of
  their real pixels (nicepatches), so every patch is full size.
- Angle: patchangle = second moments of (pixel - in-circle mean), in-circle pixels only.
- Rotation: rotatepatch, inverse mapping + bilinear, t = theta - 90 deg, long axis ends up
  vertical. In-circle pixels only, margin must be >= 2 or it reads out of bounds.
  Checked on synthetic bars only (sizes 16 and 38): always vertical afterwards.
  Still to check by eye on real patches.

### Differs from the strategy above
- Detector is still the old one: global threshold pixel >= 1.52*frame mean, not D = I - B.
- findblob groups pixels within r = 6 (a 13x13 window), not 8-connectivity.
- Area limits are fixed at 4-500, not altitude-specific.
- Rotates the patch, not sampling from the frame. Switch when adding altitude scaling.

### Not done yet, in this order
1. Detector recall check (person-box centres inside kept blobs), then the D = I - B detector.
2. Labels from box centres (0 clutter, 1 person, >= 2 cluster), once the detector is fixed.
3. Finish patches: patch size, altitude scaling, in-circle vector, zero mean/unit variance.
4. Loop over all chosen night frames, save vector + label + flight group.
5. Grouped 5-fold split, then PCA, PCA -> LDA, classifiers.

### Useful facts
- All 1,981 night images are single-channel.
- Test frames with many blobs (old detector, r = 6, threshold 1.52, area 4-500):
  1_70_80_0_07746.jpg (80 deg, 52 blobs), 1_60_70_0_07491.jpg (70 deg, 46),
  1_120_90_0_03017.jpg (90 deg, 46). Blobs not checked against boxes, many may be clutter.
- Avoid 1_80_80_0_047xx and 1_80_90_0_04796 as tests: about 95% of hot pixels end up in
  too-big blobs, so their high counts mislead.

## Report figures / experiments
- Accuracy vs d: PCA+NN, PCA+Mahalanobis, PCA->LDA+NN, PCA->LDA+Mahalanobis,
  raw-pixel NN baseline, simple "blob area > X -> cluster" rule baseline.
- Confusion matrix (expect person <-> cluster errors), eigen-patches + LDA filters,
  LDA (z1, z2) scatter by class, explained variance vs d, detector recall table over r,k.
- Comparisons: with vs without rotation alignment; reference resolution 60 m vs 130 m
  (detail vs consistent sharpness); angle range 80-90 vs 70-90; k in k-NN;
  number of PCA dims m before LDA.
- Discuss: why clusters are hardest, pixel PCA is not shift/rotation invariant (hence
  alignment), cold vehicles, official split leakage, altitude sharpness differences.
