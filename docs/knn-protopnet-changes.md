# kNN ProtoPNet: changes and how they relate to the paper

This note records each change made while adding a k-nearest-neighbor version of ProtoPNet to this repository. The source paper is Tang et al., *This EEG Looks Like These EEGs: Interpretable Interictal Epileptiform Discharge Detection With ProtoEEG-kNN* (MICCAI 2025, arXiv:2510.20846).

The paper’s prediction rule is in Section 3.0.5. After a ProtoPNet has been trained, every training sample becomes a prototype. The prediction for a new sample is the average label of its `k` nearest training samples. In the paper, `k = 10`.

Two prediction methods are in the code. Section 4 is the EEG replacement: after training, every training image becomes a prototype. Section 6 is the image method that `train-protopnet` actually runs: the learned prototypes stay, and the test score is a vote of the nearest training profiles. The sections in between record setup that is not part of either paper, and why it was still required.

## 1. `KNNPrototypePredictionHead`

**File:** `protopnet/prediction_heads.py`

**What changed.** A new prediction head, `KNNPrototypePredictionHead`, was added next to the existing linear head. It does not replace `LinearClassPrototypePredictionHead`. Training can still use the linear head. This class is the head that will be used after the neighbor replacement step.

**How this relates to the paper.** Section 3.0.5 defines the final predictor as the average label of the `k` nearest training samples. This class is that predictor. It receives one similarity score per prototype, selects the `k_neighbors` largest scores (default 10, the same `k` as the paper), and combines the labels of those prototypes.

**What the code does, in order.**

1. `prototype_labels` stores one class id per prototype, in the same order as the prototype tensors. It is saved with the model and is not trained. In the paper, these labels are the labels of the training samples that replaced the learned prototypes.
2. The activation tensor has a score at each spatial location inside each prototype. The head flattens that map and averages the strongest locations (`k_for_topk`). The result is one similarity score per prototype. The paper’s EEG model does not have this spatial map, so this pooling is not from the paper. It is required because this repository’s prototype layer returns a spatial activation grid, and the existing linear head already reduces that grid the same way. Matching that reduction lets the new head sit in the same forward path.
3. The head takes the `k_neighbors` prototypes with the highest similarity. If fewer prototypes exist than `k_neighbors`, it uses all of them. That case matters for small tests. The paper always has a full training set, so it does not discuss this case.
4. Each selected prototype votes for its own class. The output `logits` has one value per class: the fraction of the selected neighbors that belong to that class. The trainer already reads a tensor named `logits` and predicts the class with the largest value, so the head returns that name.

**Where this is not the same as the paper, and why.** The paper averages the neighbor labels as ordinary numbers. That is valid there because each EEG label is a fraction from 0 to 1: eight annotators voted yes or no, and the label is the fraction of yes votes (Section 3.0.1). CUB-200 labels are species ids from 0 to 199. Those ids are names, not scores. Averaging species 0 and species 2 would produce 1, which is a different species, not a vote. The head therefore gives each neighbor one vote for its species and averages those vote vectors. Two neighbors of species 0 produce `[1, 0]`. One neighbor of species 0 and one of species 1 produce `[0.5, 0.5]`. The winning species is the class with the largest fraction, which is the same decision the paper makes when several neighbors share a label.

**What this change does not do.** The paper’s Section 3.0.5 also replaces the learned prototypes with one prototype per training sample. This head only consumes those prototypes and their labels. It does not build them. That replacement is Section 4 below. The paper’s EEG-specific pieces are also not in this class: the SpikeNet backbone, channel weights, and the range, variance, and FFT similarity terms (Sections 3.0.2 and 3.0.3). Those terms exist because the paper compares EEG waveforms. This project compares bird photographs with the similarity function ProtoPNet already uses in this repository.

## 2. Comments in the new head

**File:** `protopnet/prediction_heads.py`, on `KNNPrototypePredictionHead` only.

**What changed.** Comments and a class docstring were added to the new head. No other file was commented, and the math was not changed.

**How this relates to the paper.** The comments name the paper’s vote (`k_neighbors`) and separate it from `k_for_topk`, which is this repository’s spatial pooling and is easy to confuse with the paper’s `k`.

**Why this was necessary even though the paper does not ask for comments.** The two `k` values do different jobs. Leaving them unnamed would make the bird adaptation look like a different algorithm from Section 3.0.5.

## 3. Setup that is not a change to the model

These steps were done on the URI Unity cluster. They are not part of the paper, and they are not edits to the model code. Each one was required before the head can be trained or evaluated on birds.

**Working copy on Unity.** The class cluster is the machine that can train on CUB-200. The repository used there is `/work/pi_csc592s3_uri_edu/Osman_uri/protopnext-classF2026`, with the virtual environment already activated. The laptop copy is where the code was written. Unity receives it with `git pull` after this commit is pushed. Until that pull, the cluster still has the original class repository.

**Bird data in this project directory.** A link named `CUB_200_2011` points at the shared class copy `/work/pi_csc592s3_uri_edu/datasets/CUB_200_2011`. The loader in `protopnet/datasets/cub200.py` looks for that folder name, which is why the link lives inside the project directory. The shared folder already contains `train`, `validation`, and `test`, so the split command was not run. The images must not be committed. The paper’s dataset is a private set of 16,499 EEG windows, which is not available here. CUB-200 is the dataset this repository already trains on, so the same kNN rule can be run on bird species instead.

## 4. kNN replacement phase

**File:** `protopnet/models/knn_protopnet.py`

**What changed.** `KNNReplacePhase` and `replace_learned_prototypes_with_neighbors` were added. `train-protopnet` does not call them. A bird run still trains with the linear head, then scores the profile vote in Section 6. Calling this replacement after that training would delete the learned prototypes, so it is left unattached.

**How this relates to the paper.** Section 3.0.5 runs after training. It throws away the learned prototypes, stores the embedding of every training sample as a prototype, and replaces the linear head with the kNN rule. `KNNReplacePhase.run_step` is that step. It reads each training image through `model.embed`, stacks those tensors into `prototype_tensors`, and installs `KNNPrototypePredictionHead` with those images' class ids. `k_neighbors` defaults to 10, the paper's `k`.

**Why the prototype parameter is replaced.** A trained CUB model has 10 prototypes per species, 2,000 in total, each a small spatial patch. The training set has about 6,000 images, and each embedding is the full feature map of that image. Copying into the old parameter cannot change its shape. The new parameter is not trained, because the paper stops gradient updates at this step.

**Why class identity is rewritten.** The prototype layer still stores a matrix saying which class each prototype belongs to. That matrix was sized for the 2,000 learned prototypes. It is rebuilt as one row per training image so the recorded species matches the label the kNN head votes with.

**Why the old prototype metadata is cleared.** Metadata from the learned prototypes describes patches chosen during projection. Those patches are gone. The new list only marks each row as a training sample. The class id used for the vote lives on the kNN head.

**What this phase does not do.** It is not attached to `train-protopnet`. The Unity bird run scores the profile vote instead, because that vote keeps the learned prototypes. The paper's EEG similarity, channel weights, and soft 0-to-1 labels are still not used. Losses that assume a linear class-connection layer, including the L1 penalty on that layer, do not apply after this swap. The paper does not keep training those terms either.

## 5. Pytest for the vote

**File:** `test/models/test_knn_protopnet.py`

**What changed.** Ten tests were added. They use small tensors. They do not read CUB-200, and they do not train a network. The first tests cover the EEG vote and the replacement. The last three cover the profile vote in Section 6: the closer profile wins, five neighbors produce a majority, and fitting the vote does not change the learned prototypes.

**How this relates to the paper.** Section 3.0.5 says the prediction is the average label of the nearest training samples. On birds, that average is a class vote. The tests check the cases that define the vote:

- Two nearest prototypes of species 0 produce logits `[1, 0]`. The predicted species is 0.
- One neighbor from species 0 and one from species 1 produce `[0.5, 0.5]`.
- Neighbors from species 0 and species 2 produce `[0.5, 0, 0.5]`. Species 1 stays at zero. This is the case that would be wrong if the species ids were averaged as numbers, which is what the paper can do only because its EEG labels are fractions from 0 to 1.
- When `k` is larger than the number of prototypes, every prototype votes. A tiny test has fewer than the paper's `k` of 10, so this guard is required there.

**Why the replacement is tested with a fake model.** One more test passes three fake embeddings and the species ids `[0, 1, 0]` through `replace_learned_prototypes_with_neighbors`. It checks that the learned prototypes are discarded, that three new prototypes are stored in the same order as those images, and that the kNN head receives those species ids. Building a real ProtoPNet or reading CUB is not required to prove that the replacement follows Section 3.0.5.

## 6. Profile kNN for images

**Files:** `protopnet/models/knn_protopnet.py`, `protopnet/train_protopnet.py`, `protopnet/run_knn_fashionmnist.py`

**What changed.** `PrototypeProfileKNN` keeps the learned prototypes. `prototype_profile` reduces each image to one similarity per prototype, using the strongest patch. `profile_neighbor_votes` predicts by the majority class among the `k` training images whose profiles are closest in squared Euclidean distance. `train-protopnet` does this after the normal CUB training finishes and prints `Profile kNN test accuracy`. The default `k` is 5. `--profile-knn-k 0` skips the vote. The Fashion-MNIST script is a small local check of the same vote: 2 prototypes per class, 100 images per class, 15 epochs, batch size 32, and `k = 5`.

**How this relates to the papers.** This is ProtoKNN (Ukai et al., ICLR 2023, Section 2.3), not the EEG replacement. ProtoKNN was measured on CUB-200. With ResNet-50, `k = 5` is their best reported setting at 87.2%. The EEG replacement remains in the file for the waveform method, and the image runner does not use it.

**Why the learned prototypes stay.** On photos, a prototype is a part. Deleting those parts and storing a full feature map per training image is the EEG paper's answer to spike diversity. Profile kNN keeps the part explanation and compares short vectors instead of thousands of feature maps.

**What this does not include.** The ICLR training losses (metric-learning classification, affiliation cluster loss, and proxy-anchor auxiliary loss) are not implemented. `train-protopnet` still uses this repository's ProtoPNet losses, and the Fashion-MNIST script trains with cross-entropy only. The 87.2% CUB figure comes from those ICLR losses plus this vote, so neither run reproduces that number.
