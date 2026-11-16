# Evaluation

How often the deployed detector is right, measured the way it is served: same decoding, same
preprocessing, same ONNX session. Everything here is reproducible from the scores committed
in [apps/inference/eval/results](../apps/inference/eval/results) with
`python -m eval.report`.

## Data

[OpenFakeTiny](https://huggingface.co/datasets/ComplexDataLab/OpenFakeTiny) at revision
`9a7f5a7`, three splits of 1,250 images each:

| Split      | Real images                  | AI-generated images                                        | Used for                 |
| ---------- | ---------------------------- | ---------------------------------------------------------- | ------------------------ |
| validation | 614, from LAION and Pexels   | 636, from 66 generators released 2022 to 2026              | fitting calibration only |
| test       | 641, from ImageNet and DOCCI | 609, from 16 generators, 567 of them released 2025 or 2026 | reporting                |
| reddit     | 230 posted to Reddit         | 1,020 posted to Reddit, 912 of them video frames           | reporting                |

The two core splits are different mixes: validation has older generators and photos from
LAION and Pexels, test has only recent generators and photos from ImageNet and DOCCI. So the
numbers below already include a shift between the data the thresholds were fitted on and the
data they are judged on. The reddit split is labelled by the dataset authors and is the
closest thing here to what people actually check.

"Recompressed" means each test image downscaled to at most 1080 px on the long side and saved
as JPEG at quality 75, roughly what a chat app does to a forwarded image.

An earlier run refused twelve real photos of 42 to 64 megapixels because the service checked
its pixel limit before the reduced-size JPEG decode. That was a bug in the service, now fixed,
and every image in all three splits is scored.

## Method

The detector outputs one logit. On validation only:

1. Platt scaling, a one-feature logistic regression, maps the logit to a probability:
   `p = sigmoid(0.4065 * logit + 1.5623)`.
2. Two cut-offs are chosen so that each confident verdict is right 90% of the time on
   validation: `p >= 0.541` is "likely AI", `p <= 0.212` is "no sign of AI", anything in
   between is "can't tell".

Those four numbers are then frozen and applied to test and reddit.

## Results

### Ranking and calibration

| Set                         | Images | AUROC | ECE raw | ECE calibrated |
| --------------------------- | ------ | ----- | ------- | -------------- |
| OpenFake test               | 1250   | 0.916 | 0.326   | 0.165          |
| OpenFake test, recompressed | 1250   | 0.851 | 0.371   | 0.187          |
| Reddit, in the wild         | 1250   | 0.747 | 0.620   | 0.364          |

AUROC says how well the scores separate AI from real images regardless of any cut-off.
ECE is the average gap between the stated probability and how often images at that
probability are actually AI, so lower is better.

### Verdicts

| Set                         | Answered | Likely AI right | No sign right | AI caught | Real cleared |
| --------------------------- | -------- | --------------- | ------------- | --------- | ------------ |
| OpenFake test               | 82.5%    | 96.1%           | 80.9%         | 52.2%     | 88.3%        |
| OpenFake test, recompressed | 77.1%    | 93.3%           | 75.8%         | 40.9%     | 82.4%        |
| Reddit, in the wild         | 68.9%    | 94.8%           | 33.9%         | 44.7%     | 56.1%        |

"Answered" is the share of images that got a confident verdict instead of "can't tell".
"Likely AI right" and "No sign right" are the precision of each verdict. "AI caught" is the
share of AI images called likely AI, "Real cleared" the share of real images called no sign
of AI.

### Per source, test split

| Source          | Label | Images | Right verdict | Wrong verdict |
| --------------- | ----- | ------ | ------------- | ------------- |
| imagenet        | real  | 444    | 95.3%         | 0.7%          |
| docci           | real  | 197    | 72.6%         | 5.1%          |
| z-image-turbo   | AI    | 187    | 51.9%         | 14.4%         |
| flux.2-klein-9b | AI    | 118    | 6.8%          | 69.5%         |
| illustrious     | AI    | 100    | 88.0%         | 2.0%          |
| gpt-image-1.5   | AI    | 65     | 36.9%         | 23.1%         |
| midjourney-7    | AI    | 44     | 84.1%         | 4.5%          |
| veo-3           | AI    | 28     | 53.6%         | 10.7%         |
| wan-video-2.5   | AI    | 16     | 81.2%         | 0.0%          |
| recraft-v3      | AI    | 14     | 50.0%         | 0.0%          |
| lumina-17-2-25  | AI    | 6      | 100.0%        | 0.0%          |
| aurora-20-1-25  | AI    | 6      | 83.3%         | 0.0%          |
| seedream-v5.0   | AI    | 5      | 40.0%         | 20.0%         |
| halfmoon-4-4-25 | AI    | 5      | 80.0%         | 0.0%          |
| nano-banana-pro | AI    | 4      | 50.0%         | 25.0%         |
| frames-23-1-25  | AI    | 4      | 75.0%         | 25.0%         |
| ideogram-2.0    | AI    | 4      | 100.0%        | 0.0%          |
| recraft-v2      | AI    | 3      | 100.0%        | 0.0%          |

### Against the detector this project used before

[Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector), a Swin
classifier fine-tuned on pairs of Wikimedia photos and SDXL images, each model at its own
uncalibrated 0.5 cut-off:

| Set                         | sdxl-detector AUROC | accuracy | communityforensics-vit AUROC | accuracy |
| --------------------------- | ------------------- | -------- | ---------------------------- | -------- |
| OpenFake test               | 0.682               | 65.8%    | 0.916                        | 66.6%    |
| OpenFake test, recompressed | 0.642               | 61.8%    | 0.851                        | 61.4%    |
| Reddit, in the wild         | 0.674               | 69.4%    | 0.747                        | 36.3%    |

### Reduced-size JPEG decoding

The service asks libjpeg to decode large JPEGs at 1/2, 1/4 or 1/8 scale, never below the
model's resize target. On the first 400 test images:

| Images | AUROC full decode | AUROC draft | Mean abs Δp | Max abs Δp |
| ------ | ----------------- | ----------- | ----------- | ---------- |
| 400    | 0.9201            | 0.9233      | 0.0074      | 0.2205     |

## What this means

**It ranks recent generators well, but its raw scores are not probabilities.** AUROC on test
is 0.916, against 0.682 for the detector this project started with. At its own 0.5 cut-off,
though, it gets only 66.6% of test images right, because it puts most generated images well
below 0.5 while still scoring them above real photos. Calibration halves the calibration error
(0.326 to 0.165), and the bands do the rest.

**"Likely AI" can be trusted. "No sign of AI" cannot.** When it says likely AI it is right
96.1% of the time on test, 93.3% after recompression and 94.8% on Reddit. But it calls only
52.2% of the test split's AI images likely AI, and on Reddit, where most images are generated,
two out of three "no sign of AI" answers are wrong. A clean result means the detector found
nothing, not that there is nothing to find. The page says this right below the result.

**There are clear blind spots.** Flux.2 klein is the worst: of 118 images, 6.8% are caught and
69.5% are called real. GPT Image 1.5 is caught 36.9% of the time and Z-Image Turbo 51.9%. The
detector does well on Illustrious (88.0%) and Midjourney 7 (84.1%).

**Real photos are not equally easy.** 95.3% of ImageNet photos are cleared against 72.6% of
DOCCI's. DOCCI's photos are newer and much larger (a median of 3.2 megapixels against 0.2),
and ImageNet may overlap with the detector's training data, so the ImageNet number is probably
the optimistic one.

**Recompression hurts.** After a chat-app style downscale and JPEG save, AUROC drops from 0.916
to 0.851 and the share of AI images caught from 52.2% to 40.9%. Images that have been forwarded
a few times are harder than originals.

**Reddit is much harder than the curated splits.** AUROC falls to 0.747, still above the old
detector's 0.674. At raw 0.5 the old detector looks more accurate there (69.4% against 36.3%),
but only because it calls two thirds of all images AI on a split that is four fifths AI.
The new detector's raw score calls 18.7% AI, which is why calibration matters there most.

**Decoding large JPEGs at reduced size costs nothing measurable.** AUROC is 0.9233 with it and
0.9201 without on the same 400 images. The mean change in probability is 0.0074, though one
image moved by 0.22.

## Reproducing

```sh
cd apps/inference
pip install -r eval/requirements.txt
python -m eval.score communityforensics-vit validation
python -m eval.score communityforensics-vit test
python -m eval.score communityforensics-vit test --recompress
python -m eval.score communityforensics-vit reddit
python -m eval.score sdxl-detector test
python -m eval.report
```

Scoring is resumable, and every run appends to `eval/results/<candidate>/<split>.jsonl`.

## Limits of this evaluation

- Per-generator rows with fewer than about 30 images are anecdotes, not measurements.
- Precision depends on how common AI images are in what gets checked. The test split is
  about half AI and reddit about four fifths, which is why the same detector looks so
  different on the two.
- The detector's training data is not fully listed. Some ImageNet photos may overlap with it,
  which would flatter the real-photo results on test. DOCCI, published in 2024, is less
  likely to.
- OpenFake is CC BY-NC 4.0. The scores are committed here, the images are not.
