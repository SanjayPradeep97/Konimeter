# Attribution for the demo images and masks

The 60 micrographs in `demo/images/` are redistributed from

> Wang, Chen (National Institute for Occupational Safety and Health), 2024.
> *Dataset of TEM Images for Carbon Nanomaterial Classification*. Harvard
> Dataverse, V1, <https://doi.org/10.7910/DVN/5O0SF7>, licensed
> **CC BY-NC 4.0** (<https://creativecommons.org/licenses/by-nc/4.0/>).

They are unmodified in content: each file is the original 8-bit greyscale
image converted losslessly from TIFF to PNG (`demo/select_demo_images.py`
chose them, `demo_images.csv` lists the original file names). They are
included here, under the same licence, so that the pipeline can be tried
without downloading the full 5,323-image dataset. Non-commercial use only;
credit the dataset above if you reuse them.

The 60 masks in `demo/masks/` were drawn by the authors of this repository
with the segmentation tool in it, as part of the 1,785-mask set behind the
paper's classification results. They are derived from the CC BY-NC images and
are provided under the same terms.

The rest of the dataset, and the remaining 1,725 masks, are not in this
repository; see the top-level README, "Data".
