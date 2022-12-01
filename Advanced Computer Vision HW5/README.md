# Advanced Computer Vision HW5

In this section, there are two notebooks for two distict image processing tasks as described below.

## Template Matching (File: Template_Matching.ipynb)

In this notebook, some airplane templates (insider ***templates*** directory) are matched with an image of several airplanes in ***planes.png***. To this end, initially, **Canny edge detector** is employed. Furthermore,  a distance image is constrcuted that is convolved with each template and is further used for calculating **Chamfer distances**. Distances lower than a defined threshold correspond to a match. Results are further refined using **Non-Maximum Suppression** (NMS) technique.
 ## Point Cloud Alignment (File: Point_Cloud_Alignment.ipynb)
 In this part, point clouds in files ***pc1.npy*** and ***pc2.npy*** are aligned using ICP and PCA algorithms.
In the figures, red points demonstrate source data, blue ones demnsotrate target data, and green points are the source data aligned with the target data using the two approaches. As compared visually, ICP algorithm yields a better matching overall.



