# Computer-Vision-SfM
An attempt at creating sparse 3D reconstructions using structure from motion pipeline and OpenCV. This is one component of a final project that tested multiple 3D reconsturction methods to each other. The report for that project can be found here as well

## How to Use
Run SfM.py using multi view stereo data set of your choice. Currently in data are the Dino Ring and Temple Ring datasets from [Middlebury](https://vision.middlebury.edu/mview/data/)

You can also use your own data and toggle by editing line 11. It currently loads data using functions in helper_functions.py

## Notes
Currently bundle adjustment does not work properly. Also if you plan to use your own data, it may take quite of lot of tuning. SfM is also very sensitive to the data so if the images are not of "good" enough quality and lack the necessary overlap, the likelihood of failing increases dramatically.
