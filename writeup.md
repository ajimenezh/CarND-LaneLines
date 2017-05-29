# **Finding Lane Lines on the Road** 


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

[masked_edges]: ./test_images_output/masked_edges.jpg "Masked Edges"

[example_1]: ./test_images_output/solidWhiteCurve.jpg "Example1"

[example_2]: ./test_images_output/solidYellowCurve2.jpg "Example2" 

[example_3]: ./test_images_output/res_105.jpg "Example3" 

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

In this project I've used two pipelines, one to find the lanes in the easy examples, and another one, with modifications in the pipeline and the draw_lines() function, to solve the challenge. i'll start describing the basic pipeline and then I'll explain the differences with the model that solves the challenge.

The first pipeline consists of 5 steps. First, I converted the images to grayscale, then I apply a Gaussian blur with a kernel size of 5 to the image. The third step is to find the edges of the image, for which Canny's algorithm is used with a low threshold of 80 and a high threshold of 160. After this, we apply a mask to the image to retain only the information relevant to finding the lanes. The mask is a polygon that hides everything outside of it, and keeps the data inside the polygon. For the lane finding algorithm, we are mostly interested in the road, so the chosen polygon is a trapezium which approximately covers the bottom image of the image, erasing a little of the left and right on top. The result so far can be seen in the next image, where we can already distinguish the lanes:

![alt text][masked_edges]

The final step is to find the lines with Hough's algorithm with suitable parameters. These parameters can be found by trying different configurations and using the one that seems to work best in most cases.
With this, we  can usually get the right and left lane, which we can identify by many methods, for example, using the slope of the lines. However, the method I used was to find the coordinate of the intersection of the line with the bottom of the image, and if this point is in the left side, the line belongs to the left lane, and similarly with the right lane. When we have done this, we have two sets of segments, but we want to find two lines. In order to find the lines, I've used the median of all the slopes of every segment that can be considered as right or left lane, and knowing the median and a point belonging to the line, we can draw the lane. The result can be seen in the following examples:

![alt text][example_1]

![alt text][example_2]

When the previous pipeline is applied to the challenge there where mainly two problems, sometimes one of the lanes wasn't detected, and sometimes the algorithm detected a line that wasn't a lane, it was a line in the road, or a shadow,... To solve this some modifications to the pipeline were made:

1. In the challenge, the right lane is yellow, but sometimes, when the color of the road is very light, the algorithm could not detect it. But it seems that Canny's algorithm also work with colors image, so if we remove the step that converts the the image to grayscale, the yellow color is different enough from its surroundings that its detected more easily. Also I increased the kernel size of the gaussian blur to 7, which helped detecting the lanes too. Another possible solution could have been to transform the yellow to white somehow.

2. There were a few cases when Hough's algorithm detected lines that weren't lanes, for example, when the pavement of the road change from an old one to a new one, it was detected as a line. To solve this, in the draw_lines() function, I only accepted lines with a slope between 20 and 70 degrees for the left lane, and 110 and 160 for the right lane. I also checked the point of the intersection of the line with the bottom of the image, ensuring that the point was valid. A possible improvement to this method could be to define the angle range as a function of the previous lanes, so if the car turns, we can know were the lanes are.

3. However, even with this changes, there were still some frames (less than 5) were the algorithm didn't detect some lane. What I did to solve this is to store the previous predictions, so if there is some frame where I didn't detect a lane, but I have enough information of were it should be, I can use that.

With this modifications the results in the challenge video were good enough.

This image shows one of the hardest frames to solve, because of the mixture of light, change of pavement, yellow lane and shadows:

![alt text][example_3]



### 2. Identify potential shortcomings with your current pipeline


One of the biggest shortcoming of this pipeline is that it doesn't work when the lanes don't have the expect form (slope and position), for example in a change of lane, where not only the slope of the lanes change, but you shoud also detect the three lanes that are involved in the manouver, the described pipeline would probably fail.

Another shortcoming is that in some roads, the lanes aren't painted, for example, in some country roads. In this case, the pipeline could probably predict the lines that separate the road, but there isn't always a clear separation.

There also the problem with curves, because although the lane detection algorithm more or less works in this situations, because the slope of the lanes changes, it doesn't detect the global curve, so you don't have enough information to know if the car needs to slow down, for example.

Also, the algorithm used could fail with discontinuous segments, if they are small enough that Hough's algorithm does not detect them.


### 3. Suggest possible improvements to your pipeline

One possible improvement already mentioned in the description of the pipeline that solves the challenge, is that because we know the detections in previous frames, we have an aproximation of where the lanes are in the next frame, so we can use this information to improve the detection.

Another improvement would be to not assume that the lanes are lines, we could at least use sets of segments in order to take curves into consideration. There are also variants of Hough's algorithm to detect curve that could be used. This would also help with discontinuous lines.
