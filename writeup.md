# **Finding Lane Lines on the Road** 

## Mohammed Amarnah

### A Writeup written for Udacity's Self Driving Car Nanodegree first project: Finding Lane Lines from a video stream.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road from a video stream
* Reflect on my work in a written report


[//]: # (Image References)


[image1]: ./test_images_output/old_outputs/solidWhiteCurve.png
[image2]: ./test_images_output/solidYellowCurve.jpg
[image3]: ./test_videos_output/all-results/curvedError.png

---

### Reflection

### 1. How I started and how draw_lines() were edited

My first step in this project was to redo all the quizzes in one function, just like we took them during the videos. Obviously color selection didn't work so well, so I just tried doing the Canny edge detection technique with the same parameters that we took them in the class. And I have to say, I got pretty good results for a first run. Here's how it looked like:

![alt text][image1]

To fix that, I did as suggested in the comment written in the draw_lines() function. I found the average slope of all hough lines of the hough transformation of the image. I found the average slope on the right lines and the left lines separately. I separated the lines based on their slope (if it was below 0 or above 0). Here is a sample output image: 

![alt text][image2]

### 2. Potential Shortcomings of the pipeline


The most potential shortcoming would be that it doesn't work on the challenge video. Meaning, it won't work on curved lane lines.

![alt text][image3]

Another shortcoming that I thought about is it working on shadows, night, and different lightning conditions. (but I haven't tested it yet so I'm not sure).


### 3. Possible improvements to the pipeline

A possible improvement for the curved lane lines problem would be to fit some good polynomial function. That thing I'll work on in the next few days, but I didn't have time to work on before the deadline.

For the problems that occur due to the lightning conditions, I have no particular idea on how to solve it, but it'd be some parameters tuning in the canny function I guess.