## Which Library is Faster for DQN

This small script tests the performance of Pillow-SIMD vs OpenCV for Reinforcement Learning algorithms.

It tests two main features I use a lot in Deep Learning:   
* Resize images: we often reduce captured images size to speed up learning
* Convert them to grayscale for the same reason

Run the script from command line:
`$python whichisFaster.py --n 10000`

Where `--n` argument is the number of itterations. You can also add `--gray` to convert the image to gray scale as well as resizing it (it resizes first then converts to gray, you can change the order if you want!). 

Note: you can replace the photo used `img.jpg` as long as it has the same name

### Results Summary:
The results varied from one machine to another, but this is what i got:
Resize only:
>>> PIL is 1.6 times faster that OpenCV
Resiz then grayscale:
>>> PIL is 3.9 times faster that OpenCV

A run of 1 million on Google Cloud machine:
$ python whichIsFaster.py --n 1000000 --gray
>>>PIL is 4.6 times faster that OpenCV