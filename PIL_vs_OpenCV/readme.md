## Which Library is Faster for DQN

This small script testing the performance of Pillow-SIMD vs OpenCV when it comes to Reinforcement Learning algorithms

It concentrates on two features I use a lot in Deep Learning: 
1- One is Resize: as we often reduce captured images (i.e. states)
2 - Convert images to grayscale to reduce overload

you can run the script from command line by typing:
`$python whichisFaster.py --n 10000`

Where `n` is the number of itterations. You can add `--gray` to convert to gray after resizing (you might want to try convert then resize as well). 

Note: you can either use the photo in this folder or replace it with another. Just make sure the name is `img.jpg`