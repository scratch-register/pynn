# pynn

This is a simple Python implementation of a neural network.

I created this for a school project and am uploading it for educational purposes. One day, I will clean it up a bit more and add additional notes...

This whole thing is currently a hot mess of spaghetti, with a great deal of extraneous code that will not make sense to most onlookers. I am working on cleaning it up to look a bit nicer, but the thought is that it is more useful to have _something_ rather than nothing for now, given that this is just for educational purposes.

To give some background on why this looks the way it does, the neural network was created to predict letters formed in sign language by a hand wearing a flex sensor-covered glove. The neural network was trained using this python script, and the weights were then transfered to an FPGA implementation of the neural net, which would run the predictions live as the glove wearer signed letters.

