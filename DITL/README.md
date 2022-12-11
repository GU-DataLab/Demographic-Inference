This repo contains the code for DITL.

Usage
1. To run the code, you need to download the Wiki data from https://portals.mdi.georgetown.edu/public/demographic-inference
2. Command: python classifier.py --inference_type gender --bin_type three --epoch 100 --fix_seq_len 200
	inference_type: gender/age
	bin_type: two/three/four for age only
	epoch: number of epochs to run
	fix_seq_len: number of posts to use

