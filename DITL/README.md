This repo contains the code for DITL.

Usage
1. To run the code, you need to download the Wiki data from https://portals.mdi.georgetown.edu/public/demographic-inference
2. classifier.py
	--- Python script to train a model from scratch.
	--- Command: python classifier.py --inference_type age --bin_type three --epoch 100 --fix_seq_len 200
	--- inference_type: gender/age; bin_type: two/three/four for age only ;epoch: number of epochs to run; fix_seq_len: number of posts to use
3. classfier_reg.py
	--- Python script to train a model with regularization
	--- Command: python classifier.py --inference_type age --bin_type three --epoch 100 --fix_seq_len 200

