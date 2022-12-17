This repo contains the code for DITL.

Usage
1. To run the code, you need to 
	* download the Wiki data from https://portals.mdi.georgetown.edu/public/demographic-inference  
	* install librares
	```
	pip install imblearn
	```

2. We use some dummy examples here to help you start. This includes
	* a random generated IMDB dataset
		* imdb_gt.csv -- IMDB ground truth with fake username and gender
		* imdb_embeddings -- the embeddings for each user from the IMDB dataset
	* a random generated Wiki dataset
		* gt.csv -- wiki ground truth with fake username and gender
		* wiki_embeddings -- the embeddings for each user from the IMDB dataset
	* a random generated model
		* under the directory model/model
		
3. classifier.py  
	* Python script to train a model **from scratch**.  
	* Command: python classifier.py --inference_type xxx --bin_type xxx --epoch xxx --fix_seq_len xxx
	  	* inference_type: gender/age
		* bin_type: two/three/four, for age only
		* epoch: number of epochs to run
		* fix_seq_len: number of posts to use
	* To run the dummy example, execute 
		* python classifier.py --inference_type gender --bin_type three --epoch 100 --fix_seq_len 200
4. classfier_reg.py  
	* Python script to train a model with **regularization**  
	* Command: python classifier.py --inference_type age --bin_type three --epoch 100 --fix_seq_len 200  
	* inference_type
		* inference_type: gender/age
		* bin_type: two/three/four, for age only
		* epoch: number of epochs to run
		* fix_seq_len: number of posts to use
	* To run the dummy example, execute 
		* python classifier_reg.py --inference_type gender --bin_type three --epoch 100 --fix_seq_len 200
		
