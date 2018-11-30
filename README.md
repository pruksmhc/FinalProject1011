# FinalProject1011

FILES: 
Model1.py contains the first model
training.py cntains the code to run the models 

How batching will work: 
Given batch of size 32, with B1 being the source langauge texts, B2 being the targetlangauge text, and C1 being the indexing vector for B1 and C2 being the indexing vector for B2. 
We will first have the source language texts ordered from longest to shortest. Feed B1 into the encoder. 
For the decoder, we will pass in B2 and the encoder_hidden. Since encoder_hidden is still indexed via C1, we have to switch the indexing to C2 so that it matches with B2. Then pass in B2 and encoder_hidden into the decoder. Since we are feeding in a word at a time to the decoder, this would mean taking a vertical slide through the batch and feeding it into the decoder at each time step. 



Some hiccups: 
-The mathcing function get_index didn't entirely work b ecause sometime syou can have two vectors with 
same sum but are different. 