# FinalProject1011

FILES: 
model_without_batching.py - this is the model that has everything

THe directory structure looks like this:
preprocessed_data_no_elmo/
	iwslt-vi-eng/

	iwslt-zh-eng/

iwslt-vi-en/ (this is what we got from Elman with stuff like train.tok.vi)
iwslt-zh-en/ (this is what we got from Elman with stuff like train.tok.vi)

run prepareDataInitial() to get all the data preprocessed, and then run trainIters() to train. 




How batching will work: 
Given batch of size 32, with B1 being the source langauge texts, B2 being the targetlangauge text, and C1 being the indexing vector for B1 and C2 being the indexing vector for B2. 
We will first have the source language texts ordered from longest to shortest. Feed B1 into the encoder. 
For the decoder, we will pass in B2 and the encoder_hidden. Since encoder_hidden is still indexed via C1, we have to switch the indexing to C2 so that it matches with B2. Then pass in B2 and encoder_hidden into the decoder. Since we are feeding in a word at a time to the decoder, this would mean taking a vertical slide through the batch and feeding it into the decoder at each time step. 


Some hiccups: 
-The mathcing function get_index didn't entirely work b ecause sometime syou can have two vectors with 
same sum but are different. 

Beam search:  https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/beam_search.py


Update: Alright, so I'm leaving batching just for now beause it might actually make things more messy. So right now, there's soemthing wrong - for Chinese-English, the BLEU score is around 0.0001. We might have to make it into a character-by-character model and also try bidirectional embedding. 

