# Kyungun Cho said we should look at https://github.com/mjpost/sacreBLEU/blob/master/sacrebleu.py#L1022-L1080
# https://github.com/awslabs/sockeye/tree/master/sockeye_contrib/sacrebleu
import sacrebleu

def calculate_bleu(predictions, labels):
	"""
	Only pass a islt of strings for both in english for our ase. 
	"""

	bleu = sacrebleu.raw_corpus_bleu(predictions, [labels], .01).score
	return bleu

calculate_bleu(["test"], ["test"])
# shoud lbe 100