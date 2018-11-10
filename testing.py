# Kyungun Cho said we should look at https://github.com/mjpost/sacreBLEU/blob/master/sacrebleu.py#L1022-L1080
# https://github.com/awslabs/sockeye/tree/master/sockeye_contrib/sacrebleu

corpus_bleu(sys_stream, ref_streams, smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
                tokenize=DEFAULT_TOKENIZER, use_effective_order=False) -> BLEU

