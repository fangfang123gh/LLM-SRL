# Should point to the srlconll library.
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=C

SRLPATH="/HOME/hitsz_mszhang/hitsz_mszhang_1/HDD_POOL/LLM_SRL/LLM-SRL"

export PERL5LIB="$SRLPATH/srlconll-1.1/lib:$PERL5LIB"
export PATH="$SRLPATH/srlconll-1.1/bin:$PATH"

perl $SRLPATH/srlconll-1.1/bin/srl-eval.pl $1 $2
# perl $SRLPATH/srlconll-1.1/bin/srl-eval.pl /HOME/hitsz_mszhang/hitsz_mszhang_1/gold_tgt_temp_file38 /HOME/hitsz_mszhang/hitsz_mszhang_1/gold_tgt_temp_file38

