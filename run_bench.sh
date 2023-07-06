# crnn
# trba
# abinet
# vitstr
# parseq-tiny
# iterparseq-tiny
# iterparseq-tiny-d8
# iterparseq-tiny-d12
# iterparseq-tiny-d12i3
# iterparseq-tiny-d8i3
# iterparseq-tiny-d8i3-allshare
# iterparseq-tiny-d8-3enc
# parseq
# parseq-d24
# parseq-dim512
# parseq-patch4
# iterparseq-d6i4
# iterparseq-d8-3enc
# iterparseq-d8i3-allshare
# iterparseq-d6i3
# iterparseq-d12-2enc
# iterparseq-d8i3
# iterparseq-d12






N_ARGS=$#
for EXP in $* ;  
do  
CMD="./bench.py +experiment=$EXP model.max_label_length=5"
echo $CMD;  
eval $CMD;
CMD="./bench.py +experiment=$EXP charset=chinese dataset=chinese model.img_size='[32,192]' model.max_label_length=5"
echo $CMD;  
eval $CMD;
done  