# crnn
# trba
# abinet
# vitstr
EXP=iterparseq-d8i3

for i in 1 2 3
do 
    for j in true false
    do
        for k in 0 1 2 3
        do
        CMD="./bench.py +experiment=$EXP model.refine_iters=$k model.decode_ar=$j  model.enc_num_iters=$i model.max_label_length=5"
        echo $CMD;  
        eval $CMD;
        CMD="./bench.py +experiment=$EXP model.refine_iters=$k model.decode_ar=$j  model.enc_num_iters=$i charset=chinese dataset=chinese model.img_size='[32,192]' model.max_label_length=5"
        echo $CMD;  
        eval $CMD;
        done
    done
done
