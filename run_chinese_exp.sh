N_ARGS=$#
for EXP in $* ;  
do  
CMD="./train.py +experiment=$EXP charset=chinese dataset=chinese model.img_size='[32,192]'"
echo $CMD;  
eval $CMD;
done  