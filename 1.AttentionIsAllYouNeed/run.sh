dataset="pdfs"
batch_size=128
epochs=5
learning_rate=0.0001
save_every=1

python train.py \
--dataset $dataset \
--batch_size $batch_size \
--epochs $epochs \
--learning_rate $learning_rate \
--save_every $save_every