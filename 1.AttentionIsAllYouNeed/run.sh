dataset="pdfs"
batch_size=64
epochs=30
learning_rate=0.001
save_every=10

python train.py \
--dataset $dataset \
--batch_size $batch_size \
--epochs $epochs \
--learning_rate $learning_rate \
--save_every $save_every