strategy='fedavg'
attack='label_flipping'

# download dataset
python3 utils/dl_dataset.py --dataset mnist

# generate partitions
python3 utils/partition_data.py --n_partitions 10 --dataset mnist

# start server and clients
python3 server.py --strategy $strategy --attack $attack &
sleep 3
python3 client.py --num 0 --attack 'none' &
python3 client.py --num 1 --attack 'none' &
python3 client.py --num 2 --attack $attack --seed 42 &
python3 client.py --num 3 --attack 'none'&
python3 client.py --num 4 --attack $attack --seed 50  &
python3 client.py --num 5 --attack 'none' &
python3 client.py --num 6 --attack $attack --seed 72 &
python3 client.py --num 7 --attack $attack --seed 80 &
python3 client.py --num 8 --attack 'none' &
python3 client.py --num 9 --attack 'none' 