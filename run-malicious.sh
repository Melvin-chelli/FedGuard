rm -r fl_logs/img/*
mkdir -p fl_logs/malicious/img

# download dataset
python3 utils/dl_dataset.py --dataset mnist

# Second case: without malicious updates
# generate partitions
python3 utils/partition_data.py --n_partitions 10 --dataset mnist --malicious

# start server and clients
python3 server.py &
sleep 3
python3 client.py --num 0 &
python3 client.py --num 3 --malicious &
python3 client.py --num 1 &
python3 client.py --num 2