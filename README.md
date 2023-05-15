

### Detecting malicious updates with CVAEs in FL scenarios

This repository contains the source code used for the experiments evaluating FedGuard.
The repository is organized as follows:
* *experiments_logs/* contains tensorboard logs of the experiments
* *scenarios/grid5000/* contains configuration files used to deploy the experiments with E2Clab
* *strategies/* contains the implementation of FedGuard (MaliciousUpdateDetectionStrategy),  Spectral, GeoMed (FedMedian) and Krum.
* *utils/* contains various utility functions

**Downloading the dataset**
```shell
python3 ./utils/dl_dataset.py --dataset mnist
```

**Partitioning the dataset**
```shell
python3 ./utils/partition_data.py --n_partitions 100 --dataset mnist --alpha 10
```

**Starting the server**
```shell
python3 server.py
	  --server_address {{ _self.url }}:8080
	  --attack {{ attack }}
	  --strategy {{ strategy }}
	  --model {{ model }}
	  --num_rounds {{ num_rounds }}
	  --min_available_clients {{ min_available_clients }}
	  --min_fit_clients {{ min_fit_clients }}
	  --fraction_fit {{ fraction_fit }}
	  --server_lr {{ server_lr }}
	  --server_momentum {{ server_momentum }}
	  --local_epochs {{ local_epochs }}
	  --cvae_local_epochs {{ cvae_local_epochs }}
	  --n_decoders_to_sample {{ n_decoders_to_sample }}
	  --n_evaluation_data_per_decoder {{ n_evaluation_data_per_decoder }}
```

**Starting a client**
```shell
python3 client.py
	  --server_address {{ _self.url }}:8080
	  --model {{ model }}
	  --strategy {{ strategy }}
	  --num {{ partition_number }}
	  --attack {{ attack }}
```

**List of attacks**
* sign_flipping
* same_value
* label_flipping
* additive_noise

**List of strategy**
* detection_strategy (FedGuard)
* spectral
* fedmedian (GeoMed)
* krum
* fedavg
