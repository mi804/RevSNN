# train transformer architectures
python train.py --T 16 --tb --amp --model RevSpikingformer 2>&1 |  tee logs/RevSpikingformer_T16.log
python train.py --T 10 --tb --amp --model RevSpikingformer 2>&1 |  tee logs/RevSpikingformer_T10.log
python train.py --T 16 --tb --amp 2>&1 |  tee logs/Spikingformer_T16.log
python train.py --T 10 --tb --amp 2>&1 |  tee logs/Spikingformer_T10.log

# train resnets
python train_resnet.py --T 16 --tb --amp --weight-decay 0. --model revs_resnet24 2>&1 |  tee logs/lif/revs_resnet24_T16_wd0.log
python train_resnet.py --T 16 --tb --amp --weight-decay 0. --model ms_resnet20 2>&1 |  tee logs/lif/ms_resnet20_T16_wd0.log
python train_resnet.py --T 10 --tb --amp --weight-decay 0. --model revs_resnet24 2>&1 |  tee logs/lif/revs_resnet24_T10_wd0.log
python train_resnet.py --T 10 --tb --amp --weight-decay 0. --model ms_resnet20 2>&1 |  tee logs/lif/ms_resnet20_T10_wd0.log
