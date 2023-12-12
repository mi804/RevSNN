# train transformer architectures
python train.py -c configs/revsformer_2_384.yml
python train.py -c configs/revsformer_4_384.yml
python train.py -c configs/spikingformer_2_384.yml
python train.py -c configs/spikingformer_4_384.yml

# train resnets
python train_resnet_origin.py --model ms_resnet18 --T 4 --tb
python train_resnet_origin.py --model ms_resnet34 --T 4 --tb
python train_resnet_origin.py --model revs_resnet21 --T 4 --tb
python train_resnet_origin.py --model revs_resnet37 --T 4 --tb
