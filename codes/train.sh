gpu='0'

exp='LA0'
data_dir='../dataset/LA/LA'
list_dir='../datalist/LA'

python train_JUDA.py  --gpu $gpu --data_dir $data_dir --list_dir $list_dir --exp $exp

# exp='Pancreas'
# data_dir='../dataset/Pancreas/pancreas_data'
# list_dir='../datalist/Pancreas'

# python train_JUDA.py  --gpu $gpu --data_dir $data_dir --list_dir $list_dir --exp $exp

# exp='AD_0'
# data_dir='../../../Datasets/TBAD128'
# list_dir='../datalist/AD/AD_0'

# python train_JUDA.py  --gpu $gpu --data_dir $data_dir --list_dir $list_dir --num_classes 3 --exp $exp