# Look for the n architectures in the archive at iteration iter with the best trade-off between the first and second objectives

# dataset=cifar100 (change num classes accordingly)

first_obj=top1
sec_obj=avg_macs
iter=30
folder=results/layerskipping_cifar100_10e_conv_seed1

python3 post_search.py \
  --get_archive --n 100 \
  --save $folder/final \
  --expr $folder/iter_$iter.stats \
  --first_obj $first_obj --sec_obj $sec_obj 