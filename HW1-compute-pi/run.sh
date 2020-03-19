#!/bin/bash

numofexp='1'
ptile='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
home=`pwd`

for exp_id in $( seq 1 $numofexp )
do
	cd $home
	result_dir=q5_3_${exp_id}
	mkdir -p $result_dir
	cd $home/$result_dir
	for ptile_num in $ptile
        do
		sed -i "s/\<ptile=.[0-9]*\>/ptile=${ptile_num}/g" ../compute_pi_mpi.job2
		bsub < ../compute_pi_mpi.job2
		sleep 10
	done
done
