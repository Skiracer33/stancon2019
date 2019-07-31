#!/bin/bash

# This script submits all the experiments
# option d is the directory for the results
# option s is number of samples to produce
# option b is for sbatch


while getopts s:d:b: option
do
case "${option}"
in
s) samples=${OPTARG};;
d) d=${OPTARG};;
b) b=${OPTARG};;
esac
done

if [ -z "$d" ]
then
echo "d not set"
exit 1
fi

if [ -z "$samples" ]
then
echo "c not set"
exit
fi


echo directory $d
echo samples $samples
mkdir $d
DATE=`date '+%m%d%H%M'`


if [ -z "$b" ]
then
    for i in {0..5}
        do 
            ./submit_compute.sh $i $samples $d &
        done
    exit
else
    for i in {0..5}
        do 
            sbatch -o $d/"$i""slurm""$DATE".out"" submit_compute.sh $i $samples $d    
        done
fi
