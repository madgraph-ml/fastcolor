#PBS -q a30
#PBS -l nodes=1:ppn=1:gpus=1:a30
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu02/marino
#PBS -o /remote/gpu02/marino/MadRecolor/madrecolor/output.txt
#PBS -e /remote/gpu02/marino/MadRecolor/madrecolor/error.txt
export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )