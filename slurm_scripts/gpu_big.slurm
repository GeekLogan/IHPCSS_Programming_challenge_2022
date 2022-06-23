#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --time=00:01:30
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --job-name=IHPCSS
#SBATCH --output=error.txt
#SBATCH --error=error.txt
#SBATCH --gres=gpu:v100:8
#SBATCH --reservation=challengegpu
#SBATCH --account=tra210016p

# OpenMP placement
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=sockets

#export PGI_ACC_TIME=1

# Run the binary
module load nvhpc/21.2
pgaccelinfo
mpirun -np ${SLURM_NTASKS} --map-by socket -bind-to socket -report-bindings ./bin/${1}/gpu_big > $2
