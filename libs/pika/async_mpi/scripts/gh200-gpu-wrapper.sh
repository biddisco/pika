#!/usr/bin/env bash

cpus=$(taskset -pc $$ | awk '{print $6}')
numa_nodes=$(hwloc-calc --physical --intersect NUMAnode $(taskset -p $$ | awk '{print "0x"$6}'))
echo "numa_nodes $numa_nodes"
IFS=',' read -r first_node other_nodes <<< "$numa_nodes"

gpus=$numa_nodes

lrank=0
grank=0
if [ -z ${OMPI_COMM_WORLD_LOCAL_RANK+x} ]
then
    let lrank=$SLURM_LOCALID
    let grank=$SLURM_PROCID
else
    let lrank=$OMPI_COMM_WORLD_LOCAL_RANK
    let grank=$OMPI_COMM_WORLD_RANK
fi
first_nic="cxi${first_node}"

if [[ $grank == 0 ]]
then
    echo "Slurm Job Hostlist: $SLURM_JOB_NODELIST"
fi
echo "Hostname: $(hostname) Rank: $grank, Local $lrank, GPUs $gpus, CPUs $cpus, NIC $first_nic"

unset LOCAL_RANK

# ----------------------------------------------
# NVIDIA : explicit nvidia settings 
export NVCOMPILER_ACC_DEFER_UPLOADS=1
export NVCOMPILER_ACC_USE_GRAPH=1
export NV_ACC_CUDA_MEMALLOCASYNC=1
export NV_ACC_CUDA_MEMALLOCASYNC_POOLSIZE=500000000000

# set correct device to use per mpi rank
export CUDA_VISIBLE_DEVICES=$gpus

# ----------------------------------------------
# LIBFABRIC: set correct libfabric network device to use per mpi rank
export FI_CXI_DEVICE_NAME=$first_nic
# Defines the maximum CPU memcpy size for HMEM device memory that is accessible by the CPU with load/store operations
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0

# ----------------------------------------------
# OPENMPI : set byte trsnsport layer to ofi
#export OMPI_MCA_btl_ofi_mode=2
#export OMPI_MCA_pml_ob1_max_rdma_per_request=1

"$@"