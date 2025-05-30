#!/usr/bin/env bash
# shellcheck disable=SC1091,SC2050,SC2170

# using options from https://github.com/paboyle/Grid/tree/develop/systems/Tursa

#SBATCH -J gen_SkewMatDef
#SBATCH -A dp358-cpn
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --qos=dev
#SBATCH --no-requeue

set -e

# load environment #############################################################
source "/mnt/lustre/tursafs1/home/dp358/dp358/dc-mukh1/pyenvs/venv311/bin/activate"

# application and parameters ###################################################
app='/mnt/lustre/tursafs1/home/dp358/dp358/dc-mukh1/CPN/general_deformations.py'
opt=('--N=3' '--i=0' '--j=0' '--pidx=0' '--epochs=600')

# run! #########################################################################
python3 ${app} "${opt[@]}" --deftype=SkewMatDef
#for deftype in 'HomogDef' 'TorusDef' ProjDef' 'SkewMatDef' 'FlowDef'; do
#    python3 ${app} "${opt[@]}" --deftype=$deftype
#done

################################################################################
