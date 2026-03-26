#!/bin/bash
# submit_all.sh
# Submits all HAC-plus Pi3 training jobs.
# Run from the cluster after pi3_run.py and setup_hac_data.sh have completed.
#
# Usage: bash ~/Pi3/hac_jobs/submit_all.sh

JOBS_DIR="$(cd "$(dirname "$0")" && pwd)"

for SCRIPT in \
    bicycle bonsai garden kitchen \
    Truck Train Caterpillar Family Francis Horse Ignatius; do

    JOB="${JOBS_DIR}/${SCRIPT}.slurm"
    JOB_ID=$(sbatch --parsable "${JOB}")
    echo "Submitted ${SCRIPT} → job ${JOB_ID}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u nicnge21"
