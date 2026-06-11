For running parallelized csv files for

sbatch --cpus-per-task=128 --mem 190000 batch_region_fluorescence.sh /scratch/bisot/ZeisData/ --output /home/bisot/shares/van_gestel_server/ZeissData/data_final

sbatch --cpus-per-task=128 --mem 190000 batch_make_videos.sh /scratch/bisot/ZeisData --output /home/bisot/shares/van_gestel_server/ZeissData/data_final


python chipanalysis/scripts/batch_chip_dynamics.py     --excel  /home/bisot/shares/van_gestel_server/ZeissData/ICP/process_info.xlsx     --config chipanalysis/scripts/config_chip_dynamics_template.yaml     --output /home/bisot/shares/van_gestel_server/ZeissData/ICP/results     --row    0 

// ...existing code...

# Make log directory first
mkdir -p logs

# Count how many rows are in the manifest (= number of array tasks - 1)
N=$(python chipanalysis/scripts/batch_chip_dynamics.py \
    --excel  /home/bisot/shares/van_gestel_server/ZeissData/ICP/process_info.xlsx \
    --config chipanalysis/scripts/config_chip_dynamics_template.yaml \
    --output /home/bisot/shares/van_gestel_server/ZeissData/ICP/results \
    --count-rows)

echo "Submitting array 0-${N}"

# Submit — %4 means at most 4 jobs running at once (adjust to your quota)
sbatch --array=0-${N}%4 \
    chipanalysis/scripts/batch_chip_dynamics.sh \
    --excel  /home/bisot/shares/van_gestel_server/ZeissData/ICP/process_info.xlsx \
    --config chipanalysis/scripts/config_chip_dynamics_template.yaml \
    --output /home/bisot/shares/van_gestel_server/ZeissData/ICP/results