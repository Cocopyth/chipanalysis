For running parallelized csv files for

sbatch --cpus-per-task=128 --mem 190000 batch_region_fluorescence.sh /scratch/bisot/ZeisData/ --output /home/bisot/shares/van_gestel_server/ZeissData/data_final

sbatch --cpus-per-task=128 --mem 190000 batch_make_videos.sh /scratch/bisot/ZeisData --output /home/bisot/shares/van_gestel_server/ZeissData/data_final
