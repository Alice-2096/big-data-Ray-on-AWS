ray up -y config.yaml
ray job submit --working-dir ./rayjob  -- python sample.py

# to delete the cluster
# ray down config.yaml