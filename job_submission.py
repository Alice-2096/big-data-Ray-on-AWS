from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://3.221.127.122:8265")

job_id = client.submit_job(
    # Entrypoint shell command to execute
    entrypoint='python sample.py',
    # Runtime environment for the job, specifying a working directory and pip package
    runtime_env={
      'working_dir': '/Users/apple/Desktop/rayjob',
      'conda': 'env_pytorch'
    }
)