# Deploying to AWS (ECS Fargate) — notes, not yet executed

I have **not** deployed this. These are the steps I would follow; run them and note the real URL before claiming it is deployed.

1. Create an ECR repo and push the image:
   `aws ecr create-repository --repository-name ragguard`
   `docker build -t ragguard . && docker tag ragguard <acct>.dkr.ecr.<region>.amazonaws.com/ragguard:latest`
   `aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com`
   `docker push <acct>.dkr.ecr.<region>.amazonaws.com/ragguard:latest`
2. Create an ECS cluster and a Fargate task definition (0.25 vCPU / 0.5 GB is enough for the extractive backend), container port 8000, env `RAGGUARD_BACKEND=extractive`.
3. Put an Application Load Balancer in front, health check path `/health`.
4. For the `anthropic` backend, store `ANTHROPIC_API_KEY` in Secrets Manager and inject it as a task secret. Never bake it into the image.
5. Add a note to the README with the live URL and an example `curl`.
